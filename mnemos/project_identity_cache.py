"""Experimental, explicitly owned cache of Git project routing, not authorization.

Not installed in adapters. Callers must still reload and validate registry state.
Global/system Git policy invalidation must be supplied as a generation value by
the owner; without one this resolver always delegates to authoritative Git.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
from typing import Callable

from .scopes import canonical_project_root


def _security_identity(path: Path, info: os.stat_result) -> object:
    """Read current ownership/access metadata; never reuse an ACL observation."""
    if os.name != "nt":
        return (info.st_uid, info.st_gid, info.st_mode)
    import ctypes
    from ctypes import wintypes

    api = ctypes.WinDLL("advapi32", use_last_error=True).GetFileSecurityW
    api.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    api.restype = wintypes.BOOL
    size = wintypes.DWORD()
    # OWNER | GROUP | DACL. The buffer is a self-relative descriptor, including
    # inheritance/protection flags. Reading it requires no SACL privilege.
    api(str(path), 7, None, 0, ctypes.byref(size))
    error = ctypes.get_last_error()
    if error != 122 or not 20 <= size.value <= 1_048_576:
        raise OSError(error, "Cannot snapshot project security descriptor")
    buffer = ctypes.create_string_buffer(size.value)
    if not api(str(path), 7, buffer, len(buffer), ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not 20 <= size.value <= len(buffer):
        raise ValueError("Invalid project security descriptor size")
    return buffer.raw[: size.value]


def _unsupported_git_environment() -> bool:
    # Configuration overrides belong to the caller's external-policy generation.
    # These routing overrides are explicitly stripped by canonical_project_root.
    # Any other Git selector may name mutable resources outside our snapshot.
    known = {
        "GIT_DIR",
        "GIT_COMMON_DIR",
        "GIT_WORK_TREE",
        "GIT_CEILING_DIRECTORIES",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
    }
    for key in os.environ:
        upper = key.upper()
        if not upper.startswith("GIT_") or upper in known:
            continue
        if any(
            upper.startswith(prefix) and upper[len(prefix) :].isdigit()
            for prefix in ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")
        ):
            continue
        return True
    return False


def _entry(path: Path) -> tuple[object, ...]:
    """Read content for files; directory identity excludes ordinary child edits."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return (str(path), "absent")
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("Reparse points require authoritative resolution")
    identity = (str(path), info.st_dev, info.st_ino, info.st_mode, _security_identity(path, info))
    if stat.S_ISDIR(info.st_mode):
        return (*identity, "directory")
    if not stat.S_ISREG(info.st_mode) or info.st_size > 65_536:
        raise ValueError("Unsupported project routing file")
    with path.open("rb") as handle:
        content = handle.read(65_537)
    if len(content) > 65_536:
        raise ValueError("Oversized project routing file")
    return (*identity, content)


def _routing_snapshot(root: Path) -> tuple[object, ...]:
    entries: list[object] = []
    directories: set[Path] = set()
    # Include every ancestor: adding a nearer repository invalidates a hit.
    for parent in (root, *root.parents):
        entries.append(_entry(parent))
        marker = parent / ".git"
        entries.append(_entry(marker))
        for name in ("HEAD", "objects", "commondir"):
            entries.append(_entry(parent / name))
        if marker.is_dir():
            directories.add(marker)
        elif marker.is_file():
            raw = marker.read_text(encoding="utf-8").strip()
            if not raw.startswith("gitdir: ") or "\n" in raw:
                raise ValueError("Unsupported Git directory pointer")
            directories.add((parent / raw[8:]).resolve(strict=True))
        if (parent / "HEAD").is_file() and (parent / "objects").is_dir():
            directories.add(parent)
    # A linked worktree's common directory is part of the routing boundary.
    for directory in list(directories):
        common = directory / "commondir"
        entries.append(_entry(common))
        if common.is_file():
            raw = common.read_text(encoding="utf-8").strip()
            if not raw or "\n" in raw:
                raise ValueError("Unsupported Git common directory pointer")
            directories.add((directory / raw).resolve(strict=True))
    for directory in sorted(directories):
        entries.append(_entry(directory))
        for name in ("HEAD", "objects", "refs", "commondir", "gitdir", "config", "config.worktree"):
            entries.append(_entry(directory / name))
    return tuple(entries)


class ProjectIdentityCache:
    """One-entry cache with caller-owned external-policy generation.

    A generation must cover Git executable/configuration policy, including
    includes and system/global config. None disables caching. Cache entries
    contain identity only, never project enablement or workspace membership.
    Owners serialize calls. Any failed refresh discards the previous entry.
    A live policy_probe samples external freshness before reuse and after
    resolution. The legacy generation argument cannot detect an external
    change during a call and is not sufficient for a mutable-policy owner.
    """

    def __init__(self, resolver: Callable[[str | Path], str] = canonical_project_root) -> None:
        self._resolver = resolver
        self._cached: tuple[object, str] | None = None

    def resolve(
        self,
        path: str | Path,
        *,
        policy_generation: str | None = None,
        policy_probe: Callable[[], str | None] | None = None,
    ) -> str:
        try:
            if policy_probe is not None and policy_generation is not None:
                raise ValueError("Supply a live policy probe or a generation, not both")
            root = Path(path).expanduser().resolve(strict=True)
            if not root.is_dir():
                raise ValueError("A project root must be a directory")

            def current_policy() -> str | None:
                if policy_probe is None:
                    return policy_generation
                try:
                    value = policy_probe()
                except Exception:
                    return None
                return value if isinstance(value, str) and value.strip() else None

            generation = current_policy()
            if not generation or _unsupported_git_environment():
                self._cached = None
                return self._resolver(root)
            executable = shutil.which("git")
            if executable is None:
                self._cached = None
                return self._resolver(root)
            executable_info = Path(executable).stat()
            execution = (
                executable,
                executable_info.st_dev,
                executable_info.st_ino,
                executable_info.st_size,
                executable_info.st_mtime_ns,
            )
            environment = tuple(sorted(os.environ.items()))
            try:
                before = _routing_snapshot(root)
            except (OSError, ValueError, UnicodeError):
                self._cached = None
                return self._resolver(root)
            key = (str(root), generation, execution, environment, before)
            if self._cached is not None and self._cached[0] == key:
                if current_policy() == generation and environment == tuple(
                    sorted(os.environ.items())
                ):
                    return self._cached[1]
                # Policy changed during hit validation. Resolve authoritatively,
                # but do not publish a cache entry from this mixed observation.
                self._cached = None
                return self._resolver(root)
            self._cached = None
            result = self._resolver(root)
            # Do not cache a result obtained while routing was changing.
            try:
                after = _routing_snapshot(root)
            except (OSError, ValueError, UnicodeError):
                return result
            after_generation = current_policy()
            if (
                before == after
                and environment == tuple(sorted(os.environ.items()))
                and after_generation == generation
            ):
                self._cached = (key, result)
            return result
        except BaseException:
            self._cached = None
            raise
