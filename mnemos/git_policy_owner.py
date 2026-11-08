"""Inactive resident-owner prototype for observable Git policy freshness."""

import hashlib
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import stat

from .git_policy_dependencies import discover_config_dependencies, observe_config_path
from .project_identity_cache import (
    _entry,
    _routing_snapshot,
    _security_identity,
    _unsupported_git_environment,
)


@dataclass(frozen=True)
class PinnedGitRuntime:
    """Reviewed manifest, not automatic approval of discovered dependencies."""

    selected: Path
    hashes: tuple[tuple[Path, str], ...]
    directories: tuple[tuple[Path, tuple[str, ...]], ...]

    def __post_init__(self):
        files = dict(self.hashes)
        directories = dict(self.directories)
        if (
            not 1 <= len(files) <= 128
            or len(files) != len(self.hashes)
            or self.selected not in files
            or not 1 <= len(directories) <= 128
            or len(directories) != len(self.directories)
        ):
            raise ValueError("Incomplete or duplicate runtime manifest")
        for path in (self.selected, *files, *directories):
            if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
                raise ValueError("Runtime paths must be absolute without parent traversal")
        if any(path.parent not in directories for path in files):
            raise ValueError("Runtime file parents require directory inventories")
        for digest in files.values():
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("Expected SHA256 runtime binding")
        for names in directories.values():
            if (
                not isinstance(names, tuple)
                or len(names) > 4096
                or names != tuple(sorted(set(names)))
            ):
                raise ValueError("Expected bounded sorted unique directory inventory")

    def validate(self, executable, observations):
        observed = {Path(row[0]): row[-1] for row in observations}
        if executable != self.selected or observed != dict(self.hashes):
            raise ValueError("Git runtime differs from reviewed binding")
        for directory, names in self.directories:
            if tuple(sorted(p.name for p in directory.iterdir())) != names:
                raise ValueError("Git runtime directory changed")


def _executable_observation(path: Path) -> tuple[object, ...]:
    parents = tuple(_entry(p) for p in reversed(path.parents))
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or getattr(before, "st_file_attributes", 0) & 0x400
        or before.st_size > 128 * 1024 * 1024
    ):
        raise ValueError("Unsupported executable identity")
    security = _security_identity(path, before)
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.lstat()

    def identity(info):
        return (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    # Windows fstat reports different mode/ctime semantics from path stat.
    # Compare the open handle's file identity/content metadata, and compare
    # all path metadata before versus after the read separately.
    def handle_identity(info):
        return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), info.st_size, info.st_mtime_ns)

    if handle_identity(before) != handle_identity(opened) or identity(before) != identity(after):
        raise ValueError("Executable changed while reading")
    if security != _security_identity(path, after):
        raise ValueError("Executable security changed while reading")
    return (str(path), parents, identity(after), security, digest)


def _home_selection_observation(environment):
    """Observe the filesystem input to the Windows wrapper's HOME fallback."""
    if os.name != "nt" or environment.get("HOME") or not environment.get("HOMEPATH"):
        return ()
    candidate = Path(environment.get("HOMEDRIVE", "") + environment["HOMEPATH"])
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Unsupported Windows home selection path")
    # Watch absent paths too: creation can switch Git away from USERPROFILE.
    # Parent observations reject aliases and observe access metadata changes.
    return observe_config_path(candidate)


class GitPolicyOwner:
    """One root, serialized calls, no authorization or persistent policy records.

    runtime_files must be the owner's reviewed installation dependencies in
    addition to the selected executable. This helper does not discover a
    wrapper's child executable or loaded libraries. It must not be integrated
    until the installation owner supplies and verifies that closure.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        runtime_files: tuple[Path, ...] = (),
        runtime_binding: PinnedGitRuntime | None = None,
    ) -> None:
        if runtime_binding is not None and runtime_files:
            raise ValueError("Use a pinned binding or unpinned experimental files, not both")
        self.root = Path(root).absolute()
        self.runtime_binding = runtime_binding
        self.runtime_files = (
            tuple(p for p, _ in runtime_binding.hashes)
            if runtime_binding is not None
            else tuple(Path(p).absolute() for p in runtime_files)
        )
        self._context = None
        self._dependencies = None

    def _observe_context(self):
        if _unsupported_git_environment():
            raise ValueError("Unsupported Git selector")
        environment = dict(os.environ)
        selected = shutil.which("git")
        if selected is None:
            raise ValueError("Git executable unavailable")
        executable = Path(os.path.abspath(selected))
        binaries = tuple(dict.fromkeys((executable, *self.runtime_files)))
        if len(binaries) > 128:
            raise ValueError("Runtime dependency limit exceeded")
        observations = tuple(_executable_observation(p) for p in binaries)
        if self.runtime_binding is not None:
            self.runtime_binding.validate(executable, observations)
        routing = _routing_snapshot(self.root)
        home_selection = _home_selection_observation(environment)
        return (tuple(sorted(environment.items())), observations, routing, home_selection), executable, environment

    def probe(self) -> str | None:
        """Return a generation only when all declared observations are stable."""
        try:
            context, executable, environment = self._observe_context()
            dependencies = self._dependencies
            if (
                context != self._context
                or dependencies is None
                or any(
                    observe_config_path(path) != observation
                    for path, observation in zip(
                        dependencies.paths, dependencies.observations, strict=True
                    )
                )
            ):
                dependencies = discover_config_dependencies(self.root, executable, environment)
            after, _, _ = self._observe_context()
            if after != context or any(
                observe_config_path(path) != observation
                for path, observation in zip(
                    dependencies.paths, dependencies.observations, strict=True
                )
            ):
                raise ValueError("Policy changed across observation")
            self._context, self._dependencies = context, dependencies
            # Environment and configuration contents are never written to disk.
            return hashlib.sha256(repr((context, dependencies)).encode("utf-8")).hexdigest()
        except Exception:
            self._context = self._dependencies = None
            return None
