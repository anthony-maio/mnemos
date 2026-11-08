"""Experimental Git configuration dependency discovery; does not enable caching."""

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess

from .project_identity_cache import _entry


class PolicyUnavailable(ValueError):
    """The caller must use authoritative Git, without reusing an identity."""


@dataclass(frozen=True)
class ConfigDependencies:
    paths: tuple[Path, ...]
    observations: tuple[tuple[object, ...], ...]


def observe_config_path(path: Path) -> tuple[object, ...]:
    # Do not resolve symlinks away before inspecting every path component.
    return tuple(_entry(p) for p in (*reversed(path.parents), path))


def discover_config_dependencies(
    root: Path, executable: Path, environment: dict[str, str]
) -> ConfigDependencies:
    """Ask Git for roots and parse every potential include, even inactive ones.

    This deliberately over-approximates conditional dependencies. Unsupported
    observations raise, rather than silently omitting a configuration source.
    The owner must separately bind executable, environment and routing state.
    """
    root, executable = Path(root), Path(executable)
    if not root.is_absolute() or not executable.is_absolute():
        raise PolicyUnavailable("Absolute root and executable required")
    env = dict(environment)
    if any("\n" in value or "\r" in value for value in env.values()):
        raise PolicyUnavailable("Multiline environment cannot identify unambiguous paths")
    # Runtime config may itself inject includes. No partial traversal is safe.
    if any(
        key.upper() == "GIT_CONFIG_PARAMETERS"
        or key.upper().startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
        or (key.upper() == "GIT_CONFIG_COUNT" and value not in {"", "0"})
        for key, value in env.items()
    ):
        raise PolicyUnavailable("Runtime configuration overrides need authoritative Git")
    for key in list(env):
        if key.upper() in {
            "GIT_DIR",
            "GIT_COMMON_DIR",
            "GIT_WORK_TREE",
            "GIT_CEILING_DIRECTORIES",
            "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        }:
            del env[key]
    env["LC_ALL"] = "C"

    def command(*args: str, missing_ok: bool = False) -> str:
        try:
            result = subprocess.run(
                [str(executable), "-C", str(root), *args],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=5,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PolicyUnavailable("Git policy discovery failed") from exc
        if missing_ok and result.returncode == 1 and not result.stdout and not result.stderr:
            return ""
        if result.returncode != 0 or len(result.stdout) > 262144:
            raise PolicyUnavailable("Git policy discovery was incomplete")
        return result.stdout.decode("utf-8", errors="strict")

    def absolute(value: str, parent: Path) -> Path:
        if not value or any(c in value for c in "\r\n\x00"):
            raise PolicyUnavailable("Unsupported configuration path")
        path = Path(value)
        if ".." in path.parts:
            # Lexically removing '..' can hide a preceding symlink traversal.
            raise PolicyUnavailable("Parent traversal requires authoritative Git")
        # abspath normalizes dot segments without following symlinks.
        return Path(os.path.abspath(path if path.is_absolute() else parent / path))

    roots = []
    for name in ("GIT_CONFIG_SYSTEM", "GIT_CONFIG_GLOBAL"):
        roots.extend(absolute(p, root) for p in command("var", name, missing_ok=True).splitlines())
    local = command(
        "rev-parse",
        "--path-format=absolute",
        "--git-path",
        "config",
        "--git-path",
        "config.worktree",
    ).splitlines()
    if len(local) != 2:
        raise PolicyUnavailable("Git did not identify both repository configuration paths")
    roots.extend(absolute(p, root) for p in local)
    found: dict[Path, tuple[object, ...]] = {}

    def visit(path: Path, ancestors: frozenset[Path]) -> None:
        if path in ancestors or len(ancestors) > 16:
            raise PolicyUnavailable("Cyclic or excessive include depth")
        if path in found:
            return
        if len(found) >= 64:
            raise PolicyUnavailable("Too many configuration dependencies")
        before = observe_config_path(path)
        found[path] = before
        entry = before[-1]
        if entry[-1] == "absent":
            return
        if not isinstance(entry[-1], bytes):
            raise PolicyUnavailable("Configuration is not a regular file")
        output = command(
            "config",
            "--no-includes",
            "--null",
            "--file",
            str(path),
            "--type=path",
            "--get-regexp",
            r"^(include\.path|includeif\..*\.path)$",
            missing_ok=True,
        )
        if output and not output.endswith("\x00"):
            raise PolicyUnavailable("Incomplete include records")
        for record in output.split("\x00")[:-1]:
            key, separator, value = record.partition("\n")
            if not separator or not (key == "include.path" or key.startswith("includeif.")):
                raise PolicyUnavailable("Invalid include record")
            visit(absolute(value, path.parent), ancestors | {path})
        if observe_config_path(path) != before:
            raise PolicyUnavailable("Configuration changed during parsing")

    try:
        for path in roots:
            visit(path, frozenset())
        if any(observe_config_path(p) != value for p, value in found.items()):
            raise PolicyUnavailable("Configuration changed during discovery")
    except (OSError, UnicodeError) as exc:
        raise PolicyUnavailable("Configuration dependency cannot be inspected") from exc
    return ConfigDependencies(tuple(found), tuple(found.values()))
