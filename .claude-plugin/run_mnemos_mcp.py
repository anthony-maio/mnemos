#!/usr/bin/env python3
"""Resolve the managed Mnemos runtime and launch its stdio MCP server.

The plugin wrapper is intentionally small because Claude Code starts it as a
stdio server. Bootstrap progress is sent to stderr so stdout stays a valid
MCP stream. Runtime defaults remain owned by ``mnemos.settings`` and
``mnemos.runtime``.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import venv
from typing import Any, Iterator


def _plugin_root() -> Path:
    env_root = os.environ.get("CLAUDE_PLUGIN_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[1]


def _venv_python(venv_dir: Path) -> Path:
    if _is_windows():
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _required_install_extras(env: dict[str, str]) -> tuple[str, ...]:
    _ = env
    return ("mcp",)


def _source_digest(plugin_root: Path) -> str:
    """Hash the source that the local install command consumes."""
    candidates: list[Path] = []
    pyproject = plugin_root / "pyproject.toml"
    if pyproject.is_file():
        candidates.append(pyproject)
    package_root = plugin_root / "mnemos"
    if package_root.is_dir():
        candidates.extend(
            path
            for path in package_root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    agents_root = plugin_root / "agents"
    if agents_root.is_dir():
        candidates.extend(
            path
            for path in agents_root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    digest = hashlib.sha256()
    for path in sorted(candidates):
        relative = path.relative_to(plugin_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _expected_install_stamp(plugin_root: Path, env: dict[str, str]) -> str:
    payload = {
        "schema": 2,
        "source_sha256": _source_digest(plugin_root),
        "extras": list(_required_install_extras(env)),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@contextmanager
def _runtime_lock(lock_path: Path) -> Iterator[None]:
    """Serialize venv creation and installation across hook/MCP processes."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if _is_windows():
            import msvcrt

            for _ in range(600):
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                raise TimeoutError(f"Timed out acquiring runtime lock: {lock_path}")
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - exercised on POSIX plugin hosts
            import fcntl

            fcntl_module: Any = fcntl
            deadline = time.monotonic() + 60.0
            while True:
                try:
                    fcntl_module.flock(handle.fileno(), fcntl_module.LOCK_EX | fcntl_module.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out acquiring runtime lock: {lock_path}")
                    time.sleep(0.1)
            try:
                yield
            finally:
                fcntl_module.flock(handle.fileno(), fcntl_module.LOCK_UN)


def _create_venv(venv_dir: Path) -> None:
    builder = venv.EnvBuilder(with_pip=True, clear=False)
    builder.create(venv_dir)


def _install_plugin(venv_python: Path, plugin_root: Path, env: dict[str, str]) -> None:
    extras = ",".join(_required_install_extras(env))
    commands = (
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            f"{plugin_root}[{extras}]",
        ],
    )
    for command in commands:
        subprocess.run(
            command,
            stdout=sys.stderr,
            stderr=sys.stderr,
            check=True,
        )


def _write_stamp(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _ensure_runtime(venv_dir: Path, plugin_root: Path, env: dict[str, str]) -> Path:
    lock_path = venv_dir / ".mnemos-runtime.lock"
    stamp_path = venv_dir / ".mnemos-install-stamp"
    with _runtime_lock(lock_path):
        expected = _expected_install_stamp(plugin_root, env)
        venv_python = _venv_python(venv_dir)
        current = stamp_path.read_text(encoding="utf-8").strip() if stamp_path.exists() else ""
        python_missing = not venv_python.exists()
        if python_missing:
            _create_venv(venv_dir)
        if not venv_python.exists():
            raise RuntimeError(f"Runtime Python was not created at {venv_python}")
        if python_missing or current != expected:
            _install_plugin(venv_python, plugin_root, env)
            _write_stamp(stamp_path, expected)
        return venv_python


def _resolve_runtime_python(plugin_root: Path, env: dict[str, str]) -> str:
    explicit_python = env.get("MNEMOS_PLUGIN_PYTHON", "").strip()
    if explicit_python:
        return explicit_python

    venv_dir = plugin_root / ".claude-plugin" / ".venv"
    return str(_ensure_runtime(venv_dir, plugin_root, env))


def _ready_runtime_python(plugin_root: Path, env: dict[str, str]) -> str | None:
    """Return the runtime only if it is already installed for this source."""
    explicit_python = env.get("MNEMOS_PLUGIN_PYTHON", "").strip()
    if explicit_python:
        return explicit_python
    venv_dir = plugin_root / ".claude-plugin" / ".venv"
    venv_python = _venv_python(venv_dir)
    try:
        current = (venv_dir / ".mnemos-install-stamp").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not venv_python.exists() or current != _expected_install_stamp(plugin_root, env):
        return None
    return str(venv_python)


def _is_windows() -> bool:
    return os.name == "nt"


def _apply_default_env(plugin_root: Path) -> dict[str, str]:
    """Preserve explicit settings and let the managed runtime own defaults."""
    _ = plugin_root
    env = dict(os.environ)
    # This wrapper is the Claude Code plugin entrypoint.  The host identity is
    # integration-owned and must not inherit a stale manual or other-host value.
    env["MNEMOS_HOST"] = "claude-code"
    return env


def main() -> int:
    plugin_root = _plugin_root()
    env = _apply_default_env(plugin_root)
    try:
        runtime_python = _resolve_runtime_python(plugin_root, env)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        # Offline pip, a failed venv, or another session holding the install
        # lock. No stamp is written, so the next server start tries again.
        print(
            f"Mnemos runtime is not installed ({type(exc).__name__}); "
            "the MCP server retries the install when it next starts",
            file=sys.stderr,
        )
        return 1

    completed = subprocess.run(
        [runtime_python, "-m", "mnemos.mcp_server"],
        env=env,
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    sys.exit(main())
