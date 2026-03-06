#!/usr/bin/env python3
"""
Bootstraps a local venv for mnemos and runs the MCP server.

This wrapper makes the plugin self-contained for Claude Code installs:
- Creates `.claude-plugin/.venv` on first run
- Installs current plugin source with `[mcp]` extras
- Applies sane default env vars for persistent memory
- Launches `python -m mnemos.mcp_server` over stdio
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path


def _plugin_root() -> Path:
    env_root = os.environ.get("CLAUDE_PLUGIN_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[1]


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _expected_install_stamp(plugin_root: Path) -> str:
    pyproject = plugin_root / "pyproject.toml"
    if not pyproject.exists():
        return "unknown"
    # Use mtime as a cheap invalidation key for local plugin updates.
    return str(int(pyproject.stat().st_mtime))


def _create_venv(venv_dir: Path) -> None:
    builder = venv.EnvBuilder(with_pip=True, clear=False)
    builder.create(venv_dir)


def _install_plugin(venv_python: Path, plugin_root: Path) -> None:
    subprocess.check_call([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            f"{plugin_root}[mcp]",
        ]
    )


def _ensure_runtime(venv_dir: Path, plugin_root: Path) -> Path:
    venv_python = _venv_python(venv_dir)
    stamp_path = venv_dir / ".mnemos-install-stamp"
    expected = _expected_install_stamp(plugin_root)

    if not venv_python.exists():
        _create_venv(venv_dir)
        _install_plugin(venv_python, plugin_root)
        stamp_path.write_text(expected, encoding="utf-8")
        return venv_python

    current = stamp_path.read_text(encoding="utf-8").strip() if stamp_path.exists() else ""
    if current != expected:
        _install_plugin(venv_python, plugin_root)
        stamp_path.write_text(expected, encoding="utf-8")

    return venv_python


def _apply_default_env(plugin_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    plugin_data_dir = plugin_root / ".claude-plugin"

    env.setdefault("MNEMOS_STORE_TYPE", "sqlite")
    env.setdefault("MNEMOS_SQLITE_PATH", str(plugin_data_dir / "mnemos.db"))

    if "MNEMOS_LLM_PROVIDER" not in env:
        if env.get("MNEMOS_OPENCLAW_API_KEY"):
            env["MNEMOS_LLM_PROVIDER"] = "openclaw"
        elif env.get("MNEMOS_OPENAI_API_KEY"):
            env["MNEMOS_LLM_PROVIDER"] = "openai"
        elif env.get("MNEMOS_OLLAMA_URL"):
            env["MNEMOS_LLM_PROVIDER"] = "ollama"
        else:
            env["MNEMOS_LLM_PROVIDER"] = "mock"

    if "MNEMOS_EMBEDDING_PROVIDER" not in env:
        llm_provider = env.get("MNEMOS_LLM_PROVIDER", "mock")
        if llm_provider in {"openai", "openclaw"}:
            env["MNEMOS_EMBEDDING_PROVIDER"] = llm_provider
        elif llm_provider == "ollama":
            env["MNEMOS_EMBEDDING_PROVIDER"] = "ollama"
        else:
            env["MNEMOS_EMBEDDING_PROVIDER"] = "simple"

    return env


def main() -> int:
    plugin_root = _plugin_root()
    venv_dir = plugin_root / ".claude-plugin" / ".venv"
    venv_python = _ensure_runtime(venv_dir, plugin_root)
    env = _apply_default_env(plugin_root)

    os.execve(
        str(venv_python),
        [str(venv_python), "-m", "mnemos.mcp_server"],
        env,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
