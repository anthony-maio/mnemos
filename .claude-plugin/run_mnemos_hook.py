#!/usr/bin/env python3
"""Forward a Claude Code hook to the explicit Mnemos runtime Python."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from run_mnemos_mcp import _apply_default_env, _plugin_root, _resolve_runtime_python


def _runtime_command(runtime_python: str, event: str) -> list[str]:
    return [runtime_python, "-m", "mnemos.adapters.claude_code", event]


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        print("Mnemos Claude hook requires one event name", file=sys.stderr)
        return 2
    event = sys.argv[1].strip()
    plugin_root: Path = _plugin_root()
    env = _apply_default_env(plugin_root)
    try:
        runtime_python = _resolve_runtime_python(plugin_root, env)
        # Inherited stdio preserves Claude's protocol streams and subprocess
        # execution returns the configured runtime's exit status reliably on
        # both POSIX and Windows, including a bare ``python`` from PATH.
        completed = subprocess.run(
            _runtime_command(runtime_python, event),
            env=env,
            check=False,
        )
        return int(completed.returncode)
    except OSError as exc:
        print(f"Mnemos Claude hook runtime failed: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
