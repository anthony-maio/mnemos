#!/usr/bin/env python3
"""Forward a Claude Code hook to the installed Mnemos runtime Python."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

# hooks.json starts this file through runpy, which does not put the script's
# directory on sys.path the way ``python script.py`` does.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_mnemos_mcp import _apply_default_env, _plugin_root, _ready_runtime_python  # noqa: E402

# Claude Code reads exit status 2 from a hook as a block: a Stop hook keeps
# Claude going and a UserPromptSubmit hook throws the prompt away. Capture is
# best effort, so nothing this wrapper returns may be 2.
_BLOCKING_EXIT = 2
_EVENTS = frozenset({"SessionStart", "UserPromptSubmit", "PreCompact", "PostCompact", "Stop"})


def _runtime_command(runtime_python: str, event: str) -> list[str]:
    return [runtime_python, "-m", "mnemos.adapters.claude_code", event]


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        print("Mnemos Claude hook requires one event name", file=sys.stderr)
        return 1
    event = sys.argv[1].strip()
    plugin_root: Path = _plugin_root()
    env = _apply_default_env(plugin_root)
    # Installing here would hold the host's hook open for a venv build and a
    # network pip install. The MCP server owns installation; until it has
    # finished, hooks skip, and transcript capture catches up afterwards.
    runtime_python = _ready_runtime_python(plugin_root, env)
    if runtime_python is None:
        safe_event = event if event in _EVENTS else "unknown"
        print(
            "mnemos hook diagnostic host=claude-code "
            f"event={safe_event} status=skipped reason=runtime_not_ready",
            file=sys.stderr,
        )
        return 0
    try:
        # Inherited stdio preserves Claude's protocol streams and subprocess
        # execution returns the configured runtime's exit status reliably on
        # both POSIX and Windows, including a bare ``python`` from PATH.
        completed = subprocess.run(
            _runtime_command(runtime_python, event),
            env=env,
            check=False,
        )
    except OSError as exc:
        print(f"Mnemos Claude hook runtime failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    status = int(completed.returncode)
    return 1 if status == _BLOCKING_EXIT else status


if __name__ == "__main__":
    raise SystemExit(main())
