"""Validate the exact private pilot hook commands before an isolated native run."""

import json
from pathlib import Path
import tomllib
from typing import Any


def codex_patch_settings(
    patch: str, *, python: Path, launcher: Path, disposable_launcher: Path
) -> str:
    """Copy only the vetted Mnemos additions, never the user's full configuration."""
    lines = patch.splitlines()
    if any(line.startswith("-") and not line.startswith("---") for line in lines):
        raise ValueError("Pilot patch must contain additions only")
    additions = (
        "\n".join(line[1:] for line in lines if line.startswith("+") and not line.startswith("+++"))
        + "\n"
    )
    data = tomllib.loads(additions)
    executable, source, target = (
        python.as_posix(),
        launcher.as_posix(),
        disposable_launcher.as_posix(),
    )
    if source == target or not all(
        path.is_absolute() for path in (python, launcher, disposable_launcher)
    ):
        raise ValueError("Pilot commands require distinct absolute launcher paths")

    def literal(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    command = "& " + literal(executable) + " -I " + literal(source) + " hook codex"
    event_settings: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
        "SessionStart": (
            {"matcher": "startup|resume|clear|compact"},
            {"additionalContextLimit": 1200},
        ),
        "UserPromptSubmit": ({}, {"additionalContextLimit": 1200}),
        "PreCompact": ({"matcher": "manual|auto"}, {"timeout": 10}),
        "PostCompact": ({"matcher": "manual|auto"}, {}),
        "Stop": ({}, {"timeout": 3}),
        "SessionEnd": ({"matcher": "other"}, {"timeout": 3}),
    }
    expected: dict[str, Any] = {
        "mcp_servers": {"mnemos": {"command": executable, "args": ["-I", source, "mcp", "codex"]}},
        "hooks": {
            event: [{**matcher, "hooks": [{"type": "command", "command": command, **options}]}]
            for event, (matcher, options) in event_settings.items()
        },
    }
    if data != expected:
        raise ValueError("Pilot patch contains unvetted commands or settings")
    # JSON/TOML escaping is preserved by replacing only the path's existing spelling.
    # Reparse to verify the replacement, including the PowerShell single-quote escaping.
    replacement = additions.replace(json.dumps(source)[1:-1], json.dumps(target)[1:-1])
    expected["mcp_servers"]["mnemos"]["args"][1] = target
    new_command = "& " + literal(executable) + " -I " + literal(target) + " hook codex"
    for entries in expected["hooks"].values():
        entries[0]["hooks"][0]["command"] = new_command
    if tomllib.loads(replacement) != expected:
        raise ValueError("Disposable pilot command transplant changed semantics")
    return replacement
