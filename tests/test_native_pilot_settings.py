"""One-run hook trust must be limited to the reviewed pilot command structure."""

import json
from pathlib import Path
import tomllib

import pytest

from benchmarks.native_hosts.pilot_settings import codex_patch_settings


def fixture_patch(python: Path, launcher: Path) -> str:
    command = f"& '{python.as_posix()}' -I '{launcher.as_posix()}' hook codex"
    lines = [
        "[mcp_servers.mnemos]",
        "command = " + json.dumps(python.as_posix()),
        "args = " + json.dumps(["-I", launcher.as_posix(), "mcp", "codex"]),
    ]
    for event, matcher, option in (
        ("SessionStart", "startup|resume|clear|compact", "additionalContextLimit = 1200"),
        ("UserPromptSubmit", None, "additionalContextLimit = 1200"),
        ("PreCompact", "manual|auto", "timeout = 10"),
        ("PostCompact", "manual|auto", ""),
        ("Stop", None, "timeout = 3"),
        ("SessionEnd", "other", "timeout = 3"),
    ):
        lines.append(f"[[hooks.{event}]]")
        if matcher:
            lines.append("matcher = " + json.dumps(matcher))
        lines.extend(
            [f"[[hooks.{event}.hooks]]", 'type = "command"', "command = " + json.dumps(command)]
        )
        if option:
            lines.append(option)
    return (
        "--- private-config\n+++ private-config\n@@ -10,0 +11,50 @@\n"
        + "\n".join("+" + line for line in lines)
        + "\n"
    )


def test_only_launcher_location_changes(tmp_path):
    python = tmp_path / "persistent runtime/python.exe"
    launcher = tmp_path / "persistent runtime/launch.py"
    target = tmp_path / "disposable runtime/launch.py"
    patch = fixture_patch(python, launcher)
    rendered = codex_patch_settings(
        patch, python=python, launcher=launcher, disposable_launcher=target
    )
    parsed = tomllib.loads(rendered)
    assert parsed["mcp_servers"]["mnemos"] == {
        "command": python.as_posix(),
        "args": ["-I", target.as_posix(), "mcp", "codex"],
    }
    assert len(parsed["hooks"]) == 6
    for entries in parsed["hooks"].values():
        assert len(entries) == 1 and len(entries[0]["hooks"]) == 1
        assert (
            entries[0]["hooks"][0]["command"]
            == f"& '{python.as_posix()}' -I '{target.as_posix()}' hook codex"
        )
    assert launcher.as_posix() in patch and launcher.as_posix() not in rendered


@pytest.mark.parametrize(
    "alter",
    [
        lambda text: text + '+[mcp_servers.other]\n+command = "unreviewed.exe"\n',
        lambda text: text.replace(' hook codex"', ' hook codex; unreviewed.exe"'),
        lambda text: text.replace("+timeout = 3", "+timeout = 300"),
        lambda text: text.replace('+type = "command"', '+type = "prompt"', 1),
        lambda text: text.replace('+matcher = "other"', '+matcher = ".*"'),
        lambda text: text
        + '+[[hooks.Stop.hooks]]\n+type = "command"\n+command = "unreviewed.exe"\n',
        lambda text: text + "+[features]\n+hooks = true\n",
        lambda text: text + '-removed = "anything"\n',
    ],
)
def test_unreviewed_patch_semantics_fail_closed(tmp_path, alter):
    python, launcher = tmp_path / "python.exe", tmp_path / "launch.py"
    with pytest.raises(ValueError):
        codex_patch_settings(
            alter(fixture_patch(python, launcher)),
            python=python,
            launcher=launcher,
            disposable_launcher=tmp_path / "isolated/launch.py",
        )


def test_live_launcher_cannot_be_reused(tmp_path):
    python, launcher = tmp_path / "python.exe", tmp_path / "launch.py"
    with pytest.raises(ValueError, match="distinct absolute"):
        codex_patch_settings(
            fixture_patch(python, launcher),
            python=python,
            launcher=launcher,
            disposable_launcher=launcher,
        )
