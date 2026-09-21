"""Codex setup edits only its own entries and leaves every other setting as written."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mnemos.hosts import apply_host_integration, codex_config_without_mnemos

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

PERSONAL = r'''# Personal Codex settings -- keep these comments
model = "gpt-5-codex"
approval_policy = "on-request"
developer_instructions = """
Prefer small diffs.
Never push to main."""

[projects.'C:\Users\Ant\repo']
trust_level = "trusted"

[projects."/Users/me/repo"]
trust_level = "trusted"

[profiles."gpt-5.1"]
model = "gpt-5.1"

[mcp_servers.github]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
env = { GITHUB_PERSONAL_ACCESS_TOKEN = "ghx" }
'''

EXISTING_HOOKS = r"""[mcp_servers.mnemos]
command = "old-python"
args = ["-m", "mnemos.mcp_server"]
startup_timeout_sec = 20

[mcp_servers.mnemos.env]
MNEMOS_USER_PROFILE_ID = "local"

[[hooks.Stop]]
[[hooks.Stop.hooks]]
type = "command"
command = "notify-send done"  # the user's own handler
[[hooks.Stop.hooks]]
type = "command"
command = "python -m mnemos.adapters.codex Stop"
"""

CASES = {
    "personal": PERSONAL,
    "windows project": r"[projects.'C:\Users\Ant\repo']" + '\ntrust_level = "trusted"\n',
    "dotted profile": '[profiles."gpt-5.1"]\nmodel = "gpt-5.1"\n',
    "multi-line string": 'developer_instructions = """\nline one\nline two"""\n',
    "existing hooks": EXISTING_HOOKS,
}


def _apply(tmp_path: Path, text: str) -> Path:
    home, cwd = tmp_path / "home", tmp_path / "project"
    (home / ".codex").mkdir(parents=True, exist_ok=True)
    cwd.mkdir(exist_ok=True)
    config = home / ".codex" / "config.toml"
    config.write_text(text, encoding="utf-8")
    apply_host_integration("codex", mnemos_config_path=tmp_path / "mnemos.toml", cwd=cwd, home=home)
    return config


@pytest.mark.parametrize("label", list(CASES))
def test_setup_preserves_every_unrelated_setting(tmp_path: Path, label: str) -> None:
    original = CASES[label]
    config = _apply(tmp_path, original)
    written = config.read_text(encoding="utf-8")
    parsed = tomllib.loads(written)
    assert codex_config_without_mnemos(parsed) == codex_config_without_mnemos(
        tomllib.loads(original)
    )
    assert parsed["mcp_servers"]["mnemos"]["args"] == ["-m", "mnemos.mcp_server"]
    assert parsed["mcp_servers"]["mnemos"]["env"]["MNEMOS_HOST"] == "codex"
    for line in original.splitlines():
        if line.startswith("#") or line.startswith("[projects."):
            assert line in written


def test_setup_keeps_user_hooks_and_replaces_its_own(tmp_path: Path) -> None:
    parsed = tomllib.loads(_apply(tmp_path, EXISTING_HOOKS).read_text(encoding="utf-8"))
    stop = [handler["command"] for group in parsed["hooks"]["Stop"] for handler in group["hooks"]]
    assert stop[0] == "notify-send done"
    assert len([command for command in stop if "mnemos.adapters.codex" in command]) == 1
    assert parsed["mcp_servers"]["mnemos"]["startup_timeout_sec"] == 20


def test_setup_is_idempotent(tmp_path: Path) -> None:
    config = _apply(tmp_path, PERSONAL)
    first = config.read_text(encoding="utf-8")
    config = _apply(tmp_path, first)
    assert config.read_text(encoding="utf-8") == first


def test_repeated_setup_never_overwrites_a_backup(tmp_path: Path) -> None:
    config = _apply(tmp_path, PERSONAL)
    _apply(tmp_path, config.read_text(encoding="utf-8"))
    backups = sorted(config.parent.glob("config.toml.bak.*"))
    assert len(backups) == 2
    assert PERSONAL in {backup.read_text(encoding="utf-8") for backup in backups}
