"""
tests/test_hosts.py — Host adapter preview/apply coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

from mnemos.hosts import apply_host_integration, preview_host_integration


def test_cursor_integration_apply_writes_minimal_config_and_backup(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    config_path = repo_dir / ".cursor" / "mcp.json"
    rule_path = repo_dir / ".cursor" / "rules" / "mnemos-memory.mdc"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "other": {"command": "other-mcp"},
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    mnemos_config_path = tmp_path / "Mnemos" / "mnemos.toml"

    preview = preview_host_integration(
        "cursor",
        mnemos_config_path=mnemos_config_path,
        cwd=repo_dir,
        home=tmp_path / "home",
    )
    assert "MNEMOS_CONFIG_PATH" in preview.preview_text
    assert "mnemos-memory.mdc" in preview.preview_text

    result = apply_host_integration(
        "cursor",
        mnemos_config_path=mnemos_config_path,
        cwd=repo_dir,
        home=tmp_path / "home",
    )

    assert result.backup_path is not None
    assert result.backup_path.exists()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["mcpServers"]["other"]["command"] == "other-mcp"
    assert payload["mcpServers"]["mnemos"]["command"] == "mnemos-mcp"
    assert (
        payload["mcpServers"]["mnemos"]["env"]["MNEMOS_CONFIG_PATH"]
        == mnemos_config_path.as_posix()
    )
    assert rule_path.exists()
    rule_text = rule_path.read_text(encoding="utf-8")
    assert "alwaysApply: true" in rule_text
    assert "mnemos_retrieve" in rule_text
    assert "mnemos_consolidate" in rule_text


def test_codex_integration_apply_writes_toml_config(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    config_path = home_dir / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
[mcp_servers.other]
command = "other-mcp"
""".strip(),
        encoding="utf-8",
    )
    mnemos_config_path = tmp_path / "Mnemos" / "mnemos.toml"

    result = apply_host_integration(
        "codex",
        mnemos_config_path=mnemos_config_path,
        cwd=tmp_path / "repo",
        home=home_dir,
    )

    assert result.backup_path is not None
    assert result.backup_path.exists()
    text = config_path.read_text(encoding="utf-8")
    assert "[mcp_servers.mnemos]" in text
    assert 'command = "mnemos-mcp"' in text
    assert "[mcp_servers.mnemos.env]" in text
    assert f'MNEMOS_CONFIG_PATH = "{mnemos_config_path.as_posix()}"' in text
    assert "[mcp_servers.other]" in text


def test_claude_integration_prefers_plugin_wrapper_when_available(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    wrapper_path = repo_dir / ".claude-plugin" / "run_mnemos_mcp.py"
    wrapper_path.parent.mkdir(parents=True)
    wrapper_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    mnemos_config_path = tmp_path / "Mnemos" / "mnemos.toml"

    preview = preview_host_integration(
        "claude-code",
        mnemos_config_path=mnemos_config_path,
        cwd=repo_dir,
        home=tmp_path / "home",
    )

    assert "run_mnemos_mcp.py" in preview.preview_text
    assert mnemos_config_path.as_posix() in preview.preview_text
