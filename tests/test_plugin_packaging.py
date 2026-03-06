"""
tests/test_plugin_packaging.py — Sanity checks for Claude Code plugin packaging files.
"""

from __future__ import annotations

import json
from pathlib import Path


def test_plugin_manifest_is_valid_json() -> None:
    manifest_path = Path(".claude-plugin/plugin.json")
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["name"] == "mnemos-memory"
    assert "mcpServers" in data
    assert "mnemos" in data["mcpServers"]


def test_marketplace_manifest_matches_plugin_name() -> None:
    marketplace_path = Path(".claude-plugin/marketplace.json")
    assert marketplace_path.exists()
    data = json.loads(marketplace_path.read_text(encoding="utf-8"))
    plugins = data.get("plugins", [])
    assert plugins
    assert plugins[0]["name"] == "mnemos-memory"


def test_plugin_wrapper_exists() -> None:
    wrapper = Path(".claude-plugin/run_mnemos_mcp.py")
    assert wrapper.exists()
