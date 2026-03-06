"""
tests/test_plugin_packaging.py — Sanity checks for Claude Code plugin packaging files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from importlib.util import module_from_spec, spec_from_file_location


def _load_plugin_wrapper():
    wrapper_path = Path(".claude-plugin/run_mnemos_mcp.py")
    spec = spec_from_file_location("mnemos_plugin_wrapper", wrapper_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_plugin_wrapper_defaults_embedding_provider_from_llm(monkeypatch) -> None:
    wrapper = _load_plugin_wrapper()
    plugin_root = Path(".").resolve()

    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "openclaw")
    monkeypatch.setenv("MNEMOS_OPENCLAW_API_KEY", "claw-key")
    monkeypatch.delenv("MNEMOS_EMBEDDING_PROVIDER", raising=False)

    env = wrapper._apply_default_env(plugin_root)

    assert env["MNEMOS_EMBEDDING_PROVIDER"] == "openclaw"


def test_plugin_wrapper_required_extras_include_qdrant(monkeypatch) -> None:
    wrapper = _load_plugin_wrapper()

    monkeypatch.setenv("MNEMOS_STORE_TYPE", "qdrant")
    extras = wrapper._required_install_extras(dict(os.environ))

    assert "mcp" in extras
    assert "qdrant" in extras
