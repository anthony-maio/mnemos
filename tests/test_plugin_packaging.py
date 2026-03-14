"""
tests/test_plugin_packaging.py — Sanity checks for Claude Code plugin packaging files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib

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


def test_plugin_manifest_versions_match_package_version() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    expected_version = project["project"]["version"]

    plugin_manifest = json.loads(Path(".claude-plugin/plugin.json").read_text(encoding="utf-8"))
    marketplace_manifest = json.loads(
        Path(".claude-plugin/marketplace.json").read_text(encoding="utf-8")
    )

    assert plugin_manifest["version"] == expected_version
    assert marketplace_manifest["plugins"][0]["version"] == expected_version


def test_plugin_manifest_does_not_force_sqlite_env() -> None:
    data = json.loads(Path(".claude-plugin/plugin.json").read_text(encoding="utf-8"))
    env = data["mcpServers"]["mnemos"].get("env", {})

    assert "MNEMOS_STORE_TYPE" not in env
    assert "MNEMOS_SQLITE_PATH" not in env


def test_plugin_manifest_registers_curator_agent() -> None:
    data = json.loads(Path(".claude-plugin/plugin.json").read_text(encoding="utf-8"))

    assert "./agents/mnemos-curator.md" in data["agents"]


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


def test_plugin_wrapper_required_extras_include_neo4j(monkeypatch) -> None:
    wrapper = _load_plugin_wrapper()

    monkeypatch.setenv("MNEMOS_STORE_TYPE", "neo4j")
    extras = wrapper._required_install_extras(dict(os.environ))

    assert "mcp" in extras
    assert "neo4j" in extras


def test_plugin_wrapper_defaults_to_user_config_path(monkeypatch) -> None:
    wrapper = _load_plugin_wrapper()
    plugin_root = Path(".").resolve()

    monkeypatch.delenv("MNEMOS_CONFIG_PATH", raising=False)
    monkeypatch.setenv("APPDATA", r"C:\Users\Test\AppData\Roaming")

    env = wrapper._apply_default_env(plugin_root)

    assert env["MNEMOS_CONFIG_PATH"] == r"C:\Users\Test\AppData\Roaming\Mnemos\mnemos.toml"


def test_plugin_hooks_wire_autostore_and_consolidation() -> None:
    hooks_path = Path("hooks/hooks.json")
    assert hooks_path.exists()

    data = json.loads(hooks_path.read_text(encoding="utf-8"))
    hooks = data["hooks"]

    assert "UserPromptSubmit" in hooks
    assert "PostToolUse" in hooks
    assert "PreCompact" in hooks
    assert "Stop" in hooks

    prompt_command = hooks["UserPromptSubmit"][0]["hooks"][0]["command"]
    tool_command = hooks["PostToolUse"][0]["hooks"][0]["command"]
    consolidate_command = hooks["PreCompact"][0]["hooks"][0]["command"]

    assert "mnemos-cli autostore-hook UserPromptSubmit" in prompt_command
    assert "mnemos-cli autostore-hook PostToolUse" in tool_command
    assert "mnemos-cli consolidate" in consolidate_command


def test_plugin_wrapper_uses_explicit_runtime_python(monkeypatch) -> None:
    wrapper = _load_plugin_wrapper()
    plugin_root = Path(".").resolve()
    expected_python = plugin_root / ".venv-test" / "python.exe"

    monkeypatch.setenv("MNEMOS_PLUGIN_PYTHON", str(expected_python))

    runtime_python = wrapper._resolve_runtime_python(plugin_root, dict(os.environ))

    assert runtime_python == str(expected_python)
