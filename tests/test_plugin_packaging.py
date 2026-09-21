"""
tests/test_plugin_packaging.py — Sanity checks for Claude Code plugin packaging files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

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


def _load_hook_wrapper():
    plugin_wrapper = _load_plugin_wrapper()
    sys.modules["run_mnemos_mcp"] = plugin_wrapper
    wrapper_path = Path(".claude-plugin/run_mnemos_hook.py")
    spec = spec_from_file_location("mnemos_plugin_hook_wrapper", wrapper_path)
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


def test_plugin_manifest_requests_eager_mnemos_startup() -> None:
    data = json.loads(Path(".claude-plugin/plugin.json").read_text(encoding="utf-8"))
    assert data["mcpServers"]["mnemos"]["alwaysLoad"] is True


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


def test_plugin_manifest_registers_recall_agent() -> None:
    data = json.loads(Path(".claude-plugin/plugin.json").read_text(encoding="utf-8"))

    assert "./agents/mnemos-recall.md" in data["agents"]


def test_plugin_wrapper_exists() -> None:
    wrapper = Path(".claude-plugin/run_mnemos_mcp.py")
    assert wrapper.exists()


def test_plugin_wrapper_leaves_managed_defaults_to_runtime(monkeypatch) -> None:
    wrapper = _load_plugin_wrapper()
    plugin_root = Path(".").resolve()

    for name in (
        "MNEMOS_CONFIG_PATH",
        "MNEMOS_STORE_TYPE",
        "MNEMOS_SQLITE_PATH",
        "MNEMOS_LLM_PROVIDER",
        "MNEMOS_EMBEDDING_PROVIDER",
        "MNEMOS_OPENCLAW_API_KEY",
        "MNEMOS_OPENAI_API_KEY",
        "MNEMOS_OLLAMA_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    env = wrapper._apply_default_env(plugin_root)

    assert all(
        name not in env
        for name in (
            "MNEMOS_CONFIG_PATH",
            "MNEMOS_STORE_TYPE",
            "MNEMOS_SQLITE_PATH",
            "MNEMOS_LLM_PROVIDER",
            "MNEMOS_EMBEDDING_PROVIDER",
        )
    )


def test_plugin_wrapper_preserves_explicit_runtime_configuration(
    monkeypatch, tmp_path: Path
) -> None:
    wrapper = _load_plugin_wrapper()
    plugin_root = tmp_path / "plugin"
    explicit = {
        "MNEMOS_CONFIG_PATH": str(tmp_path / "managed.toml"),
        "MNEMOS_STORE_TYPE": "sqlite",
        "MNEMOS_SQLITE_PATH": str(tmp_path / "memory.sqlite"),
        "MNEMOS_LLM_PROVIDER": "ollama",
        "MNEMOS_EMBEDDING_PROVIDER": "ollama",
    }
    for key, value in explicit.items():
        monkeypatch.setenv(key, value)

    env = wrapper._apply_default_env(plugin_root)

    for key, value in explicit.items():
        assert env[key] == value


def test_plugin_wrapper_binds_claude_host_over_inherited_value(monkeypatch) -> None:
    wrapper = _load_plugin_wrapper()
    monkeypatch.setenv("MNEMOS_HOST", "manual")

    env = wrapper._apply_default_env(Path("."))

    assert env["MNEMOS_HOST"] == "claude-code"


def test_plugin_wrapper_required_extras_are_fixed_to_mcp(monkeypatch) -> None:
    wrapper = _load_plugin_wrapper()

    monkeypatch.setenv("MNEMOS_STORE_TYPE", "sqlite")
    extras = wrapper._required_install_extras(dict(os.environ))

    assert extras == ("mcp",)


def test_plugin_hooks_wire_autostore_and_consolidation() -> None:
    hooks_path = Path("hooks/hooks.json")
    assert hooks_path.exists()

    data = json.loads(hooks_path.read_text(encoding="utf-8"))
    hooks = data["hooks"]

    assert "UserPromptSubmit" in hooks
    assert "SessionStart" in hooks
    assert "PreCompact" in hooks
    assert "Stop" in hooks
    assert "PostCompact" in hooks

    prompt_command = hooks["UserPromptSubmit"][0]["hooks"][0]["command"]
    session_command = hooks["SessionStart"][0]["hooks"][0]["command"]
    consolidate_command = hooks["PreCompact"][0]["hooks"][0]["command"]
    stop_command = hooks["Stop"][0]["hooks"][0]["command"]

    # runpy turns a missing script into an exception (exit 1). ``python script``
    # exits 2 instead, which Claude reads as a block.
    runner = (
        'python -c "import runpy,sys;sys.argv[:]=sys.argv[1:];'
        "runpy.run_path(sys.argv[0],run_name='__main__')\" "
        '"${CLAUDE_PLUGIN_ROOT}/.claude-plugin/run_mnemos_hook.py"'
    )
    assert prompt_command == f"{runner} UserPromptSubmit"
    assert session_command == f"{runner} SessionStart"
    assert consolidate_command == f"{runner} PreCompact"
    assert stop_command == f"{runner} Stop"
    # No shell operators: the same text has to run under bash, cmd, and PowerShell.
    assert all(
        token not in command
        for command in (prompt_command, session_command, consolidate_command, stop_command)
        for token in ("||", "&&", "2>", "$(")
    )


def test_plugin_hook_forwards_stdin_stdout_and_explicit_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    wrapper = _load_hook_wrapper()
    runtime_python = str(tmp_path / "runtime python.exe")
    captured: dict[str, object] = {}

    monkeypatch.setenv("MNEMOS_PLUGIN_PYTHON", runtime_python)
    monkeypatch.setattr(wrapper.sys, "argv", ["run_mnemos_hook.py", "Stop"])
    monkeypatch.setattr(wrapper, "_ready_runtime_python", lambda root, env: runtime_python)

    def fake_run(args, env, check):
        captured["args"] = args
        captured["env"] = env
        captured["check"] = check
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)

    assert wrapper.main() == 0
    assert captured["args"] == [
        runtime_python,
        "-m",
        "mnemos.adapters.claude_code",
        "Stop",
    ]
    assert captured["env"]["MNEMOS_PLUGIN_PYTHON"] == runtime_python
    assert captured["check"] is False


def test_plugin_bootstrap_sends_subprocess_output_to_stderr(monkeypatch, tmp_path: Path) -> None:
    wrapper = _load_plugin_wrapper()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return None

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)

    wrapper._install_plugin(tmp_path / "python.exe", tmp_path / "plugin", {})

    assert len(calls) == 2
    assert all(call[1]["stdout"] is sys.stderr for call in calls)
    assert all(call[1]["stderr"] is sys.stderr for call in calls)


def test_plugin_wrapper_uses_explicit_runtime_python(monkeypatch) -> None:
    wrapper = _load_plugin_wrapper()
    plugin_root = Path(".").resolve()
    expected_python = plugin_root / ".venv-test" / "python.exe"

    monkeypatch.setenv("MNEMOS_PLUGIN_PYTHON", str(expected_python))

    runtime_python = wrapper._resolve_runtime_python(plugin_root, dict(os.environ))

    assert runtime_python == str(expected_python)
