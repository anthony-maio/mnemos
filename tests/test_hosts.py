"""
tests/test_hosts.py — Host adapter preview/apply coverage.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import mnemos.hosts as hosts
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
    assert payload["mcpServers"]["mnemos"]["command"] == hosts.sys.executable
    assert payload["mcpServers"]["mnemos"]["args"] == ["-m", "mnemos.mcp_server"]
    assert (
        payload["mcpServers"]["mnemos"]["env"]["MNEMOS_CONFIG_PATH"]
        == mnemos_config_path.as_posix()
    )
    assert rule_path.exists()
    rule_text = rule_path.read_text(encoding="utf-8")
    assert "alwaysApply: true" in rule_text
    assert "mnemos_retrieve" in rule_text
    assert "mnemos_consolidate" in rule_text


def test_mcp_integration_env_is_allowlisted_and_preserves_existing_env(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    config_path = repo_dir / ".cursor" / "mcp.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "mnemos": {
                        "command": "old-mnemos",
                        "env": {"KEEP_EXISTING": "yes", "MNEMOS_USER_PROFILE_ID": "old"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    apply_host_integration(
        "cursor",
        mnemos_config_path=tmp_path / "Mnemos" / "mnemos.toml",
        cwd=repo_dir,
        home=tmp_path / "home",
        integration_env={
            "MNEMOS_USER_PROFILE_ID": "Alice",
            "MNEMOS_SCOPE_REGISTRY": r"D:\scopes.json",
            "MNEMOS_RUNTIME_DESCRIPTOR": r"D:\runtime.json",
        },
    )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    env = payload["mcpServers"]["mnemos"]["env"]
    assert env["KEEP_EXISTING"] == "yes"
    assert env["MNEMOS_USER_PROFILE_ID"] == "Alice"
    assert env["MNEMOS_SCOPE_REGISTRY"] == r"D:\scopes.json"
    assert env["MNEMOS_RUNTIME_DESCRIPTOR"] == r"D:\runtime.json"
    assert env["MNEMOS_CONFIG_PATH"] == (tmp_path / "Mnemos" / "mnemos.toml").as_posix()

    with pytest.raises(ValueError, match="unsupported key"):
        preview_host_integration(
            "cursor",
            mnemos_config_path=tmp_path / "Mnemos" / "mnemos.toml",
            cwd=repo_dir,
            integration_env={"OPENAI_API_KEY": "secret"},
        )


def test_codex_integration_apply_writes_toml_config(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    repo_dir = tmp_path / "repo"
    config_path = home_dir / ".codex" / "config.toml"
    agents_path = repo_dir / "AGENTS.md"
    config_path.parent.mkdir(parents=True)
    repo_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
[mcp_servers.other]
command = "other-mcp"
""".strip(),
        encoding="utf-8",
    )
    agents_path.write_text(
        "# Repo Instructions\n\nKeep answers concise.\n",
        encoding="utf-8",
    )
    mnemos_config_path = tmp_path / "Mnemos" / "mnemos.toml"

    preview = preview_host_integration(
        "codex",
        mnemos_config_path=mnemos_config_path,
        cwd=repo_dir,
        home=home_dir,
    )
    assert "MNEMOS_CONFIG_PATH" in preview.preview_text
    assert "AGENTS.md" in preview.preview_text
    assert "## Mnemos Memory" in preview.preview_text

    result = apply_host_integration(
        "codex",
        mnemos_config_path=mnemos_config_path,
        cwd=repo_dir,
        home=home_dir,
    )

    assert result.backup_path is not None
    assert result.backup_path.exists()
    text = config_path.read_text(encoding="utf-8")
    assert "[mcp_servers.mnemos]" in text
    server = tomllib.loads(text)["mcp_servers"]["mnemos"]
    assert server["command"] == hosts.sys.executable
    assert server["args"] == ["-m", "mnemos.mcp_server"]
    assert "[mcp_servers.mnemos.env]" in text
    assert f'MNEMOS_CONFIG_PATH = "{mnemos_config_path.as_posix()}"' in text
    assert "[mcp_servers.other]" in text
    agents_text = agents_path.read_text(encoding="utf-8")
    assert "# Repo Instructions" in agents_text
    assert "## Mnemos Memory" in agents_text
    assert "mnemos_retrieve" in agents_text
    assert "mnemos_consolidate" in agents_text


@pytest.mark.parametrize("host", ["codex", "cursor", "claude-code"])
def test_mcp_launch_binding_replaces_old_args_without_path_lookup(
    tmp_path: Path, monkeypatch, host: str
) -> None:
    project = tmp_path / "project"
    home = tmp_path / "home"
    interpreter = str(tmp_path / "Chosen Python" / "python.exe")
    monkeypatch.setattr(hosts.sys, "executable", interpreter)
    monkeypatch.setenv("PATH", str(tmp_path / "old-installation"))
    path = hosts._config_path_for_host(host, cwd=project, home=home)
    path.parent.mkdir(parents=True)
    original = {
        "mnemos": {"command": "mnemos-mcp", "args": ["old-wrapper.py"], "timeout": 23},
        "other": {"command": "other-mcp", "args": ["keep"]},
    }
    if host == "codex":
        path.write_text(
            "\n".join(hosts._emit_toml_sections({"mcp_servers": original})), encoding="utf-8"
        )
    else:
        path.write_text(json.dumps({"mcpServers": original}), encoding="utf-8")
    preview = preview_host_integration(
        host, mnemos_config_path=tmp_path / "settings.toml", cwd=project, home=home
    )
    payload = (
        tomllib.loads(preview.rendered_config)["mcp_servers"]
        if host == "codex"
        else json.loads(preview.rendered_config)["mcpServers"]
    )
    assert payload["mnemos"]["command"] == interpreter
    assert payload["mnemos"]["args"] == ["-m", "mnemos.mcp_server"]
    assert payload["mnemos"]["timeout"] == 23
    assert payload["other"] == original["other"]
    # Preview must not alter the saved host settings.
    assert "old-wrapper.py" in path.read_text(encoding="utf-8")


def test_codex_explicit_integration_env_reaches_mcp_and_hook_bootstrap(tmp_path: Path) -> None:
    env = {
        "MNEMOS_USER_PROFILE_ID": "Alice",
        "MNEMOS_SCOPE_REGISTRY": r"D:\scopes.json",
        "MNEMOS_RUNTIME_DESCRIPTOR": r"D:\runtime.json",
    }
    preview = preview_host_integration(
        "codex",
        mnemos_config_path=tmp_path / "Mnemos" / "mnemos.toml",
        cwd=tmp_path / "repo",
        home=tmp_path / "home",
        integration_env=env,
    )
    rendered = tomllib.loads(preview.rendered_config)
    mcp_env = rendered["mcp_servers"]["mnemos"]["env"]
    assert mcp_env["MNEMOS_USER_PROFILE_ID"] == "Alice"
    assert mcp_env["MNEMOS_SCOPE_REGISTRY"] == r"D:\scopes.json"
    assert mcp_env["MNEMOS_RUNTIME_DESCRIPTOR"] == r"D:\runtime.json"
    assert "MNEMOS_CONFIG_PATH" in mcp_env

    command = rendered["hooks"]["UserPromptSubmit"][-1]["hooks"][0]["command"]
    assert " -c " in command
    assert "OPENAI_API_KEY" not in command
    encoded = command.split("base64.b64decode(", 1)[1].split(")", 1)[0]
    encoded = encoded.strip("'").replace("''", "'")
    import base64

    hook_env = json.loads(base64.b64decode(encoded))
    assert hook_env["MNEMOS_USER_PROFILE_ID"] == "Alice"
    assert hook_env["MNEMOS_SCOPE_REGISTRY"] == r"D:\scopes.json"
    assert hook_env["MNEMOS_CONFIG_PATH"] == (tmp_path / "Mnemos" / "mnemos.toml").as_posix()


def test_codex_integration_replaces_existing_mnemos_agents_block(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    repo_dir = tmp_path / "repo"
    config_path = home_dir / ".codex" / "config.toml"
    agents_path = repo_dir / "AGENTS.md"
    config_path.parent.mkdir(parents=True)
    repo_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text("", encoding="utf-8")
    agents_path.write_text(
        """
# Repo Instructions

## Mnemos Memory

Old outdated block.

## Local Rules

Keep answers concise.
""".strip(),
        encoding="utf-8",
    )

    apply_host_integration(
        "codex",
        mnemos_config_path=tmp_path / "Mnemos" / "mnemos.toml",
        cwd=repo_dir,
        home=home_dir,
    )

    agents_text = agents_path.read_text(encoding="utf-8")
    assert agents_text.count("## Mnemos Memory") == 1
    assert "Old outdated block." not in agents_text
    assert "mnemos_inspect" in agents_text
    assert "## Local Rules" in agents_text


def test_codex_integration_installs_native_hooks_and_preserves_existing_hook(
    tmp_path: Path,
) -> None:
    home_dir = tmp_path / "home"
    repo_dir = tmp_path / "repo"
    config_path = home_dir / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True)
    repo_dir.mkdir(parents=True)
    config_path.write_text(
        """
[[hooks.SessionStart]]
matcher = "startup"

[[hooks.SessionStart.hooks]]
type = "command"
command = "echo existing"
""".strip(),
        encoding="utf-8",
    )

    preview = preview_host_integration(
        "codex", mnemos_config_path=tmp_path / "Mnemos" / "mnemos.toml", cwd=repo_dir, home=home_dir
    )
    rendered = tomllib.loads(preview.rendered_config)
    handlers = rendered["hooks"]["SessionStart"]
    assert any(item["hooks"][0]["command"] == "echo existing" for item in handlers)
    assert any("mnemos.adapters.codex" in item["hooks"][0]["command"] for item in handlers)
    assert (
        "matcher ="
        not in preview.rendered_config.split("[[hooks.UserPromptSubmit]]", 1)[1].split(
            "[[hooks.UserPromptSubmit.hooks]]", 1
        )[0]
    )
    assert "command_windows" not in preview.rendered_config
    session_start = rendered["hooks"]["SessionStart"][-1]
    assert session_start["matcher"] == "startup|resume|clear|compact"
    post_compact = rendered["hooks"]["PostCompact"][-1]
    assert "additionalContextLimit" not in post_compact["hooks"][0]


def test_codex_hook_command_quotes_installed_python_path(tmp_path: Path, monkeypatch) -> None:
    python_path = "C:\\Program Files\\Mnemos`$bin\\py'thon.exe"
    monkeypatch.setattr(hosts.sys, "executable", python_path)
    preview = preview_host_integration(
        "codex",
        mnemos_config_path=tmp_path / "Mnemos" / "mnemos.toml",
        cwd=tmp_path / "repo",
        home=tmp_path / "home",
    )

    rendered = tomllib.loads(preview.rendered_config)
    generated = rendered["hooks"]["UserPromptSubmit"][-1]["hooks"][0]
    if hosts._powershell_executable() is not None:
        assert generated["command"] == (
            "& 'C:\\Program Files\\Mnemos`$bin\\py''thon.exe' -m mnemos.adapters.codex"
        )
    else:
        assert generated["command"] == f'"{python_path}" -m mnemos.adapters.codex'
    assert "command_windows" not in generated


def test_codex_hook_command_preserves_posix_shell_shape(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hosts.sys, "platform", "linux")
    python_path = "/opt/Mnemos Python/bin/python"
    monkeypatch.setattr(hosts.sys, "executable", python_path)

    preview = preview_host_integration(
        "codex",
        mnemos_config_path=tmp_path / "Mnemos" / "mnemos.toml",
        cwd=tmp_path / "repo",
        home=tmp_path / "home",
    )
    generated = tomllib.loads(preview.rendered_config)["hooks"]["UserPromptSubmit"][-1]["hooks"][0]
    assert shlex.split(generated["command"]) == [python_path, "-m", "mnemos.adapters.codex"]


@pytest.mark.parametrize("explicit_env", [False, True])
@pytest.mark.skipif(os.name == "nt", reason="requires a POSIX executable and shell")
def test_codex_posix_hook_command_executes_quoted_interpreter_path(
    monkeypatch, tmp_path: Path, explicit_env: bool
) -> None:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("POSIX sh is unavailable")

    real_python = sys.executable
    interpreter = tmp_path / "Mnemos $`tick' Python" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(real_python)
    package = tmp_path / "stub package" / "mnemos" / "adapters"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "codex.py").write_text(
        "import json, os, sys\n"
        "payload = json.load(sys.stdin)\n"
        "print(json.dumps({'event_id': payload['event_id'], "
        "'profile': os.environ.get('MNEMOS_USER_PROFILE_ID'), "
        "'registry': os.environ.get('MNEMOS_SCOPE_REGISTRY')}))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(hosts.sys, "platform", "linux")
    monkeypatch.setattr(hosts.sys, "executable", str(interpreter))
    integration_env = (
        {
            "MNEMOS_USER_PROFILE_ID": "posix-profile",
            "MNEMOS_SCOPE_REGISTRY": str(tmp_path / "scopes.json"),
        }
        if explicit_env
        else None
    )
    command = hosts._codex_hook_command(integration_env)
    child_env = {**os.environ, "PYTHONPATH": str(tmp_path / "stub package")}
    for key in ("MNEMOS_USER_PROFILE_ID", "MNEMOS_SCOPE_REGISTRY"):
        child_env.pop(key, None)
    completed = subprocess.run(
        [shell, "-c", command],
        input='{"event_id":"quoted-path"}',
        capture_output=True,
        text=True,
        env=child_env,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout)
    assert record["event_id"] == "quoted-path"
    assert record["profile"] == ("posix-profile" if explicit_env else None)
    assert record["registry"] == (str(tmp_path / "scopes.json") if explicit_env else None)


@pytest.mark.parametrize("python_path", ["/tmp/mnemos-python\n", "/tmp/mnemos-python\x00"])
def test_codex_hook_command_rejects_nul_and_newline_paths(monkeypatch, python_path: str) -> None:
    monkeypatch.setattr(hosts.sys, "platform", "linux")
    monkeypatch.setattr(hosts.sys, "executable", python_path)
    with pytest.raises(ValueError, match="NUL or newline"):
        hosts._codex_hook_command(None)


def test_codex_hook_command_uses_cmd_fallback_when_powershell_missing(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(hosts, "_powershell_executable", lambda: None)
    python_path = r"C:\Program Files\Mnemos\python.exe"
    monkeypatch.setattr(hosts.sys, "executable", python_path)

    preview = preview_host_integration(
        "codex",
        mnemos_config_path=tmp_path / "Mnemos" / "mnemos.toml",
        cwd=tmp_path / "repo",
        home=tmp_path / "home",
    )
    generated = tomllib.loads(preview.rendered_config)["hooks"]["UserPromptSubmit"][-1]["hooks"][0]
    assert generated["command"] == f'"{python_path}" -m mnemos.adapters.codex'


@pytest.mark.parametrize(
    "python_path", [r"C:\Mnemos%PATH%\python.exe", 'C:\\Mnemos"quoted\\python.exe']
)
def test_codex_cmd_fallback_rejects_unsafe_interpreter_path(monkeypatch, python_path: str) -> None:
    monkeypatch.setattr(hosts.sys, "platform", "win32")
    monkeypatch.setattr(hosts, "_powershell_executable", lambda: None)
    monkeypatch.setattr(hosts.sys, "executable", python_path)
    with pytest.raises(ValueError, match="unsupported by Cmd"):
        hosts._codex_hook_command(None)


def test_codex_powershell_bootstrap_executes_allowlisted_env_and_stdin(
    tmp_path: Path,
) -> None:
    powershell = hosts._powershell_executable()
    if powershell is None:
        pytest.skip("PowerShell is unavailable on this host")

    stub_root = tmp_path / "stub package"
    stub_module = stub_root / "mnemos" / "adapters"
    stub_module.mkdir(parents=True)
    (stub_root / "mnemos" / "__init__.py").write_text("", encoding="utf-8")
    (stub_module / "__init__.py").write_text("", encoding="utf-8")
    output_path = tmp_path / "path $ O'Brien with spaces" / "hook-output.json"
    output_path.parent.mkdir()
    (stub_module / "codex.py").write_text(
        "import json, os, sys\n"
        "payload = json.load(sys.stdin)\n"
        "record = {'profile': os.environ.get('MNEMOS_USER_PROFILE_ID'), "
        "'scope_registry': os.environ.get('MNEMOS_SCOPE_REGISTRY'), "
        "'event_id': payload.get('event_id'), 'prompt': payload.get('prompt')}\n"
        "path = os.environ['MNEMOS_RUNTIME_DESCRIPTOR']\n"
        "open(path, 'w', encoding='utf-8').write(json.dumps(record))\n"
        "print(json.dumps({'continue': True, 'record': record}))\n",
        encoding="utf-8",
    )
    env_values = {
        "MNEMOS_CONFIG_PATH": str(tmp_path / "config $ O'Brien.toml"),
        "MNEMOS_USER_PROFILE_ID": "Power'Shell $ Profile caf\u00e9",
        "MNEMOS_SCOPE_REGISTRY": str(tmp_path / "registry $ O'Brien with spaces.json"),
        "MNEMOS_RUNTIME_DESCRIPTOR": str(output_path),
    }
    command = hosts._codex_hook_command(env_values)
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "event_id": "powershell-stub-event",
        "prompt": "stdin reaches the generated bootstrap caf\u00e9",
    }
    env = dict(os.environ)
    env["PYTHONPATH"] = str(stub_root)
    for key in env_values:
        env.pop(key, None)
    env["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        [powershell, "-NoProfile", "-Command", command],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(tmp_path),
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert output["record"]["profile"] == "Power'Shell $ Profile caf\u00e9"
    assert output["record"]["scope_registry"] == env_values["MNEMOS_SCOPE_REGISTRY"]
    assert output["record"]["event_id"] == payload["event_id"]
    assert output["record"]["prompt"] == payload["prompt"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == output["record"]


def test_hermes_integration_selects_provider_without_rewriting_config(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    config_path = home_dir / ".hermes" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """display:
  streaming: true
memory:
  enabled: true
  provider: builtin
model:
  name: local
""",
        encoding="utf-8",
    )

    preview = preview_host_integration(
        "hermes", mnemos_config_path=tmp_path / "Mnemos" / "mnemos.toml", home=home_dir
    )
    assert preview.config_path == config_path
    assert "provider: mnemos" in preview.rendered_config
    assert "name: local" in preview.rendered_config
    result = apply_host_integration(
        "hermes", mnemos_config_path=tmp_path / "Mnemos" / "mnemos.toml", home=home_dir
    )
    assert result.backup_path is not None
    text = config_path.read_text(encoding="utf-8")
    assert text.count("provider:") == 1
    assert "provider: mnemos" in text


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

    assert preview.config_path == repo_dir / ".mcp.json"
    assert "run_mnemos_mcp.py" in preview.preview_text
    assert mnemos_config_path.as_posix() in preview.preview_text


def test_claude_integration_apply_installs_mnemos_agents(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    home_dir = tmp_path / "home"
    wrapper_path = repo_dir / ".claude-plugin" / "run_mnemos_mcp.py"
    wrapper_path.parent.mkdir(parents=True)
    wrapper_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    mnemos_config_path = tmp_path / "Mnemos" / "mnemos.toml"

    preview = preview_host_integration(
        "claude-code",
        mnemos_config_path=mnemos_config_path,
        cwd=repo_dir,
        home=home_dir,
    )

    assert "mnemos-recall.md" in preview.preview_text
    assert "mnemos-curator.md" in preview.preview_text

    result = apply_host_integration(
        "claude-code",
        mnemos_config_path=mnemos_config_path,
        cwd=repo_dir,
        home=home_dir,
        integration_env={
            "MNEMOS_USER_PROFILE_ID": "alice",
            "MNEMOS_SCOPE_REGISTRY": str(tmp_path / "scopes.json"),
        },
    )

    assert result.backup_path is None
    mcp_payload = json.loads((repo_dir / ".mcp.json").read_text(encoding="utf-8"))
    mcp_env = mcp_payload["mcpServers"]["mnemos"]["env"]
    assert mcp_env["MNEMOS_USER_PROFILE_ID"] == "alice"
    assert mcp_env["MNEMOS_SCOPE_REGISTRY"] == str(tmp_path / "scopes.json")
    recall_path = home_dir / ".claude" / "agents" / "mnemos-recall.md"
    curator_path = home_dir / ".claude" / "agents" / "mnemos-curator.md"
    assert recall_path.exists()
    assert curator_path.exists()
    assert "mnemos_retrieve" in recall_path.read_text(encoding="utf-8")
    assert "mnemos_store" in curator_path.read_text(encoding="utf-8")
