"""Offline resource-boundary tests; these never launch a native host or model."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

import pytest

from benchmarks.native_hosts import native_processes as phases


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(phases, "_private", lambda path: None)
    root = tmp_path / "disposable"
    root.mkdir()
    project = root / "project"
    project.mkdir()
    registry = root / "scopes.json"
    registry.write_text("{}")
    settings = root / "settings.toml"
    settings.write_text("")
    (root / "plugin").mkdir()
    return phases.PhaseEnvironment(
        root, project, registry, root / "runtime.json", settings, "test", root / "plugin"
    )


@pytest.mark.parametrize(
    "resource", ["project", "registry", "descriptor", "settings", "plugin_snapshot"]
)
def test_resources_outside_disposable_root_are_rejected(config, resource):
    with pytest.raises(ValueError, match="escapes"):
        replace(config, **{resource: config.root.parent / "live"}).validate()


def test_environment_removes_live_selectors_credentials_and_import_overrides(monkeypatch):
    keys = [
        "MNEMOS_RUNTIME_DESCRIPTOR",
        "mnemos_user_profile_id",
        "PYTHONPATH",
        "PYTHONHOME",
        "CODEX_HOME",
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_SECURESTORAGE_CONFIG_DIR",
        "ANTHROPIC_CONFIG_DIR",
        "OPENAI_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "AWS_PROFILE",
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
    ]
    for key in keys:
        monkeypatch.setenv(key, "inherited-private-value")
    result = phases.clean_environment({"MNEMOS_RUNTIME_DESCRIPTOR": "disposable"})
    assert result["MNEMOS_RUNTIME_DESCRIPTOR"] == "disposable"
    assert all(key not in result for key in keys[1:])
    assert result["PYTHONIOENCODING"] == "utf-8"


def test_auth_copy_removes_new_file_when_acl_protection_fails(config, tmp_path, monkeypatch):
    source = tmp_path / "auth.json"
    source.write_text("synthetic credential")
    destination = config.root / "host"
    destination.mkdir()

    def fail_private(path):
        raise OSError("synthetic ACL failure")

    monkeypatch.setattr(phases, "_private", fail_private)
    with pytest.raises(OSError, match="ACL"):
        phases._auth_copy(source, destination, config.root)
    assert not (destination / source.name).exists()
    assert source.read_text() == "synthetic credential"


def test_disposable_selectors_override_storage_from_every_config_layer(config, monkeypatch):
    from mnemos.settings import load_settings

    outside = config.root.parent / "must-not-use.sqlite"
    monkeypatch.setenv("MNEMOS_SQLITE_PATH", str(outside))
    config.settings.write_text('[storage]\ntype="sqlite"\nsqlite_path="outside.sqlite"\n')
    project_config = config.project / ".mnemos" / "mnemos.toml"
    project_config.parent.mkdir()
    project_config.write_text('[storage]\ntype="memory"\nsqlite_path="also-outside.sqlite"\n')
    env = phases.clean_environment(phases._selectors(config, "claude-code", "outage"))
    resolved = load_settings(env=env, cwd=config.project)
    assert resolved.project_config_path == project_config
    assert resolved.settings.storage.type == "sqlite"
    assert Path(resolved.settings.storage.sqlite_path) == config.root / "memory.sqlite"
    assert Path(env["MNEMOS_RUNTIME_DIR"]) == config.root
    assert env["MNEMOS_AUTO_START"] == "false"
    assert not outside.exists()


def test_auth_copy_refuses_to_overwrite_existing_file(config, tmp_path):
    source = tmp_path / "auth.json"
    source.write_text("new credential")
    destination = config.root / "host"
    destination.mkdir()
    target = destination / source.name
    target.write_text("existing credential")
    with pytest.raises(FileExistsError):
        phases._auth_copy(source, destination, config.root)
    assert target.read_text() == "existing credential"


def test_codex_home_is_generated_config_directory(config):
    requested_home = config.root / "codex-home"
    generated = requested_home / ".codex" / "config.toml"
    assert phases._codex_config_home(generated, requested_home) == generated.parent.resolve()
    with pytest.raises(ValueError, match="escapes"):
        phases._codex_config_home(requested_home / "config.toml", requested_home)


def test_generated_codex_mcp_and_hooks_pin_outage_and_storage_settings(config):
    import base64

    directory = config.root / "codex-offline"
    directory.mkdir()
    home, env = phases._prepare_codex(config, directory, "codex-offline")
    parsed = tomllib.loads((home / "config.toml").read_text())
    mcp_env = parsed["mcp_servers"]["mnemos"]["env"]
    required = {
        "MNEMOS_AUTO_START": "false",
        "MNEMOS_RUNTIME_DIR": str(config.root.resolve()),
        "MNEMOS_STORE_TYPE": "sqlite",
        "MNEMOS_SQLITE_PATH": str(config.root.resolve() / "memory.sqlite"),
    }
    assert all(mcp_env.get(key) == value for key, value in required.items())
    assert all(env.get(key) == value for key, value in required.items())
    for entries in parsed["hooks"].values():
        for entry in entries:
            for hook in entry["hooks"]:
                command = hook["command"]
                encoded = command.split("base64.b64decode(", 1)[1].split(")", 1)[0]
                decoded = json.loads(base64.b64decode(encoded.strip("'").replace("''", "'")))
                assert all(decoded.get(key) == value for key, value in required.items())


def test_codex_setup_failure_records_failure_before_auth_copy(config, monkeypatch):
    monkeypatch.setattr(phases.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(phases.sys, "stdout", SimpleNamespace(isatty=lambda: True))

    def fail_setup(*args):
        raise ValueError("synthetic invalid generated config")

    monkeypatch.setattr(phases, "_prepare_codex", fail_setup)
    with pytest.raises(ValueError, match="generated config"):
        phases.run_codex_phase(
            config,
            phase="codex",
            prompt="unused",
            executable=Path("unused"),
            auth_source=Path("unused"),
        )
    record = json.loads((config.root / "codex" / "process.json").read_text())
    assert record["status"] == "failed"
    assert record["exception_type"] == "ValueError"
    assert record["auth_copy_removed"] is True
    assert "pid" not in record


def _run_codex_pinned(
    config,
    source,
    monkeypatch,
    *,
    model="codex-pinned",
    observed_model=None,
    observations=None,
    user_messages=None,
):
    home = config.root / "codex-home" / ".codex"
    home.mkdir(parents=True)
    monkeypatch.setattr(phases, "_prepare_codex", lambda *args: (home, {"CODEX_HOME": str(home)}))
    monkeypatch.setattr(phases.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(phases.sys, "stdout", SimpleNamespace(isatty=lambda: True))

    def launch(command, **kwargs):
        observed = model if observed_model is None else observed_model
        values = [observed] if observations is None else observations
        prompt_messages = ["synthetic prompt"] if user_messages is None else user_messages
        sessions = Path(kwargs["env"]["CODEX_HOME"]) / "sessions"
        sessions.mkdir(parents=True)
        (sessions / "session.jsonl").write_text(
            "".join(
                json.dumps({"type": "turn_context", "payload": {"model": value}}) + "\n"
                for value in values
            )
            + "".join(
                json.dumps(
                    {"type": "event_msg", "payload": {"type": "user_message", "message": message}}
                )
                + "\n"
                for message in prompt_messages
            )
        )
        return FakeProcess()

    monkeypatch.setattr(phases.subprocess, "Popen", launch)
    source.write_text("synthetic credential")
    return phases.run_codex_phase(
        config,
        phase="codex-pinned",
        prompt="synthetic prompt",
        executable=Path("codex.exe"),
        auth_source=source,
        model=model,
        migration_notice_target="gpt-5.6-sol",
    )


def test_codex_model_pin_is_command_bound_and_verified(config, tmp_path, monkeypatch):
    result = _run_codex_pinned(config, tmp_path / "auth.json", monkeypatch)
    assert result["requested_model"] == "codex-pinned"
    assert result["observed_models"] == ["codex-pinned"]
    assert result["model_verified"] is True
    assert result["user_prompt_count"] == 1
    assert result["user_prompt_verified"] is True
    assert result["migration_notice_pair"] == {"from": "codex-pinned", "to": "gpt-5.6-sol"}
    command = json.loads((config.root / "codex-pinned" / "process.json").read_text())["command"]
    assert command[command.index("--model") + 1] == "codex-pinned"
    override = tomllib.loads(command[command.index("-c") + 1])
    assert override == {"notice": {"model_migrations": {"codex-pinned": "gpt-5.6-sol"}}}


def test_codex_legacy_mode_has_no_model_pin_or_observation_requirement(
    config, tmp_path, monkeypatch
):
    home = config.root / "codex-home" / ".codex"
    home.mkdir(parents=True)
    monkeypatch.setattr(phases, "_prepare_codex", lambda *args: (home, {"CODEX_HOME": str(home)}))
    monkeypatch.setattr(phases.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(phases.sys, "stdout", SimpleNamespace(isatty=lambda: True))
    process = FakeProcess()
    monkeypatch.setattr(phases.subprocess, "Popen", lambda *args, **kwargs: process)
    source = tmp_path / "auth.json"
    source.write_text("synthetic credential")
    result = phases.run_codex_phase(
        config,
        phase="codex-legacy",
        prompt="synthetic prompt",
        executable=Path("codex.exe"),
        auth_source=source,
    )
    assert result["requested_model"] is None
    assert "observed_models" not in result
    command = json.loads((config.root / "codex-legacy" / "process.json").read_text())["command"]
    assert "--model" not in command


def test_codex_model_pin_rejects_missing_observation(config, tmp_path, monkeypatch):
    home = config.root / "codex-home" / ".codex"
    home.mkdir(parents=True)
    monkeypatch.setattr(phases, "_prepare_codex", lambda *args: (home, {"CODEX_HOME": str(home)}))
    monkeypatch.setattr(phases.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(phases.sys, "stdout", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(phases.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    source = tmp_path / "auth.json"
    source.write_text("synthetic credential")
    with pytest.raises(ValueError, match="model observation"):
        phases.run_codex_phase(
            config,
            phase="codex-missing-model",
            prompt="synthetic prompt",
            executable=Path("codex.exe"),
            auth_source=source,
            model="codex-pinned",
        )


def test_codex_model_pin_rejects_mismatched_observation(config, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="model observation"):
        _run_codex_pinned(
            config,
            tmp_path / "auth.json",
            monkeypatch,
            model="codex-pinned",
            observed_model="wrong-model",
        )
    record = json.loads((config.root / "codex-pinned" / "process.json").read_text())
    assert record["requested_model"] == "codex-pinned"
    assert record["observed_models"] == ["wrong-model"]
    assert record["model_verified"] is False
    assert record["status"] == "failed"


@pytest.mark.parametrize("observations", [["codex-pinned", "wrong-model"], ["codex-pinned", None]])
def test_codex_model_pin_rejects_later_bad_observation(config, tmp_path, monkeypatch, observations):
    with pytest.raises(ValueError, match="model observation"):
        _run_codex_pinned(
            config,
            tmp_path / "auth.json",
            monkeypatch,
            observations=observations,
        )
    record = json.loads((config.root / "codex-pinned" / "process.json").read_text())
    assert record["status"] == "failed"
    assert record["auth_copy_removed"] is True


@pytest.mark.parametrize("user_messages", [[], ["synthetic prompt", "/exit"], ["changed prompt"]])
def test_codex_model_pin_rejects_prompt_integrity_failure(
    config, tmp_path, monkeypatch, user_messages
):
    with pytest.raises(ValueError, match="user prompt observation"):
        _run_codex_pinned(
            config,
            tmp_path / "auth.json",
            monkeypatch,
            user_messages=user_messages,
        )
    record = json.loads((config.root / "codex-pinned" / "process.json").read_text())
    assert record["user_prompt_count"] == len(user_messages)
    assert record["user_prompt_verified"] is False
    assert record["status"] == "failed"
    assert record["auth_copy_removed"] is True


@pytest.mark.parametrize("model", ["", "bad/model", "x" * 129])
def test_codex_model_pin_validates_slug_before_phase(config, model, monkeypatch):
    monkeypatch.setattr(phases.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(phases.sys, "stdout", SimpleNamespace(isatty=lambda: True))
    with pytest.raises(ValueError, match="model"):
        phases.run_codex_phase(
            config,
            phase="codex-invalid-model",
            prompt="synthetic prompt",
            executable=Path("codex.exe"),
            auth_source=Path("unused"),
            model=model,
        )
    assert not (config.root / "codex-invalid-model").exists()


@pytest.mark.parametrize(
    "model,target,match",
    [(None, "gpt-5.6-sol", "requires a pinned model"), ("codex-pinned", "bad/model", "target")],
)
def test_codex_migration_notice_validates_before_phase(config, model, target, match, monkeypatch):
    monkeypatch.setattr(phases.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(phases.sys, "stdout", SimpleNamespace(isatty=lambda: True))
    with pytest.raises(ValueError, match=match):
        phases.run_codex_phase(
            config,
            phase="codex-invalid-migration",
            prompt="synthetic prompt",
            executable=Path("codex.exe"),
            auth_source=Path("unused"),
            model=model,
            migration_notice_target=target,
        )
    assert not (config.root / "codex-invalid-migration").exists()


class FakeProcess:
    pid = 12345

    def __init__(self, *, times_out=False):
        self.returncode = None
        self.times_out = times_out
        self.waits = []

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.times_out and self.returncode is None:
            raise subprocess.TimeoutExpired("synthetic-host", timeout)
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def kill(self):
        self.returncode = -9


def _run_claude(config, source, **kwargs):
    return phases.run_claude_phase(
        config,
        phase="test-phase",
        prompt="synthetic fixture prompt",
        executable=Path("synthetic-claude.exe"),
        auth_source=source,
        allowed_tools=[phases.CLAUDE_PREFIX + "mnemos_retrieve"],
        **kwargs,
    )


@pytest.mark.parametrize("failure", ["spawn", "timeout", "none"])
def test_authentication_cleaned_on_process_failure_timeout_and_success(
    config, tmp_path, monkeypatch, failure
):
    source = tmp_path / ".credentials.json"
    source.write_text("synthetic credential")
    process = FakeProcess(times_out=failure == "timeout")

    def launch(command, **kwargs):
        home = Path(kwargs["env"]["CLAUDE_CONFIG_DIR"])
        assert (home / source.name).read_text() == "synthetic credential"
        assert kwargs["shell"] is False
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert command[command.index("--max-budget-usd") + 1] == "0.5"
        if failure == "spawn":
            raise OSError("synthetic launch failure")
        return process

    monkeypatch.setattr(phases.subprocess, "Popen", launch)
    monkeypatch.setattr(
        phases,
        "terminate_owned_process",
        lambda process: process.kill() if process.poll() is None else None,
    )
    if failure == "spawn":
        with pytest.raises(OSError, match="launch failure"):
            _run_claude(config, source)
    else:
        result = _run_claude(config, source)
        assert result["status"] == ("failed" if failure == "timeout" else "completed")
        assert result["timed_out"] == (failure == "timeout")
        assert process.waits == [180]
    record = json.loads((config.root / "test-phase" / "process.json").read_text())
    assert record["auth_copy_removed"] is True
    assert not list(config.root.rglob(".credentials.json"))
    assert source.read_text() == "synthetic credential"


def test_cleanup_error_cannot_leave_a_success_report_or_skip_auth_removal(config, monkeypatch):
    auth = config.root / "auth.json"
    auth.write_text("synthetic credential")
    process = FakeProcess()

    def fail_termination(process):
        raise OSError("synthetic termination error")

    monkeypatch.setattr(phases, "terminate_owned_process", fail_termination)
    result = {"status": "completed"}
    with pytest.raises(RuntimeError, match="cleanup failed"):
        phases._cleanup_phase(config.root, result, process, auth, phases.time.monotonic())
    report = json.loads((config.root / "process.json").read_text())
    assert report["status"] == "failed"
    assert report["auth_copy_removed"] is True
    assert report["cleanup_errors"] == [
        "process_termination:OSError",
        "owned_process_still_running",
    ]


@pytest.mark.parametrize(
    "bounds", [{"max_seconds": 181}, {"max_budget_usd": 0.51}, {"max_budget_usd": 0}]
)
def test_invalid_bounds_rejected_before_creating_phase(config, bounds):
    with pytest.raises(ValueError, match="bounds"):
        _run_claude(config, Path("unused"), **bounds)
    assert not (config.root / "test-phase").exists()


def test_unknown_mnemos_prefixed_tool_rejected_before_phase(config):
    with pytest.raises(ValueError, match="named Mnemos"):
        phases.run_claude_phase(
            config,
            phase="bad",
            prompt="unused",
            executable=Path("unused"),
            auth_source=Path("unused"),
            allowed_tools=[phases.CLAUDE_PREFIX + "execute_shell"],
        )
    assert not (config.root / "bad").exists()


@pytest.mark.skipif(phases.os.name != "nt", reason="Windows taskkill race")
def test_taskkill_error_after_natural_exit_is_not_a_cleanup_failure(monkeypatch):
    process = FakeProcess()

    def taskkill(command, **kwargs):
        assert command == ["taskkill", "/PID", "12345", "/T", "/F"]
        process.returncode = 0
        return SimpleNamespace(returncode=128)

    monkeypatch.setattr(phases.subprocess, "run", taskkill)
    phases.terminate_owned_process(process)
    assert process.returncode == 0
