"""Offline resource-boundary tests; these never launch a native host or model."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

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
