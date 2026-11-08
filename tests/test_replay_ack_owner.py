"""Failure handling in the disposable comparison owner."""

import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


def helper():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmarks/native_hosts/replay_ack_owner.py")
    )


def test_cleanup_attempts_every_resource_after_failure():
    calls = []

    def fail():
        calls.append("worker")
        raise RuntimeError("stop failed")

    errors = helper()["close_resources"](
        SimpleNamespace(stop=fail),
        SimpleNamespace(stop=lambda: calls.append("memory")),
        [SimpleNamespace(close=lambda: calls.append("store"))],
    )
    assert calls == ["worker", "memory", "store"]
    assert errors == [{"resource": "worker", "error_type": "RuntimeError"}]


def test_control_sharing_violation_recovers(monkeypatch):
    module = helper()
    attempts = []

    def read():
        attempts.append(True)
        if len(attempts) == 1:
            raise PermissionError("sharing violation")
        return b'{"operation":"stop"}'

    monkeypatch.setattr(module["time"], "sleep", lambda _: None)
    assert module["read_control"](SimpleNamespace(read_bytes=read)) == {"operation": "stop"}
    assert len(attempts) == 2


def test_cleanup_preserves_stores_owned_by_live_thread():
    calls = []

    def stop():
        raise TimeoutError("Owner still closing")

    errors = helper()["close_resources"](
        SimpleNamespace(stop=lambda: None),
        SimpleNamespace(stop=stop, _owner_thread=SimpleNamespace(is_alive=lambda: True)),
        [SimpleNamespace(close=lambda: calls.append("closed"))],
    )
    assert calls == []
    assert errors == [
        {"resource": "memory", "error_type": "TimeoutError"},
        {"resource": "stores", "error_type": "OwnerStillAlive"},
    ]


@pytest.mark.parametrize(
    "missing",
    [
        "MNEMOS_CONFIG_PATH",
        "MNEMOS_SCOPE_REGISTRY",
        "MNEMOS_RUNTIME_DIR",
        "MNEMOS_RUNTIME_DESCRIPTOR",
        "MNEMOS_HOOK_WORKER_DESCRIPTOR",
        "MNEMOS_AUTO_START",
    ],
)
def test_missing_isolation_variable_fails_before_start(tmp_path, monkeypatch, missing):
    folder = tmp_path / "candidate"
    settings = {
        "MNEMOS_CONFIG_PATH": tmp_path / "project/settings.toml",
        "MNEMOS_SCOPE_REGISTRY": tmp_path / "scopes.json",
        "MNEMOS_RUNTIME_DIR": folder,
        "MNEMOS_RUNTIME_DESCRIPTOR": folder / "memory.json",
        "MNEMOS_HOOK_WORKER_DESCRIPTOR": folder / "worker.json",
        "MNEMOS_AUTO_START": "false",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, str(value))
    validate = helper()["validate_paths"]
    validate(tmp_path, "candidate", folder / "memory.json", tmp_path / "scopes.json")
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError):
        validate(tmp_path, "candidate", folder / "memory.json", tmp_path / "scopes.json")


def test_control_sharing_violation_has_deadline(monkeypatch):
    module = helper()
    clock = iter([0, 3])
    monkeypatch.setattr(module["time"], "monotonic", lambda: next(clock))

    def read():
        raise PermissionError("sharing violation")

    with pytest.raises(PermissionError):
        module["read_control"](SimpleNamespace(read_bytes=read))


def test_malformed_control_is_not_retried():
    attempts = []

    def read():
        attempts.append(True)
        return b"{"

    with pytest.raises(json.JSONDecodeError):
        helper()["read_control"](SimpleNamespace(read_bytes=read))
    assert len(attempts) == 1
