"""Resident hooks retain capture durability, routing checks and host output."""

from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest

from mnemos import hook_client
from mnemos.client import MnemosClient, Outbox
from mnemos.hook_worker import HookDispatcher, HookWorker
from mnemos.scopes import ScopeRegistry
from mnemos.service import MnemosService


def test_non_windows_hook_uses_python_transport(monkeypatch):
    from mnemos import _winhttp_hook

    calls = []

    def python_post(port, token, wire, timeout):
        calls.append((port, token, wire, timeout))
        return b"{}"

    def forbidden_native(*args, **kwargs):
        raise AssertionError("Non-Windows hook attempted the native transport")

    monkeypatch.setattr(hook_client, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(hook_client, "_post_http", python_post)
    monkeypatch.setattr(_winhttp_hook, "post", forbidden_native)
    assert hook_client._post(12345, "synthetic-token", b"request", 2.0) == b"{}"
    assert calls == [(12345, "synthetic-token", b"request", 2.0)]


def test_environment_binding_preserves_case_sensitive_variable_names():
    first = {"PATH": "first", "Path": "unchanged"}
    second = {"PATH": "different", "Path": "unchanged"}
    assert hook_client.environment_fingerprint(first) != hook_client.environment_fingerprint(second)
    assert hook_client.environment_fingerprint(first) == hook_client.environment_fingerprint(
        {**first, hook_client.WORKER_DESCRIPTOR_ENV: "routing-only"}
    )


@pytest.fixture
def configured(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.upper().startswith("MNEMOS_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MNEMOS_CONFIG_PATH", str(tmp_path / "settings.toml"))
    monkeypatch.setenv("MNEMOS_SCOPE_REGISTRY", str(tmp_path / "scopes.json"))
    monkeypatch.setenv("MNEMOS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("MNEMOS_RUNTIME_DESCRIPTOR", str(tmp_path / "memory-runtime.json"))
    monkeypatch.setenv("MNEMOS_USER_PROFILE_ID", "profile")
    monkeypatch.setenv("MNEMOS_AUTO_START", "false")
    monkeypatch.setenv("MNEMOS_CLAUDE_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="profile", enabled=True)
    registry.save(tmp_path / "scopes.json")
    return tmp_path


def payload(root, *, turn="turn-1", event="Stop"):
    return {
        "hook_event_name": event,
        "cwd": str(root),
        "session_id": "session",
        "turn_id": turn,
        "timestamp": "2026-11-07T00:00:00+00:00",
        "last_assistant_message": "Keep the bounded test command.",
        "prompt": "Keep the bounded test command.",
    }


def send(root, value, *, host="codex"):
    return hook_client.request_hook(
        host, value["hook_event_name"], value, cwd=str(root), descriptor_path=root / "worker.json"
    )


class MemoryDispatcher:
    def __init__(self, root):
        self.root = root
        self.captures = []

    async def dispatch(self, operation, value):
        if operation == "capture":
            # The durable record must exist before the memory service sees it.
            queued = Outbox(self.root / "outbox.json").pending()
            assert any(r["payload"]["event_id"] == value["event_id"] for r in queued)
            self.captures.append(value)
            return {"status": "committed", "committed": True}
        return {"facts": [], "status": "ok"}


def test_worker_outage_capture_survives_worker_restart_and_replays(configured):
    root = configured
    worker = HookWorker("codex", descriptor_path=root / "worker.json", cwd=root)
    worker.start()
    try:
        result = send(root, payload(root))
        assert result["status"] == "handled" and result["degraded"] is True
        assert result["captured"] == 1 and result["output"] == {"continue": True}
    finally:
        worker.stop()
    assert len(Outbox(root / "outbox.json").pending()) == 1
    assert not (root / "worker.json").exists()
    recorder = MemoryDispatcher(root)
    memory = MnemosService(dispatcher=recorder, descriptor_path=root / "memory-runtime.json")
    memory.start()
    restarted = HookWorker("codex", descriptor_path=root / "worker.json", cwd=root)
    restarted.start()
    try:
        result = send(root, payload(root, event="SessionStart"))
        assert result["status"] == "handled" and not result["degraded"]
        assert len(recorder.captures) == 1
        assert not Outbox(root / "outbox.json").pending()
    finally:
        restarted.stop()
        memory.stop()


def test_worker_checks_outbox_written_by_another_client_between_events(configured):
    root = configured
    recorder = MemoryDispatcher(root)
    memory = MnemosService(dispatcher=recorder, descriptor_path=root / "memory-runtime.json")
    memory.start()
    worker = HookWorker("codex", descriptor_path=root / "worker.json", cwd=root)
    worker.start()
    try:
        assert send(root, payload(root, event="SessionStart"))["status"] == "handled"
        Outbox(root / "outbox.json").append({"event_id": "external", "content": "Queued elsewhere"})
        assert send(root, payload(root, event="SessionStart"))["status"] == "handled"
        assert [r["event_id"] for r in recorder.captures] == ["external"]
    finally:
        worker.stop()
        memory.stop()


def test_membership_is_reloaded_and_secrets_never_reach_outbox(configured):
    root = configured
    worker = HookWorker("codex", descriptor_path=root / "worker.json", cwd=root)
    worker.start()
    try:
        value = payload(root)
        value["last_assistant_message"] = "password=synthetic-private-value"
        result = send(root, value)
        assert result["captured"] == 0 and result["degraded"]
        assert not (root / "outbox.json").exists()
        registry = ScopeRegistry.load(root / "scopes.json")
        next(iter(registry.projects.values())).enabled = False
        registry.save(root / "scopes.json")
        result = send(root, payload(root, turn="turn-2"))
        assert result["adapter_status"] == "manual_mcp" and result["captured"] == 0
        assert not (root / "outbox.json").exists()
    finally:
        worker.stop()


@pytest.mark.parametrize("change", ["settings", "environment", "directory"])
def test_changed_configuration_declines_before_capture(configured, monkeypatch, change):
    root = configured
    worker = HookWorker("codex", descriptor_path=root / "worker.json", cwd=root)
    worker.start()
    try:
        if change == "settings":
            (root / "settings.toml").write_text("[modules]\nsleep = true\n")
        elif change == "environment":
            monkeypatch.setenv("HOME", str(root / "different-home"))
        else:
            nested = root / "nested"
            nested.mkdir()
            monkeypatch.chdir(nested)
        assert send(root, payload(root)) is None
        assert not (root / "outbox.json").exists()
    finally:
        worker.stop()


def test_memory_descriptor_and_lock_cannot_be_reused(configured):
    for name in ("memory-runtime.json", "memory-runtime.txt"):
        with pytest.raises(ValueError, match="memory service descriptor"):
            HookWorker("codex", descriptor_path=configured / name, cwd=configured)


@pytest.mark.asyncio
async def test_each_event_allows_a_new_bounded_startup_attempt(configured, monkeypatch):
    client = MnemosClient(auto_start=True)
    attempts = []

    def spawn():
        if not client._autostart_attempted:
            attempts.append(1)
            client._autostart_attempted = True

    async def unavailable(**kwargs):
        return None

    monkeypatch.setattr(client, "_spawn_once", spawn)
    monkeypatch.setattr(client, "_wait_for_descriptor", unavailable)
    dispatcher = HookDispatcher("codex", cwd=configured, client=client)
    try:
        for turn in ("one", "two"):
            request = {
                "hook_protocol": 1,
                "request_id": turn,
                "host": "codex",
                "event": "Stop",
                "payload": payload(configured, turn=turn),
                "cwd": str(configured),
                "environment_sha256": hook_client.environment_fingerprint(),
            }
            result = await dispatcher.dispatch("hook", request)
            assert result["captured"] == 1 and result["degraded"]
        assert len(attempts) == 2 and len(client.outbox.pending()) == 2
    finally:
        await dispatcher.close()


def test_wrong_authentication_does_not_dispatch(configured):
    worker = HookWorker("codex", descriptor_path=configured / "worker.json", cwd=configured)
    descriptor = worker.start()
    try:
        with httpx.Client(trust_env=False) as client:
            response = client.post(
                f'http://127.0.0.1:{descriptor["port"]}/v1/dispatch',
                json={"operation": "hook", "payload": payload(configured)},
            )
        assert response.status_code == 401
        assert not (configured / "outbox.json").exists()
    finally:
        worker.stop()


def test_claude_prompt_uses_existing_provisional_capture_and_output(configured):
    root = configured
    worker = HookWorker("claude-code", descriptor_path=root / "worker.json", cwd=root)
    worker.start()
    try:
        result = send(root, payload(root, event="UserPromptSubmit"), host="claude-code")
        assert result["status"] == "handled" and result["captured"] == 1
        assert result["degraded"] and result["output"] == {}
        queued = Outbox(root / "outbox.json").pending()
        assert queued and queued[0]["payload"]["source_kind"] == "claude_code_transcript"
    finally:
        worker.stop()


def test_missing_worker_falls_back_to_the_original_hook(configured, monkeypatch, capsys):
    monkeypatch.setenv(hook_client.WORKER_DESCRIPTOR_ENV, str(configured / "absent.json"))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload(configured))))
    assert hook_client.main(["codex"]) == 0
    assert json.loads(capsys.readouterr().out) == {"continue": True}
    assert len(Outbox(configured / "outbox.json").pending()) == 1


def test_unknown_worker_outcome_never_falls_back_or_prints_payload(configured, monkeypatch, capsys):
    monkeypatch.setenv(hook_client.WORKER_DESCRIPTOR_ENV, str(configured / "worker.json"))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload(configured))))

    def unknown(*args, **kwargs):
        raise hook_client.HookOutcomeUnknown("sensitive response text")

    def forbidden(*args, **kwargs):
        pytest.fail("Ambiguous outcome must not be retried")

    monkeypatch.setattr(hook_client, "request_hook", unknown)
    monkeypatch.setattr(hook_client, "_fallback", forbidden)
    assert hook_client.main(["codex"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == {"continue": True}
    assert "sensitive" not in output.err and "Keep the bounded" not in output.err
    assert "HookOutcomeUnknown" in output.err


def test_fast_command_imports_no_runtime_dependencies(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc,sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in {'pydantic','httpx','mnemos.settings','mnemos.client','mnemos.service','mnemos.engine','numpy','mnemos.scopes'}:
            raise AssertionError('Fast command imported '+fullname)
sys.path.insert(0,sys.argv[1]);sys.meta_path.insert(0,BlockRuntime())
from mnemos.hook_client import main
assert main(['codex']) == 0
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(root)],
        input="{}",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_real_command_uses_worker_without_importing_adapter_runtime(configured, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv(hook_client.WORKER_DESCRIPTOR_ENV, str(configured / "worker.json"))
    recorder = MemoryDispatcher(configured)
    memory = MnemosService(dispatcher=recorder, descriptor_path=configured / "memory-runtime.json")
    memory.start()
    worker = HookWorker("codex", descriptor_path=configured / "worker.json", cwd=configured)
    worker.start()
    script = """
import importlib.abc,sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in {'pydantic','httpx','mnemos.settings','mnemos.client','mnemos.service','mnemos.scopes','mnemos.adapters.codex'}:
            raise AssertionError('Fast command fell back or imported '+fullname)
sys.path.insert(0,sys.argv[1]);sys.meta_path.insert(0,BlockRuntime())
from mnemos.hook_client import main
raise SystemExit(main(['codex']))
"""
    try:
        for turn in ("native-command-one", "native-command-two"):
            result = subprocess.run(
                [sys.executable, "-I", "-c", script, str(root)],
                input=json.dumps(payload(configured, turn=turn)),
                text=True,
                capture_output=True,
                cwd=configured,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            assert result.returncode == 0 and not result.stderr, result.stderr
            assert json.loads(result.stdout) == {"continue": True}
        assert len(recorder.captures) == 2
        assert recorder.captures[0]["event_id"] != recorder.captures[1]["event_id"]
        assert not Outbox(configured / "outbox.json").pending()
    finally:
        worker.stop()
        memory.stop()


def test_lost_reply_after_durable_capture_is_unknown_without_resend(configured):
    dispatcher = HookDispatcher("codex", cwd=configured)

    class LostReply:
        async def dispatch(self, operation, request):
            result = await dispatcher.dispatch(operation, request)
            assert result["captured"] == 1
            raise RuntimeError("Simulated failure after durable capture")

        async def close(self):
            await dispatcher.close()

    service = MnemosService(dispatcher=LostReply(), descriptor_path=configured / "worker.json")
    descriptor = service.start()
    descriptor.update(
        hook_protocol=1,
        adapter_host="codex",
        environment_sha256=hook_client.environment_fingerprint(),
    )
    from mnemos.service import _write_private_json

    _write_private_json(configured / "worker.json", descriptor)
    try:
        with pytest.raises(hook_client.HookOutcomeUnknown):
            send(configured, payload(configured))
        pending = Outbox(configured / "outbox.json").pending()
        assert (
            len(pending) == 1
            and pending[0]["payload"]["content"] == "Keep the bounded test command."
        )
    finally:
        service.stop()


@pytest.mark.parametrize("fault", ["request_id", "host", "environment", "protocol", "late_decline"])
def test_misbound_or_ambiguous_replies_never_allow_fallback(configured, fault):
    class WrongReply:
        def dispatch(self, operation, request):
            response = {
                "hook_protocol": 1,
                "request_id": request["request_id"],
                "host": "codex",
                "environment_sha256": request["environment_sha256"],
                "status": "handled",
                "dispatched": True,
                "adapter_status": "ok",
                "output": {"continue": True},
            }
            if fault == "request_id":
                response["request_id"] = "another-request"
            elif fault == "host":
                response["host"] = "claude-code"
            elif fault == "environment":
                response["environment_sha256"] = "another-environment"
            elif fault == "protocol":
                response["hook_protocol"] = True
            else:
                response["status"] = "declined"
            return response

    service = MnemosService(dispatcher=WrongReply(), descriptor_path=configured / "worker.json")
    descriptor = service.start()
    descriptor.update(
        hook_protocol=1,
        adapter_host="codex",
        environment_sha256=hook_client.environment_fingerprint(),
    )
    from mnemos.service import _write_private_json

    _write_private_json(configured / "worker.json", descriptor)
    try:
        with pytest.raises(hook_client.HookOutcomeUnknown):
            send(configured, payload(configured))
    finally:
        service.stop()
