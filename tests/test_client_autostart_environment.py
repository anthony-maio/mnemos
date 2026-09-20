"""Autostart opt-out must survive every service client entrypoint."""

from argparse import Namespace
import io
import json
import socket
import subprocess
from types import SimpleNamespace

import pytest

from mnemos import bridge, managed_cli
from mnemos.adapters.claude_code import ClaudeCodeAdapter
from mnemos.client import MnemosClient, Outbox
from mnemos.service import MnemosService
from tests.test_claude_adapter import registry_for
from tests.test_client_stale_descriptor import RecordingDispatcher, stale_descriptor


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOS_RUNTIME_DESCRIPTOR", str(tmp_path / "runtime.json"))
    monkeypatch.setenv("MNEMOS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("MNEMOS_SQLITE_PATH", str(tmp_path / "memory.sqlite"))

    def forbid_spawn(*args, **kwargs):
        pytest.fail("Disabled client attempted to spawn a service")

    monkeypatch.setattr(
        "mnemos.client.subprocess",
        SimpleNamespace(
            Popen=forbid_spawn,
            DEVNULL=subprocess.DEVNULL,
            CREATE_NO_WINDOW=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ),
    )
    return tmp_path


@pytest.mark.parametrize("value", ["false", "0", "no", "off", " FALSE "])
async def test_disabled_capture_persists_without_spawning(isolated, monkeypatch, value):
    monkeypatch.setenv("MNEMOS_AUTO_START", value)
    client = MnemosClient()
    payload = {"event_id": "outage-capture", "content": "Keep the project marker"}
    result = await client.capture(payload)
    assert result["status"] == "queued"
    assert result["committed"] is False
    assert client.auto_start is False
    assert not client.descriptor_path.exists()
    assert Outbox(isolated / "outbox.json").pending()[0]["payload"] == payload
    assert not (isolated / "memory.sqlite").exists()


@pytest.mark.parametrize("value", [None, "true", "1", "yes", "on", " TRUE "])
def test_default_or_enabled_autostart_remains_enabled(isolated, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("MNEMOS_AUTO_START", raising=False)
    else:
        monkeypatch.setenv("MNEMOS_AUTO_START", value)
    assert MnemosClient().auto_start is True


@pytest.mark.parametrize("value", ["", "disabled", "flase"])
def test_invalid_setting_fails_before_start_or_write(isolated, monkeypatch, value):
    monkeypatch.setenv("MNEMOS_AUTO_START", value)
    with pytest.raises(ValueError, match="MNEMOS_AUTO_START"):
        MnemosClient()
    assert not list(isolated.glob("*.json"))


@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_argument_overrides_environment(isolated, monkeypatch, explicit):
    monkeypatch.setenv("MNEMOS_AUTO_START", "false" if explicit else "true")
    assert MnemosClient(auto_start=explicit).auto_start is explicit


async def test_claude_default_client_queues_prompt_with_autostart_disabled(isolated, monkeypatch):
    monkeypatch.setenv("MNEMOS_AUTO_START", "false")
    adapter = ClaudeCodeAdapter(
        registry=registry_for(isolated),
        user_profile_id="profile-a",
        checkpoint_directory=isolated / "checkpoints",
    )
    await adapter.handle(
        "UserPromptSubmit",
        {"session_id": "outage", "cwd": str(isolated), "prompt": "Keep the project marker"},
    )
    pending = Outbox(isolated / "outbox.json").pending()
    assert len(pending) == 1
    assert pending[0]["payload"]["content"] == "Keep the project marker"
    assert not (isolated / "runtime.json").exists()


async def test_bridge_capture_honors_autostart_opt_out(isolated, monkeypatch):
    monkeypatch.setenv("MNEMOS_AUTO_START", "false")
    request = {"operation": "capture", "payload": {"event_id": "bridge", "content": "marker"}}
    monkeypatch.setattr(
        bridge.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(json.dumps(request).encode()))
    )
    responses = []
    monkeypatch.setattr(bridge, "_write_json_response", responses.append)
    assert (
        await bridge._serve_once(
            descriptor=isolated / "runtime.json", max_request_bytes=1000, deadline=1
        )
        == 0
    )
    assert responses[0]["status"] == "queued"
    assert len(Outbox(isolated / "outbox.json").pending()) == 1


async def test_cli_stats_honors_autostart_opt_out(isolated, monkeypatch):
    monkeypatch.setenv("MNEMOS_AUTO_START", "false")
    result = await managed_cli.run_command(Namespace(command="stats"))
    assert result["status"] == "degraded"
    assert result["committed"] is False
    assert not (isolated / "runtime.json").exists()


async def test_cli_stop_never_autostarts_even_when_env_enables(isolated, monkeypatch):
    monkeypatch.setenv("MNEMOS_AUTO_START", "true")
    result = await managed_cli.run_command(Namespace(command="service-stop"))
    assert result["status"] == "not_running"


async def test_env_disabled_client_can_replay_to_existing_owner(isolated, monkeypatch):
    monkeypatch.setenv("MNEMOS_AUTO_START", "false")
    client = MnemosClient()
    payload = {"event_id": "recovery", "content": "Restore the queued marker"}
    assert (await client.capture(payload))["status"] == "queued"
    dispatcher = RecordingDispatcher()
    owner = MnemosService(dispatcher=dispatcher, descriptor_path=isolated / "runtime.json")
    owner.start()
    recovered = MnemosClient()
    try:
        assert (await recovered.request("health", {}))["committed"] is True
        assert ("capture", payload) in dispatcher.calls
        assert Outbox(isolated / "outbox.json").pending() == []
    finally:
        await recovered.close()
        owner.stop()


async def test_env_disabled_client_preserves_stale_descriptor_and_queues(isolated, monkeypatch):
    monkeypatch.setenv("MNEMOS_AUTO_START", "false")
    descriptor = isolated / "runtime.json"
    with socket.socket() as reserved:
        stale_descriptor(descriptor, reserved)
        before = descriptor.read_bytes()
        client = MnemosClient(request_timeout=0.1)
        try:
            result = await client.capture({"event_id": "stale", "content": "Keep marker"})
            assert result["status"] == "queued"
            assert descriptor.read_bytes() == before
            assert len(Outbox(isolated / "outbox.json").pending()) == 1
        finally:
            await client.close()
