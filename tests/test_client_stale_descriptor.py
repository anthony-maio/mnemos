"""Recover from dead descriptors without displacing a reachable service owner."""

from __future__ import annotations

import asyncio
import socket
import ssl
from pathlib import Path
from typing import Any

import pytest
import httpx

import mnemos.client as client_module
from mnemos.client import MnemosClient, ServiceClientError
from mnemos.service import MnemosService, _write_private_json


@pytest.fixture(autouse=True)
def enable_default_autostart(monkeypatch: pytest.MonkeyPatch) -> None:
    # These tests exercise the default launcher path; each one replaces Popen.
    monkeypatch.setenv("MNEMOS_AUTO_START", "1")


class RecordingDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((operation, payload))
        return {"schema_version": 1, "status": "committed", "committed": True}


class Launcher:
    def __init__(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code

    def poll(self) -> int | None:
        return self.exit_code


def stale_descriptor(path: Path, reserved_socket: socket.socket) -> dict[str, Any]:
    reserved_socket.bind(("127.0.0.1", 0))
    value = {
        "schema_version": 1,
        "host": "127.0.0.1",
        "port": reserved_socket.getsockname()[1],
        "pid": 123456,
        "token": "dead-fixture-owner",
    }
    # Keep the port reserved but not listening, so a replacement cannot reuse it.
    _write_private_json(path, value)
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["capture", "retrieve"])
async def test_waits_for_replacement_instead_of_retrying_dead_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    path = tmp_path / "runtime.json"
    dispatcher = RecordingDispatcher()
    replacement = MnemosService(dispatcher=dispatcher, descriptor_path=path)
    client = MnemosClient(descriptor_path=path, startup_timeout=2, request_timeout=0.2)
    starts = 0
    scheduled = None
    with socket.socket() as reserved:
        stale = stale_descriptor(path, reserved)
        original_bytes = path.read_bytes()

        def spawn(*args: Any, **kwargs: Any) -> Launcher:
            nonlocal starts, scheduled
            starts += 1
            assert path.read_bytes() == original_bytes
            # Publication happens on the next loop turn, after Popen returns.
            scheduled = asyncio.get_running_loop().call_soon(replacement.start)
            return Launcher()

        monkeypatch.setattr(client_module.subprocess, "Popen", spawn)
        payload = {"event_id": "stale-owner-event", "content": "Durable recovery convention"}
        if operation == "retrieve":
            payload = {"query": "recovery convention"}
        try:
            result = await client.request(operation, payload)
            assert result.get("committed") is True, result
            assert starts == 1
            assert replacement.descriptor is not None
            assert replacement.descriptor["port"] != stale["port"]
            assert dispatcher.calls == [(operation, payload)]
            assert client.outbox.pending() == []
        finally:
            if scheduled is not None:
                scheduled.cancel()
            await client.close()
            replacement.stop()


@pytest.mark.asyncio
async def test_transient_lost_reply_reuses_live_owner_without_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "runtime.json"
    dispatcher = RecordingDispatcher()
    service = MnemosService(dispatcher=dispatcher, descriptor_path=path)
    service.start()
    descriptor_bytes = path.read_bytes()
    client = MnemosClient(descriptor_path=path, startup_timeout=1, request_timeout=1)
    original_dispatch = client._dispatch_http
    attempts = 0

    async def lose_first_reply(
        operation: str, payload: dict[str, Any], owner: dict[str, Any]
    ) -> dict[str, Any]:
        nonlocal attempts
        result = await original_dispatch(operation, payload, owner)
        attempts += 1
        if attempts == 1:
            raise ServiceClientError("lost committed reply", unavailable=True)
        return result

    def unexpected_spawn(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("A reachable existing owner must not launch another service")

    monkeypatch.setattr(client, "_dispatch_http", lose_first_reply)
    monkeypatch.setattr(client_module.subprocess, "Popen", unexpected_spawn)
    payload = {"event_id": "live-owner-event", "content": "Keep the current owner"}
    try:
        result = await client.capture(payload)
        assert result.get("committed") is True
        assert attempts == 2
        # Authenticated GET readiness never enters the mutating dispatcher.
        assert dispatcher.calls == [("capture", payload), ("capture", payload)]
        assert client.outbox.pending() == []
        assert path.read_bytes() == descriptor_bytes
    finally:
        await client.close()
        service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [37, None], ids=["exited", "still_starting"])
async def test_stale_descriptor_does_not_hide_exited_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int | None
) -> None:
    path = tmp_path / "runtime.json"
    client = MnemosClient(descriptor_path=path, startup_timeout=0.15, request_timeout=0.1)
    starts = 0

    def spawn(*args: Any, **kwargs: Any) -> Launcher:
        nonlocal starts
        starts += 1
        return Launcher(exit_code=exit_code)

    monkeypatch.setattr(client_module.subprocess, "Popen", spawn)
    with socket.socket() as reserved:
        stale_descriptor(path, reserved)
        original_bytes = path.read_bytes()
        try:
            result = await client.capture({"event_id": "still-queued", "content": "Retain me"})
            assert result["status"] == "queued" and result["committed"] is False
            if exit_code is None:
                assert "startup timed out" in result["error"]
            else:
                assert "exited before startup" in result["error"] and "37" in result["error"]
            assert starts == 1 and len(client.outbox.pending()) == 1
            assert path.read_bytes() == original_bytes
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_health_authorization_failure_does_not_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "runtime.json"
    service = MnemosService(dispatcher=RecordingDispatcher(), descriptor_path=path)
    owner = service.start()
    _write_private_json(path, {**owner, "token": "invalid-fixture-token"})
    client = MnemosClient(descriptor_path=path, startup_timeout=1, request_timeout=1)

    async def first_connection_fails(
        operation: str, payload: dict[str, Any], descriptor: dict[str, Any]
    ) -> dict[str, Any]:
        raise ServiceClientError("initial connection failed", unavailable=True)

    def unexpected_spawn(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Authorization rejection must not launch a replacement")

    monkeypatch.setattr(client, "_dispatch_http", first_connection_fails)
    monkeypatch.setattr(client_module.subprocess, "Popen", unexpected_spawn)
    try:
        result = await client.retrieve({"query": "owner readiness"})
        assert result["status"] == "degraded"
        assert "401" in result["error"]
    finally:
        await client.close()
        service.stop()


@pytest.mark.asyncio
async def test_owner_recovers_while_competing_launcher_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "runtime.json"
    dispatcher = RecordingDispatcher()
    service = MnemosService(dispatcher=dispatcher, descriptor_path=path)
    service.start()
    original_bytes = path.read_bytes()
    client = MnemosClient(descriptor_path=path, startup_timeout=1, request_timeout=1)
    original_dispatch = client._dispatch_http
    original_probe = client._owner_reachable
    dispatches = probes = starts = 0

    async def first_connection_fails(
        operation: str, payload: dict[str, Any], descriptor: dict[str, Any]
    ) -> dict[str, Any]:
        nonlocal dispatches
        dispatches += 1
        if dispatches == 1:
            raise ServiceClientError("transient connection failure", unavailable=True)
        return await original_dispatch(operation, payload, descriptor)

    async def first_probe_fails(
        descriptor: dict[str, Any], *, timeout: float | None = None
    ) -> bool:
        nonlocal probes
        probes += 1
        if probes == 1:
            return False
        return await original_probe(descriptor, timeout=timeout)

    def spawn(*args: Any, **kwargs: Any) -> Launcher:
        nonlocal starts
        starts += 1
        return Launcher(exit_code=0)

    monkeypatch.setattr(client, "_dispatch_http", first_connection_fails)
    monkeypatch.setattr(client, "_owner_reachable", first_probe_fails)
    monkeypatch.setattr(client_module.subprocess, "Popen", spawn)
    try:
        result = await client.retrieve({"query": "existing owner recovered"})
        assert result.get("committed") is True
        assert dispatches == probes == 2 and starts == 1
        assert dispatcher.calls == [("retrieve", {"query": "existing owner recovered"})]
        assert path.read_bytes() == original_bytes
    finally:
        await client.close()
        service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["health", "dispatch"])
async def test_plaintext_loopback_does_not_load_certificate_bundles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    service = MnemosService(
        dispatcher=RecordingDispatcher(), descriptor_path=tmp_path / "runtime.json"
    )
    owner = service.start()
    client = MnemosClient(descriptor_path=tmp_path / "runtime.json", auto_start=False)
    original_client = httpx.AsyncClient
    options: list[dict[str, Any]] = []

    def record_client(**kwargs: Any) -> httpx.AsyncClient:
        options.append(kwargs)
        return original_client(**kwargs)

    def forbidden_bundle_load(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Plaintext loopback must not load a CA bundle")

    monkeypatch.setattr(ssl, "create_default_context", forbidden_bundle_load)
    monkeypatch.setattr(client_module.httpx, "AsyncClient", record_client)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "absent-cert.pem"))
    try:
        if route == "health":
            assert await client._owner_reachable(owner)
        else:
            assert (await client.retrieve({"query": "local owner"}))["committed"] is True
        assert len(options) == 1
        assert options[0]["trust_env"] is False
        assert options[0]["follow_redirects"] is False
        context = options[0]["verify"]
        assert isinstance(context, ssl.SSLContext)
        assert context.check_hostname is True
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.get_ca_certs() == []
    finally:
        await client.close()
        service.stop()


@pytest.mark.asyncio
async def test_exited_launcher_allows_owner_recovery_until_startup_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "runtime.json"
    client = MnemosClient(descriptor_path=path, startup_timeout=1, request_timeout=1)
    probes: list[float | None] = []
    dispatches = starts = 0

    async def dispatch(
        operation: str, payload: dict[str, Any], owner: dict[str, Any]
    ) -> dict[str, Any]:
        nonlocal dispatches
        dispatches += 1
        if dispatches == 1:
            raise ServiceClientError("transient transport failure", unavailable=True)
        return {"status": "committed", "committed": True}

    async def probe(owner: dict[str, Any], *, timeout: float | None = None) -> bool:
        probes.append(timeout)
        return len(probes) >= 3

    def spawn(*args: Any, **kwargs: Any) -> Launcher:
        nonlocal starts
        starts += 1
        return Launcher(exit_code=0)

    monkeypatch.setattr(client, "_dispatch_http", dispatch)
    monkeypatch.setattr(client, "_owner_reachable", probe)
    monkeypatch.setattr(client_module.subprocess, "Popen", spawn)
    with socket.socket() as reserved:
        stale_descriptor(path, reserved)
        original_bytes = path.read_bytes()
        try:
            result = await client.retrieve({"query": "recovering owner"})
            assert result.get("committed") is True, result
            assert dispatches == 2 and starts == 1 and len(probes) == 3
            assert probes[0] is None
            assert probes[1] is not None and probes[2] is not None
            assert 0 < probes[2] < probes[1] <= client.startup_timeout
            assert path.read_bytes() == original_bytes
        finally:
            await client.close()
