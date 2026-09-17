from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from pathlib import Path

import httpx
import pytest

from mnemos.client import MnemosClient, Outbox
from mnemos.service import MnemosService, SCHEMA_VERSION, read_runtime_descriptor


class RecordingDispatcher:
    def __init__(self, *, committed: bool = True) -> None:
        self.committed = committed
        self.calls: list[tuple[str, dict, int | None]] = []

    async def dispatch(self, operation: str, payload: dict) -> dict:
        self.calls.append((operation, payload, threading.get_ident()))
        await asyncio.sleep(0)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "committed" if self.committed else "pending",
            "committed": self.committed,
            "operation": operation,
        }


@pytest.mark.asyncio
async def test_closing_host_client_does_not_terminate_shared_runtime(tmp_path):
    class RunningServiceProcess:
        def poll(self):
            return None

        def terminate(self):
            raise AssertionError("Host client must not terminate the shared service")

        def kill(self):
            raise AssertionError("Host client must not kill the shared service")

    client = MnemosClient(descriptor_path=tmp_path / "runtime.json", auto_start=False)
    client._child = RunningServiceProcess()
    await client.close()


@pytest.fixture
def descriptor_path(tmp_path: Path) -> Path:
    return tmp_path / "runtime.json"


def start_service(path: Path, dispatcher: RecordingDispatcher) -> MnemosService:
    service = MnemosService(
        dispatcher=dispatcher,
        descriptor_path=path,
        startup_timeout=2,
        dispatch_timeout=2,
    )
    service.start()
    return service


def test_duplicate_launcher_exits_without_stopping_owner(descriptor_path: Path) -> None:
    service = start_service(descriptor_path, RecordingDispatcher())
    original = service.descriptor
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mnemos.service", "--descriptor", str(descriptor_path)],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert result.returncode == 0, result.stderr
        assert read_runtime_descriptor(descriptor_path) == original
        assert service.submit("health", {})["committed"] is True
    finally:
        service.stop()


@pytest.mark.asyncio
async def test_loopback_dispatch_auth_and_origin(descriptor_path: Path) -> None:
    dispatcher = RecordingDispatcher()
    service = start_service(descriptor_path, dispatcher)
    try:
        descriptor = read_runtime_descriptor(descriptor_path)
        assert descriptor is not None
        url = f"http://127.0.0.1:{descriptor['port']}/v1/dispatch"
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {descriptor['token']}"},
                json={"operation": "capture", "payload": {"event_id": "e1"}},
            )
            assert response.status_code == 200
            assert response.json()["committed"] is True

            unauthorized = await client.post(url, json={"operation": "capture", "payload": {}})
            assert unauthorized.status_code == 401
            forbidden = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {descriptor['token']}",
                    "Origin": "https://evil.example",
                },
                json={"operation": "capture", "payload": {}},
            )
            assert forbidden.status_code == 403
        assert dispatcher.calls[0][0] == "capture"
    finally:
        service.stop()


@pytest.mark.asyncio
async def test_client_autostarts_once_and_waits_for_descriptor(
    descriptor_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatcher = RecordingDispatcher()
    service = MnemosService(
        dispatcher=dispatcher,
        descriptor_path=descriptor_path,
        startup_timeout=2,
    )
    starts = 0

    class FakeProcess:
        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            return

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def fake_popen(*args, **kwargs) -> FakeProcess:
        nonlocal starts
        starts += 1
        service.start()
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    client = MnemosClient(
        descriptor_path=descriptor_path,
        auto_start=True,
        startup_timeout=2,
        request_timeout=2,
    )
    try:
        result = await client.request("retrieve", {"query": "local"}, queue_on_failure=False)
        assert result["committed"] is True
        assert starts == 1
    finally:
        service.stop()


@pytest.mark.asyncio
async def test_invalid_json_and_max_body_are_rejected(descriptor_path: Path) -> None:
    service = MnemosService(
        dispatcher=RecordingDispatcher(),
        descriptor_path=descriptor_path,
        max_body_bytes=80,
        startup_timeout=2,
    )
    service.start()
    try:
        descriptor = service.descriptor
        assert descriptor is not None
        url = f"http://127.0.0.1:{descriptor['port']}/dispatch"
        headers = {"Authorization": f"Bearer {descriptor['token']}"}
        async with httpx.AsyncClient(trust_env=False) as client:
            bad = await client.post(url, headers=headers, content=b"{")
            assert bad.status_code == 400
            large = await client.post(
                url,
                headers=headers,
                content=json.dumps({"operation": "capture", "payload": {"content": "x" * 200}}),
            )
            assert large.status_code == 413
    finally:
        service.stop()


@pytest.mark.asyncio
async def test_dispatcher_runs_on_single_owner_loop_thread(descriptor_path: Path) -> None:
    dispatcher = RecordingDispatcher()
    service = start_service(descriptor_path, dispatcher)
    try:
        client = MnemosClient(
            descriptor_path=descriptor_path,
            auto_start=False,
            request_timeout=2,
        )
        result = await client.request("health", {})
        assert result["committed"] is True
        assert dispatcher.calls[0][2] == service.owner_thread_id
    finally:
        service.stop()


def test_simultaneous_startup_returns_one_owned_descriptor(descriptor_path: Path) -> None:
    first = MnemosService(
        dispatcher=RecordingDispatcher(), descriptor_path=descriptor_path, startup_timeout=2
    )
    second = MnemosService(
        dispatcher=RecordingDispatcher(), descriptor_path=descriptor_path, startup_timeout=2
    )
    try:
        first_descriptor = first.start()
        second_descriptor = second.start()
        assert first_descriptor["port"] == second_descriptor["port"]
        assert first.owner_thread_id is not None
        assert second.owner_thread_id is None
    finally:
        second.stop()
        first.stop()


def test_concurrent_startup_waits_for_the_single_owner(descriptor_path: Path) -> None:
    services = [
        MnemosService(
            dispatcher=RecordingDispatcher(),
            descriptor_path=descriptor_path,
            startup_timeout=2,
        )
        for _ in range(2)
    ]
    barrier = threading.Barrier(2)
    results: list[dict] = []
    errors: list[BaseException] = []

    def start(service: MnemosService) -> None:
        try:
            barrier.wait()
            results.append(service.start())
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=start, args=(service,)) for service in services]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
    try:
        assert not errors
        assert len(results) == 2
        assert results[0]["port"] == results[1]["port"]
    finally:
        for service in services:
            service.stop()


@pytest.mark.asyncio
async def test_service_restart_publishes_fresh_descriptor(descriptor_path: Path) -> None:
    first = start_service(descriptor_path, RecordingDispatcher())
    first_descriptor = first.descriptor
    assert first_descriptor is not None
    first.stop()
    assert read_runtime_descriptor(descriptor_path) is None

    second_dispatcher = RecordingDispatcher()
    second = start_service(descriptor_path, second_dispatcher)
    try:
        second_descriptor = second.descriptor
        assert second_descriptor is not None
        assert second_descriptor["token"] != first_descriptor["token"]
        client = MnemosClient(descriptor_path=descriptor_path, auto_start=False)
        result = await client.request("health", {})
        assert result["committed"] is True
        assert len(second_dispatcher.calls) == 1
    finally:
        second.stop()


@pytest.mark.asyncio
async def test_client_queues_sanitized_capture_and_replays_once(
    descriptor_path: Path, tmp_path: Path
) -> None:
    outbox_path = tmp_path / "events.json"
    client = MnemosClient(
        descriptor_path=descriptor_path,
        outbox_path=outbox_path,
        auto_start=False,
        request_timeout=0.2,
    )
    queued = await client.capture(
        {"event_id": "stable-event", "content": "Remember me at person@example.com"}
    )
    assert queued["status"] == "queued"
    assert queued["committed"] is False
    assert "person@example.com" not in outbox_path.read_text(encoding="utf-8")
    assert len(client.outbox) == 1

    dispatcher = RecordingDispatcher()
    service = start_service(descriptor_path, dispatcher)
    try:
        replay = await client.replay_outbox()
        assert replay["acknowledged"] == 1
        assert len(client.outbox) == 0
        assert dispatcher.calls[0][1]["event_id"] == "stable-event"
        assert "person@example.com" not in dispatcher.calls[0][1]["content"]
        replay_again = await client.replay_outbox()
        assert replay_again["acknowledged"] == 0
        assert len(dispatcher.calls) == 1
    finally:
        service.stop()


@pytest.mark.asyncio
async def test_uncommitted_replay_retains_outbox_entry(
    descriptor_path: Path, tmp_path: Path
) -> None:
    outbox = Outbox(tmp_path / "events.json")
    event_id = outbox.append({"event_id": "event-1", "content": "safe content"})
    assert event_id == "event-1"
    dispatcher = RecordingDispatcher(committed=False)
    service = start_service(descriptor_path, dispatcher)
    client = MnemosClient(descriptor_path=descriptor_path, outbox=outbox, auto_start=False)
    try:
        replay = await client.replay_outbox()
        assert replay["acknowledged"] == 0
        assert len(outbox) == 1
    finally:
        service.stop()


def test_outbox_concurrent_writers_keep_every_profile_event(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "shared.json"

    def append(index):
        return Outbox(path).append(
            {
                "event_id": "same-native-id",
                "content": "safe fact",
                "context": {
                    "user_profile_id": f"profile-{index}",
                    "host": "codex",
                    "session_id": "s",
                },
            }
        )

    with ThreadPoolExecutor(max_workers=4) as workers:
        ids = list(workers.map(append, range(12)))
    assert len(set(ids)) == 12
    assert len(Outbox(path).pending()) == 12


def test_outbox_rejects_secret_source_metadata_before_persistence(tmp_path):
    outbox = Outbox(tmp_path / "blocked.json")
    with pytest.raises(ValueError):
        outbox.append({"event_id": "one", "content": "safe", "source_ref": "api_key=neverpersist"})
    assert not outbox.path.exists()


def test_slow_dispatcher_close_keeps_ownership_until_stop_completes(tmp_path: Path) -> None:
    from mnemos.service import _ProcessLock

    closing = threading.Event()
    release = threading.Event()

    class SlowClose(RecordingDispatcher):
        async def close(self) -> None:
            closing.set()
            for _ in range(500):
                if release.is_set():
                    return
                await asyncio.sleep(0.01)

    service = MnemosService(
        dispatcher=SlowClose(),
        descriptor_path=tmp_path / "slow-close.json",
        shutdown_timeout=0.05,
    )
    service.start()
    contender = _ProcessLock(service.lock_path)
    try:
        with pytest.raises(TimeoutError, match="shutting down"):
            service.stop()
        assert closing.is_set()
        assert service._owner_thread is not None and service._owner_thread.is_alive()
        assert not contender.acquire()
        with pytest.raises(TimeoutError, match="shutting down"):
            service.stop()
        assert not contender.acquire()
        release.set()
        service._owner_thread.join(timeout=2)
        service.stop()
        assert contender.acquire()
        assert not service.descriptor_path.exists()
    finally:
        release.set()
        contender.release()
        service.stop()
