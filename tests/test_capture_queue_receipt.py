import pytest

from mnemos.client import MnemosClient, Outbox
from mnemos.service import MnemosService
from mnemos.config import MemorySafetyConfig
from mnemos.memory_safety import MemoryWriteFirewall


def payload():
    return {
        "event_id": "turn-1",
        "content": "The sample color is green.",
        "context": {"user_profile_id": "test", "host": "codex", "session_id": "session-1"},
    }


@pytest.mark.asyncio
async def test_receipt_survives_client_restart_and_replays_original_identity(tmp_path, monkeypatch):
    path = tmp_path / "outbox.json"
    descriptor = tmp_path / "runtime.json"
    client = MnemosClient(descriptor_path=descriptor, outbox=Outbox(path))

    def forbidden(*args, **kwargs):
        raise AssertionError("Queue acknowledgment must not start or contact the service")

    monkeypatch.setattr(client, "_descriptor", forbidden)
    monkeypatch.setattr(client, "_spawn_once", forbidden)
    monkeypatch.setattr(client, "_dispatch_http", forbidden)
    try:
        result = client.enqueue_capture(payload())
        assert result["status"] == "queued" and result["durable_queued"] is True
        assert result["committed"] is False and "job_id" not in result
        assert result["event_id"] == "turn-1"
        entry = Outbox(path).pending()[0]
        assert result["outbox_event_id"] == entry["event_id"] != result["event_id"]
        assert entry["payload"] == payload()
        assert client.enqueue_capture(payload()) == result
        assert len(Outbox(path)) == 1
    finally:
        await client.close()
    calls = []

    class Dispatcher:
        async def dispatch(self, operation, data):
            calls.append((operation, data))
            return {"status": "committed", "committed": True}

    service = MnemosService(
        dispatcher=Dispatcher(), descriptor_path=descriptor, lock_path=tmp_path / "runtime.lock"
    )
    service.start()
    restarted = MnemosClient(descriptor_path=descriptor, outbox=Outbox(path), auto_start=False)
    try:
        replay = await restarted.replay_outbox()
        assert replay["acknowledged"] == 1 and replay["pending"] == 0
        assert calls == [("capture", payload())]
        assert (await restarted.replay_outbox())["acknowledged"] == 0
    finally:
        await restarted.close()
        service.stop()


@pytest.mark.asyncio
async def test_write_failure_has_no_acknowledgment(tmp_path, monkeypatch):
    outbox = Outbox(tmp_path / "outbox.json")

    def fail(*args):
        raise OSError("Simulated disk failure")

    monkeypatch.setattr(outbox, "_write", fail)
    client = MnemosClient(descriptor_path=tmp_path / "runtime.json", outbox=outbox)
    try:
        with pytest.raises(OSError, match="disk failure"):
            client.enqueue_capture(payload())
        assert not outbox.path.exists()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_identity_collision_preserves_original_without_success_receipt(tmp_path):
    client = MnemosClient(
        descriptor_path=tmp_path / "runtime.json", outbox_path=tmp_path / "outbox.json"
    )
    try:
        assert client.enqueue_capture(payload())["durable_queued"]
        result = client.enqueue_capture({**payload(), "content": "Different content."})
        assert result["status"] == "blocked" and not result["durable_queued"]
        assert Outbox(tmp_path / "outbox.json").pending()[0]["payload"] == payload()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_queue_receipt_obeys_selected_write_policy(tmp_path):
    path = tmp_path / "outbox.json"
    selected = Outbox(path, firewall=MemoryWriteFirewall(MemorySafetyConfig(pii_action="block")))
    client = MnemosClient(descriptor_path=tmp_path / "runtime.json", outbox=selected)
    try:
        result = client.enqueue_capture({**payload(), "content": "Contact test.person@example.com"})
        assert result["status"] == "blocked" and not result["durable_queued"]
        assert not path.exists()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_receipt_identity_matches_serialized_payload(tmp_path):
    client = MnemosClient(descriptor_path=tmp_path / "runtime.json")
    try:
        result = client.enqueue_capture({**payload(), "event_id": 42})
        assert result["event_id"] == client.outbox.pending()[0]["payload"]["event_id"] == "42"
    finally:
        await client.close()
