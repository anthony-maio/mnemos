"""An explicitly supplied capture outbox retains its location and write policy."""

from pathlib import Path

import pytest

from mnemos.client import MnemosClient, Outbox
from mnemos.config import MemorySafetyConfig
from mnemos.memory_safety import MemoryWriteFirewall
from mnemos.service import MnemosService


@pytest.mark.asyncio
@pytest.mark.parametrize("seed_pending", [False, True])
async def test_supplied_outbox_keeps_capture_available_at_selected_path_after_restart(
    tmp_path: Path, seed_pending: bool
) -> None:
    descriptor = tmp_path / "runtime.json"
    selected_path = tmp_path / "selected" / "capture.json"
    outbox = Outbox(selected_path)
    expected_ids = []
    if seed_pending:
        outbox.append({"event_id": "earlier", "content": "Keep the earlier event."})
        expected_ids.append("earlier")
    client = MnemosClient(descriptor_path=descriptor, outbox=outbox, auto_start=False)
    try:
        queued = await client.capture(
            {"event_id": "selected-capture", "content": "Keep the selected recovery path."}
        )
        assert queued["status"] == "queued"
        expected_ids.append("selected-capture")
        assert [
            entry["payload"]["event_id"] for entry in Outbox(selected_path).pending()
        ] == expected_ids
        assert not descriptor.with_name("outbox.json").exists()
    finally:
        await client.close()

    calls = []

    class Dispatcher:
        async def dispatch(self, operation, payload):
            calls.append((operation, payload))
            return {"schema_version": 1, "status": "committed", "committed": True}

    service = MnemosService(
        dispatcher=Dispatcher(), descriptor_path=descriptor, lock_path=tmp_path / "runtime.lock"
    )
    service.start()
    restarted = MnemosClient(
        descriptor_path=descriptor, outbox=Outbox(selected_path), auto_start=False
    )
    try:
        replay = await restarted.replay_outbox()
        assert replay == {
            "schema_version": 1,
            "acknowledged": len(expected_ids),
            "failed": 0,
            "pending": 0,
        }
        assert [payload["event_id"] for _, payload in calls] == expected_ids
        assert {operation for operation, _ in calls} == {"capture"}
        assert await restarted.replay_outbox() == {
            "schema_version": 1,
            "acknowledged": 0,
            "failed": 0,
            "pending": 0,
        }
        assert len(calls) == len(expected_ids)
    finally:
        await restarted.close()
        service.stop()


@pytest.mark.asyncio
async def test_empty_supplied_outbox_keeps_its_stricter_write_policy(tmp_path: Path) -> None:
    selected = Outbox(
        tmp_path / "strict" / "capture.json",
        firewall=MemoryWriteFirewall(MemorySafetyConfig(pii_action="block")),
    )
    client = MnemosClient(
        descriptor_path=tmp_path / "runtime.json", outbox=selected, auto_start=False
    )
    try:
        result = await client.capture(
            {"event_id": "blocked-address", "content": "Contact person@example.com."}
        )
        assert result["status"] == "blocked"
        assert result["committed"] is False
        assert not selected.path.exists()
        assert not (tmp_path / "outbox.json").exists()
    finally:
        await client.close()
