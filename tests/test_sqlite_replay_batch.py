import asyncio

import pytest

from mnemos.client import MnemosClient
from mnemos.sqlite_outbox import SQLiteOutbox


@pytest.mark.asyncio
async def test_replay_batches_successes_and_retains_failed_and_concurrent_events(tmp_path):
    queue = SQLiteOutbox(tmp_path / "queue.sqlite")
    client = MnemosClient(outbox=queue, auto_start=False)
    try:
        for i in range(35):
            queue.append({"content": str(i)}, event_id=str(i))
        calls = 0

        async def request(operation, payload, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                queue.append({"content": "arrived during replay"}, event_id="new")
            return {"status": "degraded" if payload["content"] == "17" else "committed"}

        client.request = request
        statements = []
        queue._connection.set_trace_callback(statements.append)
        result = await client.replay_outbox()
        assert calls == 35
        assert result == {"schema_version": 1, "acknowledged": 34, "failed": 1, "pending": 2}
        assert [r["event_id"] for r in queue.pending()] == ["17", "new"]
        # One durable concurrent append plus two bounded acknowledgment batches.
        assert statements.count("COMMIT") == 3
    finally:
        await client.close()
        queue.close()


@pytest.mark.asyncio
async def test_cancelled_replay_retains_unacknowledged_successes_with_original_ids(tmp_path):
    queue = SQLiteOutbox(tmp_path / "queue.sqlite")
    client = MnemosClient(outbox=queue, auto_start=False)
    try:
        for i in range(3):
            queue.append({"content": str(i)}, event_id=str(i))
        original = queue.pending()
        calls = 0

        async def request(operation, payload, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise asyncio.CancelledError
            return {"status": "committed"}

        client.request = request
        with pytest.raises(asyncio.CancelledError):
            await client.replay_outbox()
        queue.close()
        assert queue.pending() == original

        async def duplicate(operation, payload, **kwargs):
            return {"status": "committed", "duplicate": True}

        client.request = duplicate
        assert (await client.replay_outbox())["acknowledged"] == 3
        assert queue.pending() == []
    finally:
        await client.close()
        queue.close()
