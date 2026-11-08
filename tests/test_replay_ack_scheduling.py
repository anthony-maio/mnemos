import asyncio
import sqlite3
import threading

import pytest

from mnemos.client import MnemosClient
from mnemos.sqlite_outbox import SQLiteOutbox
from mnemos.outbox_replay import OutboxReplayPump


@pytest.mark.asyncio
async def test_replay_commit_does_not_block_next_loop_callback(tmp_path, monkeypatch):
    queue = SQLiteOutbox(tmp_path / "queue.sqlite")
    queue.append({"event_id": "original", "content": "Synthetic queued capture."})
    client = MnemosClient(outbox=queue, auto_start=False)

    async def committed(*args, **kwargs):
        return {"committed": True}

    monkeypatch.setattr(client, "request", committed)
    entered, release, next_request = (threading.Event() for _ in range(3))
    observed = []
    loop = asyncio.get_running_loop()

    def hold_commit(action, value, *unused):
        if action == sqlite3.SQLITE_TRANSACTION and value == "COMMIT":
            entered.set()
            if not release.wait(5):
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    queue._connection.set_authorizer(hold_commit)

    def observe_loop():
        if entered.wait(5):
            loop.call_soon_threadsafe(next_request.set)
            observed.append(next_request.wait(1))
        release.set()

    observer = threading.Thread(target=observe_loop, daemon=True)
    observer.start()
    try:
        result = await client.replay_outbox()
        assert observed == [True], "Replay commit prevented the next request from starting"
        assert result["acknowledged"] == 1 and result["pending"] == 0
    finally:
        release.set()
        observer.join(timeout=5)
        queue._connection.set_authorizer(None)
        await client.close()
        queue.close()
    reopened = SQLiteOutbox(queue.path)
    try:
        assert reopened.pending() == []
    finally:
        reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_commit", [False, True])
async def test_shutdown_joins_commit_after_repeated_cancellation(
    tmp_path, monkeypatch, fail_commit
):
    queue = SQLiteOutbox(tmp_path / "queue.sqlite")
    queue.append({"event_id": "original", "content": "Synthetic queued capture."})
    if fail_commit:
        queue._connection.set_authorizer(
            lambda action, value, *rest: (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_TRANSACTION and value == "COMMIT"
                else sqlite3.SQLITE_OK
            )
        )
    client = MnemosClient(outbox=queue, auto_start=False)
    entered, release, finished = (threading.Event() for _ in range(3))
    original_ack = queue.acknowledge_many
    close_observations = []
    original_close = client.close

    def held_ack(ids):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("Test release was not signaled")
        try:
            return original_ack(ids)
        finally:
            finished.set()

    async def committed(*args, **kwargs):
        return {"committed": True}

    async def checked_close():
        close_observations.append(finished.is_set())
        await original_close()

    monkeypatch.setattr(queue, "acknowledge_many", held_ack)
    monkeypatch.setattr(client, "request", committed)
    monkeypatch.setattr(client, "close", checked_close)
    pump = OutboxReplayPump(client)
    pump.start()
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        closing = asyncio.create_task(pump.close())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not closing.done() and not close_observations
        pump._task.cancel()
        await asyncio.sleep(0)
        assert not closing.done() and not close_observations
    finally:
        release.set()
        if closing is not None:
            await asyncio.wait_for(closing, 5)
        else:
            await pump.close()
        queue.close()
    assert finished.is_set() and close_observations == [True]
    reopened = SQLiteOutbox(queue.path)
    try:
        assert [row["payload"]["event_id"] for row in reopened.pending()] == (
            ["original"] if fail_commit else []
        )
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_failed_threaded_commit_propagates_and_retains_original_event(tmp_path, monkeypatch):
    queue = SQLiteOutbox(tmp_path / "queue.sqlite")
    queue.append({"event_id": "original", "content": "Synthetic queued capture."})
    client = MnemosClient(outbox=queue, auto_start=False)

    async def committed(*args, **kwargs):
        return {"committed": True}

    monkeypatch.setattr(client, "request", committed)
    queue._connection.set_authorizer(
        lambda action, value, *rest: (
            sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_TRANSACTION and value == "COMMIT"
            else sqlite3.SQLITE_OK
        )
    )
    try:
        with pytest.raises(ValueError, match="not acknowledged"):
            await client.replay_outbox()
    finally:
        await client.close()
        queue.close()
    reopened = SQLiteOutbox(queue.path)
    try:
        assert [row["payload"]["event_id"] for row in reopened.pending()] == ["original"]
    finally:
        reopened.close()
