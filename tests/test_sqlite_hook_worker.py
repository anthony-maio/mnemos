import asyncio
import time

import pytest

from mnemos.client import Outbox
from mnemos.hook_worker import HookDispatcher, HookWorker
from mnemos.sqlite_outbox import SQLiteOutbox
from mnemos.utils.storage import SQLiteStore
from tests.test_hook_worker import configured, payload, send  # noqa: F401
from tests.test_queued_capture_real_service import memory_owner, wait_empty


@pytest.mark.parametrize("backend,queued", [("invalid", True), ("sqlite", False)])
def test_invalid_backend_configuration_is_rejected(configured, backend, queued):
    with pytest.raises(ValueError):
        HookWorker(
            "codex",
            descriptor_path=configured / "worker.json",
            cwd=configured,
            queue_captures=queued,
            outbox_backend=backend,
        )
    assert not (configured / "outbox.sqlite").exists()


def test_sqlite_worker_rejects_existing_json_owner(configured):
    root = configured
    original = HookWorker(
        "codex", descriptor_path=root / "worker.json", cwd=root, queue_captures=True
    )
    original.start()
    other = HookWorker(
        "codex",
        descriptor_path=root / "worker.json",
        cwd=root,
        queue_captures=True,
        outbox_backend="sqlite",
    )
    try:
        with pytest.raises(RuntimeError, match="not this hook worker"):
            other.start()
        assert send(root, payload(root))["capture_state"] == "queued"
    finally:
        other.stop()
        original.stop()


def test_shutdown_closes_both_pumps_and_owned_connection(configured):
    async def check():
        dispatcher = HookDispatcher(
            "codex", cwd=configured, queue_captures=True, outbox_backend="sqlite"
        )
        dispatcher.start()
        assert isinstance(dispatcher.client.outbox, SQLiteOutbox)
        dispatcher.client.outbox.pending()
        connection = dispatcher.client.outbox._connection
        assert connection is not None
        await dispatcher.close()
        assert dispatcher.client.outbox._connection is None
        assert dispatcher.replay._task.done() and dispatcher.compatibility_replay._task.done()
        await dispatcher.close()

    asyncio.run(check())


def test_json_backlog_added_after_start_is_drained_by_timer(configured):
    root = configured
    # Obtain a valid original payload through the ordinary queued worker.
    first = HookWorker("codex", descriptor_path=root / "worker.json", cwd=root, queue_captures=True)
    first.start()
    try:
        send(root, payload(root))
        entry = Outbox(root / "outbox.json").pending()[0]
    finally:
        first.stop()
    Outbox(root / "outbox.json").acknowledge(entry["event_id"])
    memory = memory_owner(root)
    memory.start()
    worker = HookWorker(
        "codex",
        descriptor_path=root / "worker.json",
        cwd=root,
        queue_captures=True,
        outbox_backend="sqlite",
    )
    descriptor = worker.start()
    try:
        assert descriptor["outbox_backend"] == "sqlite"
        # Wait for the initial legacy pump pass before adding fallback backlog.
        deadline = time.monotonic() + 3
        while (
            worker._dispatcher.compatibility_replay.last_result is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert worker._dispatcher.compatibility_replay.last_result is not None
        Outbox(root / "outbox.json").append(entry["payload"])
        wait_empty(root, "sqlite")
        reader = SQLiteStore(str(root / "memory.sqlite"))
        try:
            assert len(reader.list_episodic_events()) == len(reader.list_jobs()) == 1
        finally:
            reader.close()
    finally:
        worker.stop()
        memory.stop()


def test_rollback_to_json_replays_remaining_sqlite_queue(configured):
    root = configured
    worker = HookWorker(
        "codex",
        descriptor_path=root / "worker.json",
        cwd=root,
        queue_captures=True,
        outbox_backend="sqlite",
    )
    worker.start()
    try:
        assert send(root, payload(root))["capture_state"] == "queued"
    finally:
        worker.stop()
    queue = SQLiteOutbox(root / "outbox.sqlite")
    try:
        original = queue.pending()[0]["payload"]
    finally:
        queue.close()
    # Duplicate fallback backlog must not create a second service event/job.
    Outbox(root / "outbox.json").append(original)
    memory = memory_owner(root)
    memory.start()
    restored = HookWorker(
        "codex", descriptor_path=root / "worker.json", cwd=root, queue_captures=True
    )
    descriptor = restored.start()
    try:
        assert descriptor["outbox_backend"] == "json"
        wait_empty(root, "sqlite")
        reader = SQLiteStore(str(root / "memory.sqlite"))
        try:
            assert len(reader.list_episodic_events()) == len(reader.list_jobs()) == 1
        finally:
            reader.close()
    finally:
        restored.stop()
        memory.stop()
