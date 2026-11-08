import asyncio
import threading
import time

import pytest

from mnemos.client import Outbox
from mnemos.hook_worker import HookWorker
from mnemos.service import MnemosService
from tests.test_hook_worker import configured, payload, send, MemoryDispatcher  # noqa: F401


def test_queued_stop_returns_before_delivery_and_restart_drains_without_new_hook(configured):
    root = configured
    entered = threading.Event()

    class Stalled:
        async def dispatch(self, operation, value):
            entered.set()
            await asyncio.Event().wait()

    memory = MnemosService(dispatcher=Stalled(), descriptor_path=root / "memory-runtime.json")
    memory.start()
    worker = HookWorker(
        "codex", descriptor_path=root / "worker.json", cwd=root, queue_captures=True
    )
    worker.start()
    try:
        result = send(root, payload(root))
        assert result["status"] == "handled"
        assert result["capture_state"] == "queued" and result["queued"] == 1
        assert result["output"] == {"continue": True}
        assert "job_id" not in result
        pending = Outbox(root / "outbox.json").pending()
        assert result["event_ids"] == [pending[0]["payload"]["event_id"]]
        assert entered.wait(2)
    finally:
        worker.stop()
        memory.stop()
    assert len(Outbox(root / "outbox.json")) == 1
    recorder = MemoryDispatcher(root)
    memory = MnemosService(dispatcher=recorder, descriptor_path=root / "memory-runtime.json")
    memory.start()
    restarted = HookWorker(
        "codex", descriptor_path=root / "worker.json", cwd=root, queue_captures=True
    )
    restarted.start()
    try:
        # No new hook request is sent: owner startup must drain leftover work.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and Outbox(root / "outbox.json").pending():
            time.sleep(0.01)
        assert not Outbox(root / "outbox.json").pending()
        assert [r["event_id"] for r in recorder.captures] == result["event_ids"]
    finally:
        restarted.stop()
        memory.stop()


def test_queue_mode_cannot_silently_reuse_synchronous_worker(configured):
    root = configured
    worker = HookWorker("codex", descriptor_path=root / "worker.json", cwd=root)
    worker.start()
    try:
        queued = HookWorker(
            "codex", descriptor_path=root / "worker.json", cwd=root, queue_captures=True
        )
        with pytest.raises(RuntimeError, match="not this hook worker"):
            queued.start()
        assert send(root, payload(root))["status"] == "handled"
    finally:
        worker.stop()


def test_queue_mode_rejects_unimplemented_host(configured):
    with pytest.raises(ValueError, match="only for Codex"):
        HookWorker(
            "claude-code",
            descriptor_path=configured / "worker.json",
            cwd=configured,
            queue_captures=True,
        )
