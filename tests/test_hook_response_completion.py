from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from mnemos.hook_worker import HookWorker
from mnemos.hook_client import HookOutcomeUnknown
from mnemos.outbox_replay import OutboxReplayPump
from mnemos.service import MnemosService, _RequestHandler
from tests.test_hook_worker import configured, payload, send, MemoryDispatcher  # noqa: F401


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate()


@pytest.mark.parametrize("broken_write", [False, True])
def test_http_notifies_only_after_write_attempt_even_if_write_fails(
    configured, monkeypatch, broken_write
):
    root = configured
    recorder = MemoryDispatcher(root)
    memory = MnemosService(dispatcher=recorder, descriptor_path=root / "memory-runtime.json")
    worker = HookWorker(
        "codex", descriptor_path=root / "worker.json", cwd=root, queue_captures=True
    )
    entered, release, notified = threading.Event(), threading.Event(), threading.Event()
    original_write = _RequestHandler._write_json
    original_notify = OutboxReplayPump.notify

    def observe_notify(pump):
        notified.set()
        return original_notify(pump)

    def gated_write(handler, status, result):
        if handler.server.service is worker and result.get("capture_state") == "queued":
            entered.set()
            assert release.wait(3)
            if broken_write:
                raise BrokenPipeError("synthetic response failure")
        return original_write(handler, status, result)

    monkeypatch.setattr(OutboxReplayPump, "notify", observe_notify)
    monkeypatch.setattr(_RequestHandler, "_write_json", gated_write)
    memory.start()
    worker.start()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(send, root, payload(root))
            try:
                assert entered.wait(3)
                assert not notified.is_set()
            finally:
                release.set()
            if broken_write:
                with pytest.raises(HookOutcomeUnknown):
                    future.result(timeout=5)
            else:
                result = future.result(timeout=5)
                assert result["capture_state"] == "queued"
        assert notified.wait(3)
        wait_for(lambda: len(recorder.captures) == 1)
    finally:
        release.set()
        worker.stop()
        memory.stop()
    # A late completion racing shutdown must not resurrect an owner or raise.
    worker.response_finished("hook", {"capture_state": "queued"})


def test_direct_submit_still_notifies_replay(configured):
    root = configured
    recorder = MemoryDispatcher(root)
    memory = MnemosService(dispatcher=recorder, descriptor_path=root / "memory-runtime.json")
    worker = HookWorker(
        "codex", descriptor_path=root / "worker.json", cwd=root, queue_captures=True
    )
    memory.start()
    worker.start()
    try:
        result = worker.submit(
            "hook",
            {
                "hook_protocol": 1,
                "request_id": "direct",
                "host": "codex",
                "environment_sha256": worker.environment_sha256,
                "event": "Stop",
                "payload": payload(root),
                "cwd": str(root),
            },
        )
        assert result["capture_state"] == "queued"
        wait_for(lambda: len(recorder.captures) == 1)
    finally:
        worker.stop()
        memory.stop()


def test_timer_recovers_when_response_notification_is_lost(configured, monkeypatch):
    root = configured
    original_init = OutboxReplayPump.__init__

    def short_retry(pump, client, **kwargs):
        original_init(pump, client, retry_seconds=0.05)

    monkeypatch.setattr(OutboxReplayPump, "__init__", short_retry)
    recorder = MemoryDispatcher(root)
    memory = MnemosService(dispatcher=recorder, descriptor_path=root / "memory-runtime.json")
    worker = HookWorker(
        "codex", descriptor_path=root / "worker.json", cwd=root, queue_captures=True
    )
    monkeypatch.setattr(worker, "response_finished", lambda *args: None)
    memory.start()
    worker.start()
    try:
        assert send(root, payload(root))["capture_state"] == "queued"
        wait_for(lambda: len(recorder.captures) == 1)
    finally:
        worker.stop()
        memory.stop()
