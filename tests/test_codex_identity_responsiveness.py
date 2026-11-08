"""Slow project discovery must not stop the resident worker's event loop."""

import asyncio
import threading

import pytest

from mnemos.adapters.codex import CodexAdapter
from mnemos.scopes import ScopeContext, ScopeRegistry
from tests.test_codex_adapter import RecordingClient


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_pending_identity_allows_other_tasks_and_cancellation(monkeypatch, cancel):
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    client = RecordingClient()
    adapter = CodexAdapter(client=client, registry=ScopeRegistry(), user_profile_id="local")

    def delayed_context(payload):
        entered.set()
        # A bounded watchdog keeps the regression from hanging on the old code.
        release.wait(2)
        finished.set()
        return ScopeContext(user_profile_id="local", session_id="s", host="codex")

    monkeypatch.setattr(adapter, "_context", delayed_context)
    task = asyncio.create_task(adapter.handle("SessionStart", {"session_id": "s"}))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert not task.done(), "Git discovery blocked the event loop until it returned"
        assert not finished.is_set()
        assert client.retrieves == []
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
            assert await asyncio.to_thread(finished.wait, 3)
            await asyncio.sleep(0)
            assert client.retrieves == []
            assert client.captures == []
        else:
            release.set()
            await asyncio.wait_for(task, 3)
            assert len(client.retrieves) == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert await asyncio.to_thread(finished.wait, 3)
