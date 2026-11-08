"""Identity lookup must leave the Claude worker's event loop responsive."""

import asyncio
import threading

import pytest

from mnemos.adapters.claude_code import ClaudeCodeAdapter
from mnemos.scopes import ScopeContext, ScopeRegistry
from tests.test_codex_adapter import RecordingClient


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["recall", "start", "capture"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_pending_identity_does_not_block_or_dispatch_after_cancel(
    tmp_path, monkeypatch, path, cancel
):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    client = RecordingClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=ScopeRegistry(),
        user_profile_id="local",
        checkpoint_directory=tmp_path,
    )
    transcript_calls = []

    def delayed_context(payload):
        entered.set()
        release.wait(2)
        finished.set()
        return ScopeContext(user_profile_id="local", session_id="s", host="claude-code")

    async def transcript(payload, context):
        transcript_calls.append(context)
        return 0, set(), None

    monkeypatch.setattr(adapter, "_context", delayed_context)
    monkeypatch.setattr(adapter, "_capture_transcript", transcript)
    monkeypatch.setenv("MNEMOS_AUTO_RETRIEVE", "true")
    payload = {"session_id": "s"}
    call = (
        adapter.retrieve_context("SessionStart", payload)
        if path == "recall"
        else adapter.handle("PreCompact" if path == "capture" else "SessionStart", payload)
    )
    task = asyncio.create_task(call)
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert not task.done(), "Identity discovery blocked the event loop"
        assert not finished.is_set()
        assert client.retrieves == [] and transcript_calls == []
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
            assert await asyncio.to_thread(finished.wait, 3)
            await asyncio.sleep(0)
            assert client.retrieves == [] and client.captures == [] and transcript_calls == []
        else:
            release.set()
            await asyncio.wait_for(task, 3)
            assert len(transcript_calls if path == "capture" else client.retrieves) == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert await asyncio.to_thread(finished.wait, 3)
