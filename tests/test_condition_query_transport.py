import asyncio
from dataclasses import replace
import json

import pytest

from benchmarks.local_memory.condition_query import bind_condition_transport, interpret_conditions
from benchmarks.local_memory.condition_query_probe import build_condition_cases


def case():
    return build_condition_cases("test-transport-v1")[0]


@pytest.mark.asyncio
async def test_complete_reply_keeps_witnesses_and_raw_text():
    item = case()
    raw = json.dumps(item.expected)
    calls = []

    async def provider(prompt, schema):
        calls.append((prompt, schema))
        return dict(done=True, response=raw)

    result = await interpret_conditions(
        item.request, bind_condition_transport(provider), lambda: None
    )
    assert result.status == "interpreted"
    assert result.applicability == "applies"
    assert result.raw_response == raw
    assert len(result.circumstances.items) == 2
    assert len(calls) == 1
    assert item.request.source_text not in calls[0][0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wire", [None, {}, {"done": 1, "response": "{}"}, {"done": True, "response": 1}]
)
async def test_unconfirmed_provider_never_becomes_unknown_or_retries(wire):
    calls = []

    async def provider(*args):
        calls.append(args)
        return wire

    result = await interpret_conditions(
        case().request, bind_condition_transport(provider), lambda: None
    )
    assert result.status == "unavailable"
    assert result.applicability is None and result.circumstances is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_malformed_completed_reply_retained_for_diagnosis():
    async def provider(*args):
        return dict(done=True, response="bad JSON")

    result = await interpret_conditions(
        case().request, bind_condition_transport(provider), lambda: None
    )
    assert result.status == "unavailable" and result.raw_response == "bad JSON"
    assert result.applicability is None


@pytest.mark.asyncio
async def test_changed_request_rejected_before_provider():
    async def provider(*args):
        pytest.fail("Changed request reached provider")

    req = replace(case().request, query="changed")
    result = await interpret_conditions(req, bind_condition_transport(provider), lambda: None)
    assert result.status == "unavailable" and result.error == "ValueError"


@pytest.mark.asyncio
async def test_revision_change_during_provider_cannot_authorize_applicability():
    item = case()
    current = True

    def fence():
        if not current:
            raise RuntimeError("Revision changed")

    async def provider(*args):
        nonlocal current
        current = False
        return dict(done=True, response=json.dumps(item.expected))

    result = await interpret_conditions(item.request, bind_condition_transport(provider), fence)
    assert result.status == "unavailable" and result.error == "RuntimeError"
    assert result.applicability is None
    assert result.raw_response == json.dumps(item.expected)


@pytest.mark.asyncio
async def test_owner_cancellation_propagates_and_stops_provider():
    started, stopped = asyncio.Event(), asyncio.Event()

    async def provider(*args):
        started.set()
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    task = asyncio.create_task(
        interpret_conditions(case().request, bind_condition_transport(provider), lambda: None)
    )
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_deadline_cancels_single_attempt_without_applicability(monkeypatch):
    from benchmarks.local_memory import condition_query

    real_wait_for = asyncio.wait_for
    deadlines, calls = [], []
    stopped = asyncio.Event()

    async def accelerated_wait(awaitable, timeout):
        deadlines.append(timeout)
        return await real_wait_for(awaitable, timeout=0.01)

    async def transport(request):
        calls.append(request.request_id)
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    monkeypatch.setattr(condition_query.asyncio, "wait_for", accelerated_wait)
    result = await interpret_conditions(case().request, transport, lambda: None)
    assert deadlines == [15] and len(calls) == 1 and stopped.is_set()
    assert result.status == "unavailable" and result.error == "TimeoutError"
    assert result.applicability is None


@pytest.mark.asyncio
async def test_reply_from_another_request_cannot_be_reused():
    from benchmarks.local_memory.compact_query_wire import TransportReply

    foreign = case()

    async def transport(request):
        return TransportReply(
            foreign.request.request_id,
            foreign.request.binding_sha256,
            True,
            json.dumps(foreign.expected),
        )

    result = await interpret_conditions(case().request, transport, lambda: None)
    assert result.status == "unavailable" and result.applicability is None
