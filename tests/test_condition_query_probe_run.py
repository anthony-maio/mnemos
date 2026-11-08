import asyncio
import json

import pytest

from benchmarks.local_memory.evaluation_journal import EvaluationJournal, read_journal
from benchmarks.local_memory.condition_query_probe import build_condition_cases
from benchmarks.local_memory.condition_query_probe_run import run_condition_probe


def cases():
    return build_condition_cases("runner-test-v1")[:2]


@pytest.mark.asyncio
async def test_complete_malformed_reply_is_retained_without_retry(tmp_path):
    cohort = cases()
    calls = []

    async def provider(prompt, schema):
        calls.append(prompt)
        return dict(done=True, response="{}" if len(calls) == 1 else json.dumps(cohort[1].expected))

    path = tmp_path / "events.jsonl"
    with EvaluationJournal(path) as journal:
        rows = await run_condition_probe(
            [(c.key, c.request) for c in cohort], provider, journal, lambda: None
        )
    records = read_journal(path)
    assert len(calls) == 2
    assert [r["protocol_valid"] for r in rows] == [False, True]
    assert records[-1]["kind"] == "run_complete"
    starts = [r["data"] for r in records if r["kind"] == "request_start"]
    assert all(set(r) == {"key", "owner_request", "prompt", "schema"} for r in starts)
    assert [r["data"]["wire"]["response"] for r in records if r["kind"] == "provider_reply"] == [
        "{}",
        json.dumps(cohort[1].expected),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["exception", "incomplete", "cancel", "before-log", "reply-log", "observation-log", "revoked"],
)
async def test_uncertain_or_unrecordable_requests_never_advance(failure):
    cohort = cases()
    calls, events = [], []
    revoked = False

    async def provider(prompt, schema):
        nonlocal revoked
        calls.append(prompt)
        if failure == "exception":
            raise TimeoutError("uncertain")
        if failure == "cancel":
            raise asyncio.CancelledError
        if failure == "revoked":
            revoked = True
        return dict(done=failure != "incomplete", response=json.dumps(cohort[0].expected))

    def fence():
        if revoked:
            raise PermissionError("source revoked")

    def emit(kind, data):
        fail_kind = {
            "before-log": "request_start",
            "reply-log": "provider_reply",
            "observation-log": "query_observation",
        }.get(failure)
        if kind == fail_kind:
            raise OSError("journal unavailable")
        events.append(kind)

    args = ([(c.key, c.request) for c in cohort], provider, emit, fence)
    if failure in {"exception", "incomplete"}:
        assert await run_condition_probe(*args) == []
        assert events[-1] == "run_stopped"
    else:
        expected = (
            asyncio.CancelledError
            if failure == "cancel"
            else PermissionError if failure == "revoked" else OSError
        )
        with pytest.raises(expected):
            await run_condition_probe(*args)
    assert len(calls) == (0 if failure == "before-log" else 1)
    assert "run_complete" not in events


@pytest.mark.asyncio
async def test_invalid_schedule_fails_before_logging_or_calling():
    item = cases()[0]

    def forbidden(*args):
        pytest.fail("Invalid schedule must not start")

    with pytest.raises(ValueError):
        await run_condition_probe([(item.key, item.request)] * 2, forbidden, forbidden, forbidden)


@pytest.mark.asyncio
async def test_owner_cancellation_stops_pending_call_and_logs_it(tmp_path):
    cohort = cases()
    entered = asyncio.Event()
    provider_cancelled = asyncio.Event()
    calls = []

    async def provider(prompt, schema):
        calls.append(prompt)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            provider_cancelled.set()

    path = tmp_path / "cancelled.jsonl"
    with EvaluationJournal(path) as journal:
        task = asyncio.create_task(
            run_condition_probe(
                [(c.key, c.request) for c in cohort], provider, journal, lambda: None
            )
        )
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert provider_cancelled.is_set() and len(calls) == 1
    records = read_journal(path)
    assert records[-1]["kind"] == "request_cancelled"
    assert not any(r["kind"] in {"provider_reply", "run_complete"} for r in records)
