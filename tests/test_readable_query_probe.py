import asyncio
from collections import Counter
from dataclasses import replace
import json

import pytest

from benchmarks.local_memory.readable_query import bind_readable_transport, interpret_readable
from benchmarks.local_memory.readable_query_probe import (
    build_readable_cases,
    prepare_readable_probe,
)
from benchmarks.local_memory.packed_query_probe import build_cases
from tests.test_relation_claim_references import prepared
from tests.test_relation_inventory_cache import IDENTITY
from tests.test_readable_query_wire import body


def test_authored_probe_has_separate_labels_and_order_pairs():
    cases = build_readable_cases(IDENTITY, IDENTITY)
    execution, grading = prepare_readable_probe(cases)
    assert len(execution["requests"]) == len(grading["labels"]) == 22
    assert Counter(c.group for c in cases) == {"semantic": 18, "control": 4}
    assert all(n == 2 for n in Counter(c.pair for c in cases).values())
    old_queries = {c.request.snapshot.query for c in build_cases(IDENTITY, IDENTITY)}
    assert all(c.request.snapshot.query not in old_queries for c in cases)
    for row, label in zip(execution["requests"], grading["labels"], strict=True):
        assert set(row) == {"key", "owner_request", "prompt", "schema"}
        assert row["key"] == label["key"]
        assert "expected" not in json.loads(row["owner_request"]["model_input"])
    for pair in {c.pair for c in cases}:
        first, second = [c for c in cases if c.pair == pair]
        assert first.request.snapshot.query == second.request.snapshot.query
        assert first.request.snapshot.candidates == tuple(
            reversed(second.request.snapshot.candidates)
        )

        def by_fact(case):
            return {
                ref.fact_id: case.expected["judgments"][alias]
                for alias, ref in case.request.aliases
            }

        assert by_fact(first) == by_fact(second)


def test_wrong_authored_outcome_and_duplicate_keys_fail_preparation():
    cases = build_readable_cases(IDENTITY, IDENTITY)
    with pytest.raises(ValueError):
        prepare_readable_probe((cases[0], cases[0]))
    with pytest.raises(ValueError):
        prepare_readable_probe((replace(cases[0], status="wrong"),))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "incomplete", "malformed", "revoked", "cancelled"])
async def test_transport_binds_once_and_fails_without_retry(failure):
    snapshot, cache = await prepared()
    calls, revoked = [], False
    raw = json.dumps(body(("MA", "XA")))

    async def provider(prompt, schema):
        nonlocal revoked
        calls.append(prompt)
        assert schema["type"] == "object"
        if failure == "cancelled":
            raise asyncio.CancelledError
        if failure == "revoked":
            revoked = True
        return dict(done=failure != "incomplete", response="{}" if failure == "malformed" else raw)

    def fence():
        if revoked:
            raise PermissionError("source revoked")

    call = interpret_readable(
        snapshot, IDENTITY, IDENTITY, cache, bind_readable_transport(provider), fence
    )
    if failure == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await call
    else:
        result = await call
        assert result.decision.status == ("selected" if failure is None else "unavailable")
        if failure is None:
            assert result.raw_response == raw
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_missing_annotation_prevents_provider_call():
    snapshot, cache = await prepared()
    snapshot = replace(
        snapshot, candidates=(replace(snapshot.candidates[0], version=2), snapshot.candidates[1])
    )

    async def provider(prompt, schema):
        pytest.fail("Missing competitor must block interpretation")

    result = await interpret_readable(
        snapshot, IDENTITY, IDENTITY, cache, bind_readable_transport(provider), lambda: None
    )
    assert result.decision.status == "unavailable"
