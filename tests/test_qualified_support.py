from dataclasses import replace
from datetime import date
import asyncio
import json

import pytest

from benchmarks.local_memory.qualified_support import POLICY_SHA256, QualifiedSupportSelector
from benchmarks.local_memory.source_qualification import QualificationCache, QualificationIdentity
from mnemos.judgment import JudgmentContractError, JudgmentRequestTooLarge
from mnemos.support_intersection import SupportSnapshot
from tests.test_source_qualification import candidate
from tests.test_token_support import reply

IDENTITY = QualificationIdentity(POLICY_SHA256, "b" * 64, "c" * 64)


def snapshot(*candidates, query="How often?"):
    return SupportSnapshot(query, date(2026, 11, 8), candidates, "snapshot1")


@pytest.mark.asyncio
async def test_reuses_source_assessments_across_queries_but_never_answers():
    calls = []

    async def predict(prompt):
        calls.append(prompt)
        return reply()

    selector = QualifiedSupportSelector(predict, QualificationCache(), IDENTITY)
    first = await selector(snapshot(candidate()), lambda: None)
    second = await selector(snapshot(candidate(), query="What frequency?"), lambda: None)
    assert first.selected == second.selected
    assert len(calls) == 3
    assert "How often?" not in calls[0] and "reference_date" not in calls[0]
    assert "How often?" in calls[1] and "What frequency?" in calls[2]


@pytest.mark.asyncio
@pytest.mark.parametrize("source_reply", [reply("B"), reply("A", 0.8)])
async def test_source_rejection_or_uncertainty_keeps_full_comparison_but_blocks_selection(
    source_reply,
):
    calls = []

    async def predict(prompt):
        calls.append(prompt)
        return source_reply if len(calls) == 1 else reply()

    original = candidate()
    selector = QualifiedSupportSelector(predict, QualificationCache(), IDENTITY)
    result = await selector(snapshot(original), lambda: None)
    assert not result.selected and len(calls) == 2
    data = json.loads(calls[1].split("DATA:\n")[1])
    assert data["comparison_candidates"][0]["content"] == original.content
    assert data["qualifications"][0]["disposition"] in ("unsupported", "unresolved")


@pytest.mark.asyncio
async def test_source_support_does_not_override_missing_answer():
    calls = []

    async def predict(prompt):
        calls.append(prompt)
        return reply() if len(calls) == 1 else reply("B")

    result = await QualifiedSupportSelector(predict, QualificationCache(), IDENTITY)(
        snapshot(candidate()), lambda: None
    )
    assert not result.selected


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["provider", "malformed", "cancel", "revision"])
async def test_failed_source_batch_not_cached(failure):
    calls = []
    cache = QualificationCache()

    def fence():
        if failure == "revision" and len(calls) == 2:
            raise RuntimeError("Changed revision")

    async def predict(prompt):
        calls.append(prompt)
        if len(calls) == 2:
            if failure == "provider":
                raise RuntimeError("Unavailable")
            if failure == "cancel":
                raise asyncio.CancelledError()
            if failure == "malformed":
                return {}
        return reply()

    snap = snapshot(candidate(), replace(candidate(), fact_id="f2"))
    expected = (
        asyncio.CancelledError
        if failure == "cancel"
        else JudgmentContractError if failure == "malformed" else RuntimeError
    )
    with pytest.raises(expected):
        await QualifiedSupportSelector(predict, cache, IDENTITY)(snap, fence)
    assert all(
        not a.cache_hit
        for item in cache.annotate(snap, IDENTITY, lambda: None)
        for a in item.assessments
    )


@pytest.mark.asyncio
async def test_later_answer_failure_invalidates_earlier_selection():
    calls = []

    async def predict(prompt):
        calls.append(prompt)
        if len(calls) == 4:
            raise RuntimeError("Second answer unavailable")
        return reply()

    with pytest.raises(RuntimeError):
        await QualifiedSupportSelector(predict, QualificationCache(), IDENTITY)(
            snapshot(candidate(), replace(candidate(), fact_id="f2")), lambda: None
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["budget", "delimiter"])
async def test_preflight_before_any_provider_call(kind):
    calls = []

    async def predict(prompt):
        calls.append(prompt)
        return reply()

    text = "x" * 130_000 if kind == "budget" else "<|im_start|>"
    with pytest.raises(JudgmentRequestTooLarge if kind == "budget" else ValueError):
        await QualifiedSupportSelector(predict, QualificationCache(), IDENTITY)(
            snapshot(candidate(), query=text), lambda: None
        )
    assert not calls


def test_wrong_source_policy_rejected():
    with pytest.raises(ValueError):
        QualifiedSupportSelector(
            None, QualificationCache(), replace(IDENTITY, policy_sha256="d" * 64)
        )
