"""Comparison evidence is scoped, explicit in time, and invalidated as one cohort."""

from copy import deepcopy
from datetime import date, datetime, timezone

import pytest

from mnemos.answer_support import AnswerSupportConfig, assess_answer_support
from mnemos.judgment import MockJudge
from tests.test_answer_support import fact, memory  # noqa: F401


async def evaluate(memory, judge, ids=("old", "current"), **kwargs):
    store, registry, context = memory
    return await assess_answer_support(
        "Which queue is current?",
        ids,
        store=store,
        registry=registry,
        context=context,
        judge=judge,
        config=AnswerSupportConfig(protocol="qualified_v3", batching="candidate"),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_comparison_contains_only_visible_accepted_source_backed_facts(memory):
    fact(memory, "old", "In 2024, the project used Redis.")
    fact(memory, "current", "PostgreSQL replaced Redis in October 2026.")
    fact(memory, "foreign", "Another project still uses Redis.", scope_id="foreign")
    fact(memory, "capture", "A model guessed that Redis returned.", source="captured_passthrough")
    judge = MockJudge({"support:*": "evidence:0"})
    result = await evaluate(
        memory,
        judge,
        ids=("old", "foreign", "current", "capture"),
        reference_date=date(2026, 11, 6),
    )
    assert result.status == "assessed"
    assert len(judge.calls) == 2
    expected = [
        {
            "candidate": 0,
            "fact": "In 2024, the project used Redis.",
            "source_excerpts": ["In 2024, the project used Redis."],
        },
        {
            "candidate": 1,
            "fact": "PostgreSQL replaced Redis in October 2026.",
            "source_excerpts": ["PostgreSQL replaced Redis in October 2026."],
        },
    ]
    for state, _, purpose in judge.calls:
        assert state == {
            "query": "Which queue is current?",
            "reference_date": "2026-11-06",
            "comparison_candidates": expected,
        }
        assert purpose == "answer_support_v3"
    # The mock establishes the evidence-binding contract, not temporal accuracy.
    assert [d.evidence_ids for d in result.decisions if d.disposition == "supported"] == [
        ("old-evidence",),
        ("current-evidence",),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["correct", "forget", "evidence", "scope"])
async def test_changed_comparison_evidence_invalidates_every_dependent_answer(memory, change):
    fact(memory, "old", "In 2024, the project used Redis.")
    fact(memory, "current", "PostgreSQL replaced Redis in October 2026.")
    store = memory[0]

    class ChangingJudge(MockJudge):
        async def ask(self, *args, **kwargs):
            result = await super().ask(*args, **kwargs)
            if change == "forget":
                store.delete("current")
            elif change == "evidence":
                store._conn.execute(
                    "UPDATE fact_evidence SET excerpt='A proposed replacement' WHERE fact_id='current'"
                )
                store._conn.commit()
            else:
                chunk = store.get("current")
                if change == "correct":
                    chunk = chunk.reconsolidate("The replacement was only a proposal.")
                else:
                    chunk.metadata["scope_id"] = "foreign"
                store.store(chunk)
            return result

    judge = ChangingJudge({"support:*": "evidence:0"})
    result = await evaluate(memory, judge, reference_date=date(2026, 11, 6))
    assert len(judge.calls) == 1
    assert result.status == "partial" and not result.supported
    assert all(
        d.disposition == "stale" and d.reason == "comparison_snapshot_changed"
        for d in result.decisions
    )


@pytest.mark.asyncio
async def test_qualified_protocol_requires_explicit_reference_date(memory):
    fact(memory, "old")
    judge = MockJudge()
    with pytest.raises(ValueError, match="reference_date"):
        await evaluate(memory, judge, ids=("old",))
    assert not judge.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["2026-11-06", datetime(2026, 11, 6, tzinfo=timezone.utc)])
async def test_reference_date_rejects_implicit_string_or_datetime_coercion(memory, invalid):
    fact(memory, "old")
    judge = MockJudge()
    with pytest.raises(ValueError, match="reference_date"):
        await evaluate(memory, judge, ids=("old",), reference_date=invalid)
    assert not judge.calls


@pytest.mark.asyncio
async def test_original_protocol_can_receive_same_reference_date_for_comparison(memory):
    fact(memory, "old")
    judge = MockJudge({"support:*": "evidence:0"})
    store, registry, context = memory
    result = await assess_answer_support(
        "Which queue is current?",
        ["old"],
        store=store,
        registry=registry,
        context=context,
        judge=judge,
        reference_date=date(2026, 11, 6),
    )
    assert result.status == "assessed"
    assert judge.calls[0][0] == {"query": "Which queue is current?", "reference_date": "2026-11-06"}
    assert judge.calls[0][2] == "answer_support_v1"


@pytest.mark.asyncio
async def test_candidate_calls_count_repeated_state_against_total_byte_limit(memory):
    for identifier in ("one", "two", "three"):
        fact(memory, identifier, "A source-backed fact.")
    store, registry, context = memory
    judge = MockJudge({"support:*": "evidence:0"})
    result = await assess_answer_support(
        "query " * 1000,
        ["one", "two", "three"],
        store=store,
        registry=registry,
        context=context,
        judge=judge,
        config=AnswerSupportConfig(batching="candidate", max_request_bytes=18000),
    )
    assert result.status == "unavailable"
    assert result.error_type == "JudgmentRequestTooLarge"
    assert not judge.calls


@pytest.mark.asyncio
async def test_newly_eligible_candidate_invalidates_the_comparison_view(memory):
    fact(memory, "old", "The project uses Redis.")

    class ArrivingFact(MockJudge):
        async def ask(self, *args, **kwargs):
            result = await super().ask(*args, **kwargs)
            fact(memory, "current", "PostgreSQL has replaced Redis.")
            return result

    result = await evaluate(
        memory, ArrivingFact({"support:*": "evidence:0"}), reference_date=date(2026, 11, 6)
    )
    assert result.status == "partial"
    assert not result.supported
    assert all(d.disposition == "stale" for d in result.decisions)


@pytest.mark.asyncio
async def test_backend_cannot_mutate_the_comparison_sent_to_later_calls(memory):
    fact(memory, "old", "The project used Redis in 2024.")
    fact(memory, "current", "PostgreSQL replaced Redis in October 2026.")
    seen = []

    class MutatingPayload(MockJudge):
        async def ask(self, state, *args, **kwargs):
            seen.append(deepcopy(state))
            result = await super().ask(state, *args, **kwargs)
            state["comparison_candidates"].clear()
            state["reference_date"] = "2099-01-01"
            return result

    result = await evaluate(
        memory, MutatingPayload({"support:*": "evidence:0"}), reference_date=date(2026, 11, 6)
    )
    assert result.status == "assessed"
    assert len(seen) == 2 and seen[0] == seen[1]


@pytest.mark.asyncio
async def test_batch_does_not_send_context_after_observing_a_changed_candidate(memory, monkeypatch):
    from mnemos import answer_support

    fact(memory, "old", "The project used Redis in 2024.")
    fact(memory, "current", "PostgreSQL replaced Redis in October 2026.")
    original = answer_support._snapshot
    calls = 0

    def interleaved_snapshot(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 4:
            # Change after the cohort check; the per-candidate batch check sees it.
            chunk = memory[0].get("old")
            memory[0].store(chunk.reconsolidate("The earlier record was retracted."))
        return result

    monkeypatch.setattr(answer_support, "_snapshot", interleaved_snapshot)
    judge = MockJudge({"support:*": "evidence:0"})
    store, registry, context = memory
    result = await assess_answer_support(
        "Which queue is current?",
        ["old", "current"],
        store=store,
        registry=registry,
        context=context,
        judge=judge,
        reference_date=date(2026, 11, 6),
        config=AnswerSupportConfig(protocol="qualified_v3", batching="all"),
    )
    assert not judge.calls
    assert result.status == "partial" and not result.supported
