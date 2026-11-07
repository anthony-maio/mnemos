"""Contrast decisions retain source binding and whole-comparison invalidation."""

from datetime import date

import pytest

from mnemos.answer_support import AnswerSupportConfig, assess_answer_support
from mnemos.judgment import ChoiceAnswer, MockJudge
from mnemos.utils.reliability import MnemosTransientError
from tests.test_answer_support import fact, memory  # noqa: F401


async def assess(memory, judge, ids=("one",), *, reference_date=date(2026, 11, 6), **options):
    store, registry, context = memory
    return await assess_answer_support(
        "Which queue is current?",
        ids,
        store=store,
        registry=registry,
        context=context,
        judge=judge,
        reference_date=reference_date,
        config=AnswerSupportConfig(protocol="contrast_v5", batching="all", **options),
    )


@pytest.mark.asyncio
async def test_joint_comparison_preserves_own_sources_and_excludes_foreign_or_raw_facts(memory):
    fact(memory, "one", "The queue has not been approved.")
    fact(memory, "two", "The next queue is proposed but has not started.")
    fact(memory, "foreign", "Private foreign content", scope_id="foreign")
    fact(memory, "raw", "Unaccepted raw content", source="captured_passthrough")
    before = [memory[0].get(key).model_dump() for key in ("one", "two")]
    judge = MockJudge({"support:*": "evidence:0"})
    result = await assess(memory, judge, ("one", "foreign", "two", "raw"))
    assert result.status == "assessed"
    assert len(judge.calls) == 1
    state, questions, purpose = judge.calls[0]
    assert purpose == "answer_support_v5"
    assert set(questions) == {"support:0", "support:1"}
    assert [row["fact"] for row in state["comparison_candidates"]] == [
        row["content"] for row in before
    ]
    assert [row.evidence_ids for row in result.decisions if row.disposition == "supported"] == [
        ("one-evidence",),
        ("two-evidence",),
    ]
    assert [memory[0].get(key).model_dump() for key in ("one", "two")] == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "choice,disposition",
    [
        ("different_request", "related"),
        ("time_unestablished", "unresolved"),
        ("unsupported_source", "unsupported"),
        ("conflict", "unresolved"),
    ],
)
async def test_each_rejection_prevents_supported_output(memory, choice, disposition):
    fact(memory, "one")
    result = await assess(memory, MockJudge({"support:*": choice}))
    assert result.status == "assessed" and not result.supported
    assert result.decisions[0].disposition == disposition
    assert result.decisions[0].reason == choice
    assert not result.decisions[0].evidence_ids


@pytest.mark.asyncio
async def test_low_probability_support_remains_unresolved(memory):
    fact(memory, "one")
    result = await assess(
        memory,
        MockJudge(
            {"support:*": ChoiceAnswer("evidence:0", {"evidence:0": 0.79, "conflict": 0.21}, 0.79)}
        ),
    )
    assert not result.supported and result.decisions[0].reason == "low_probability"


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", ["extra", "missing", "foreign_source"])
async def test_invalid_response_is_failure_and_never_abstention_credit(memory, malformed):
    fact(memory, "one")

    class Invalid(MockJudge):
        async def ask(self, *args, **kwargs):
            result = await super().ask(*args, **kwargs)
            if malformed == "extra":
                result.answers["support:99"] = result.answers["support:0"]
            elif malformed == "missing":
                result.answers.pop("support:0")
            else:
                result.answers["support:0"] = ChoiceAnswer("evidence:1", {"evidence:1": 1.0}, 1.0)
            return result

    result = await assess(memory, Invalid({"support:*": "evidence:0"}))
    assert result.status == "unavailable" and not result.supported
    assert result.error_type == "JudgmentContractError"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["correct", "forget", "source", "arrive", "scope"])
async def test_comparison_changes_invalidate_every_decision(memory, change):
    fact(memory, "one")
    if change != "arrive":
        fact(memory, "two")
    store = memory[0]

    class Changed(MockJudge):
        async def ask(self, *args, **kwargs):
            result = await super().ask(*args, **kwargs)
            if change == "arrive":
                fact(memory, "two")
            elif change == "forget":
                store.delete("two")
            elif change == "source":
                store._conn.execute(
                    "UPDATE fact_evidence SET excerpt='Only proposed' WHERE fact_id='two'"
                )
                store._conn.commit()
            else:
                chunk = store.get("two")
                if change == "scope":
                    chunk.metadata["scope_id"] = "foreign"
                else:
                    chunk = chunk.reconsolidate("The current queue was retracted.")
                store.store(chunk)
            return result

    result = await assess(memory, Changed({"support:*": "evidence:0"}), ("one", "two"))
    assert result.status == "partial" and not result.supported
    assert all(row.disposition == "stale" for row in result.decisions)


@pytest.mark.asyncio
async def test_missing_reference_date_is_rejected_before_provider(memory):
    fact(memory, "one")
    judge = MockJudge()
    with pytest.raises(ValueError, match="reference_date"):
        await assess(memory, judge, reference_date=None)
    assert not judge.calls


@pytest.mark.asyncio
async def test_request_limit_applies_before_provider(memory):
    fact(memory, "one")
    judge = MockJudge()
    result = await assess(memory, judge, max_request_bytes=10)
    assert result.status == "unavailable" and result.error_type == "JudgmentRequestTooLarge"
    assert not judge.calls


@pytest.mark.asyncio
async def test_provider_exception_keeps_failure_distinct_from_abstention(memory):
    fact(memory, "one")

    class Failed(MockJudge):
        async def ask(self, *args, **kwargs):
            raise MnemosTransientError("offline")

    result = await assess(memory, Failed())
    assert result.status == "unavailable" and result.error_type == "MnemosTransientError"
    assert not result.supported
