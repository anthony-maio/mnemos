"""The composition adapters preserve the previously measured component inputs."""

from dataclasses import replace
from datetime import date

import pytest

from benchmarks.local_memory.support_intersection_selectors import (
    ContrastSourceSelector,
    ExtractiveSourceSelector,
    QAOutput,
)
from mnemos.answer_support import AnswerSupportConfig, assess_answer_support
from mnemos.judgment import MockJudge
from mnemos.support_intersection import assess_support_intersection
from tests.test_answer_support import fact, memory  # noqa: F401


def model_output(query, context, *, positive=True):
    lo = context.index("Retained source excerpt:\n") + len("Retained source excerpt:\n")
    return QAOutput(
        input_ids=(1, 2, 3, 4, 5),
        offsets=((0, 0), (0, len(query)), (0, 0), (lo - 1, len(context)), (0, 0)),
        sequence_ids=(None, 0, None, 1, None),
        attention_mask=(1, 1, 1, 1, 1),
        # Larger scores in the question and special tokens must be ignored.
        start_logits=(0.0, 50.0, 50.0, 2.0 if positive else 0.0, 50.0),
        end_logits=(0.0, 50.0, 50.0, 2.0 if positive else 0.0, 50.0),
        cls_index=0,
        question_token_count=1,
    )


async def run(memory, predict, *, judge=None, observe=None):
    store, registry, context = memory
    judge = judge or MockJudge({"support:*": "evidence:0"})
    return await assess_support_intersection(
        "Which release check applies?",
        ("one", "two"),
        store=store,
        registry=registry,
        context=context,
        reference_date=date(2026, 11, 7),
        extractive=ExtractiveSourceSelector(predict, observe),
        contrast=ContrastSourceSelector(
            store=store, registry=registry, context=context, judge=judge
        ),
    )


@pytest.mark.asyncio
async def test_source_span_and_contrast_full_cohort_inputs_match_original_policy(memory):
    fact(memory, "one", "The release checklist is pending approval.")
    fact(memory, "two", "The proposed audit checklist starts in December.")
    records, requests = [], []

    async def predict(query, context):
        requests.append((query, context))
        return model_output(query, context, positive="pending approval" in context)

    judge = MockJudge({"support:*": "evidence:0"})
    result = await run(memory, predict, judge=judge, observe=records.append)
    assert result.status == "assessed"
    assert [item.fact_id for item in result.supported] == ["one"]
    assert len(requests) == len(records) == 2
    assert all(
        c.startswith("Reference date: 2026-11-07\nRetained source excerpt:\n") for _, c in requests
    )
    assert records[0]["decision"]["best_span"]["text"] == memory[0].get("one").content
    assert result.components[0].selected[0].evidence_ids == ("one-evidence",)
    assert len(judge.calls[0][0]["comparison_candidates"]) == 2
    original = MockJudge({"support:*": "evidence:0"})
    await assess_answer_support(
        "Which release check applies?",
        ("one", "two"),
        store=memory[0],
        registry=memory[1],
        context=memory[2],
        judge=original,
        reference_date=date(2026, 11, 7),
        config=AnswerSupportConfig(protocol="contrast_v5", batching="all"),
    )
    assert judge.calls == original.calls


@pytest.mark.asyncio
async def test_source_span_absent_from_fact_cannot_select_the_claim(memory):
    fact(memory, "one", "The release was approved.")
    fact(memory, "two", assertion_status="unknown")
    memory[0]._conn.execute(
        "UPDATE fact_evidence SET excerpt='The release is only proposed.' WHERE fact_id='one'"
    )
    memory[0]._conn.commit()

    async def predict(query, context):
        return model_output(query, context)

    result = await run(memory, predict)
    assert result.status == "assessed" and not result.supported
    assert result.components[0].selected == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed", ["nan", "length", "question_limit", "sequence_limit", "token_type"]
)
async def test_malformed_qa_output_cannot_hide_behind_successful_contrast(memory, malformed):
    fact(memory, "one")
    fact(memory, "two")
    records = []
    judge = MockJudge({"support:*": "evidence:0"})

    async def predict(query, context):
        value = model_output(query, context)
        if malformed == "nan":
            return replace(value, start_logits=(float("nan"), *value.start_logits[1:]))
        if malformed == "length":
            return replace(value, input_ids=value.input_ids[:-1])
        if malformed == "question_limit":
            return replace(value, question_token_count=65)
        if malformed == "sequence_limit":
            return replace(value, input_ids=(1,) * 513)
        return replace(value, input_ids=(True, *value.input_ids[1:]))

    result = await run(memory, predict, judge=judge, observe=records.append)
    assert result.status == "unavailable" and not result.supported
    assert result.error_type == "JudgmentContractError" and not judge.calls
    assert records[0]["status"] == "unavailable" and "raw_output" in records[0]


@pytest.mark.asyncio
async def test_failure_on_later_excerpt_invalidates_earlier_qa_selection(memory):
    fact(memory, "one", "Release checklist one.")
    fact(memory, "two", "Release checklist two.")

    async def predict(query, context):
        value = model_output(query, context)
        return replace(value, cls_index=100) if context.endswith("two.") else value

    result = await run(memory, predict)
    assert result.status == "unavailable" and not result.supported


@pytest.mark.asyncio
async def test_contrast_source_change_invalidates_the_completed_qa_decision(memory):
    fact(memory, "one")
    fact(memory, "two")

    async def predict(query, context):
        return model_output(query, context)

    class ChangeEvidence(MockJudge):
        async def ask(self, *args, **kwargs):
            result = await super().ask(*args, **kwargs)
            memory[0]._conn.execute(
                "UPDATE fact_evidence SET excerpt='Retracted' WHERE fact_id='two'"
            )
            memory[0]._conn.commit()
            return result

    result = await run(memory, predict, judge=ChangeEvidence({"support:*": "evidence:0"}))
    assert result.status == "partial" and not result.supported
    assert all(item.disposition == "stale" for item in result.decisions)


@pytest.mark.asyncio
async def test_qa_rechecks_cohort_before_calling_model_on_the_next_source(memory):
    fact(memory, "one")
    fact(memory, "two")
    calls = []
    judge = MockJudge({"support:*": "evidence:0"})

    async def predict(query, context):
        calls.append(context)
        memory[0].delete("two")
        return model_output(query, context)

    result = await run(memory, predict, judge=judge)
    assert len(calls) == 1 and not judge.calls
    assert result.status == "partial" and not result.supported
    assert all(item.disposition == "stale" for item in result.decisions)
