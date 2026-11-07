"""Every component and its retained comparison evidence must authorize support."""

import asyncio
from copy import deepcopy
from datetime import date

import pytest

from mnemos.answer_support import AnswerSupportConfig, assess_answer_support
from mnemos.judgment import ChoiceAnswer, MockJudge
from tests.test_answer_support import fact, memory  # noqa: F401


def scripted(**changes):
    return {"value:*": "present", "source:*": "evidence:0", "cohort:*": "clear", **changes}


async def evaluate(memory, judge, ids=("one",), *, batching="candidate", **config):
    store, registry, context = memory
    return await assess_answer_support(
        "Which queue is current?",
        ids,
        store=store,
        registry=registry,
        context=context,
        judge=judge,
        reference_date=date(2026, 11, 6),
        config=AnswerSupportConfig(protocol="checked_v4", batching=batching, **config),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("batching", ["all", "candidate"])
async def test_three_checks_bind_own_evidence_and_preserve_qualified_content(memory, batching):
    fact(memory, "one", "Redis has not been approved for the production queue.")
    fact(memory, "two", "PostgreSQL is proposed for next year; no change is approved.")
    judge = MockJudge(scripted())
    before = [memory[0].get(key).model_dump() for key in ("one", "two")]
    result = await evaluate(memory, judge, ("one", "two"), batching=batching)
    assert result.status == "assessed"
    assert [d.evidence_ids for d in result.decisions] == [("one-evidence",), ("two-evidence",)]
    assert [c.content for c in result.supported] == [item["content"] for item in before]
    assert [memory[0].get(key).model_dump() for key in ("one", "two")] == before
    assert all(purpose == "answer_support_v4" for _, _, purpose in judge.calls)
    # The scripted answers test combination and binding, not semantic accuracy.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key,choice,disposition",
    [
        ("value:*", "absent", "related"),
        ("value:*", "unresolved", "unresolved"),
        ("source:*", "unsupported", "unsupported"),
        ("source:*", "unresolved", "unresolved"),
        ("cohort:*", "conflict", "unresolved"),
        ("cohort:*", "unresolved", "unresolved"),
    ],
)
async def test_each_negative_component_prevents_support(memory, key, choice, disposition):
    fact(memory, "one")
    result = await evaluate(memory, MockJudge(scripted(**{key: choice})))
    assert result.status == "assessed" and not result.supported
    assert result.decisions[0].disposition == disposition
    assert not result.decisions[0].evidence_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("component", ["value", "source", "cohort"])
async def test_each_component_must_meet_probability_threshold(memory, component):
    fact(memory, "one")
    expected = {"value": "present", "source": "evidence:0", "cohort": "clear"}

    def low_probability(key, question, state):
        chosen = expected[key.split(":")[0]]
        if key.startswith(component + ":"):
            alternative = next(option for option in question.keys if option != chosen)
            return ChoiceAnswer(chosen, {chosen: 0.79, alternative: 0.21}, 0.79)
        return ChoiceAnswer(
            chosen, {option: float(option == chosen) for option in question.keys}, 1.0
        )

    result = await evaluate(memory, MockJudge(low_probability))
    assert not result.supported and result.status == "assessed"
    assert result.decisions[0].disposition == "unresolved"
    assert result.decisions[0].reason == "low_probability"


@pytest.mark.asyncio
async def test_component_score_is_minimum_not_an_averaged_probability(memory):
    fact(memory, "one")
    answers = {"value": ("present", 0.9), "source": ("evidence:0", 0.8), "cohort": ("clear", 1.0)}

    def probabilities(key, question, state):
        chosen, probability = answers[key.split(":")[0]]
        alternate = next(option for option in question.keys if option != chosen)
        return ChoiceAnswer(chosen, {chosen: probability, alternate: 1 - probability}, probability)

    result = await evaluate(memory, MockJudge(probabilities))
    assert result.decisions[0].disposition == "supported"
    assert result.decisions[0].probability == 0.8


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["missing", "extra", "foreign_evidence"])
async def test_malformed_component_results_are_failures_without_abstention_credit(memory, mutation):
    fact(memory, "one")

    class Malformed(MockJudge):
        async def ask(self, *args, **kwargs):
            result = await super().ask(*args, **kwargs)
            if mutation == "missing":
                result.answers.pop("cohort:0")
            elif mutation == "extra":
                result.answers["source:99"] = result.answers["source:0"]
            else:
                result.answers["source:0"] = ChoiceAnswer("evidence:1", {"evidence:1": 1.0}, 1.0)
            return result

    result = await evaluate(memory, Malformed(scripted()))
    assert result.status == "unavailable" and not result.supported
    assert result.error_type == "JudgmentContractError"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["correct", "forget", "source", "scope", "arrive"])
async def test_comparison_changes_invalidate_all_three_component_results(memory, change):
    fact(memory, "one", "The queue used Redis in 2024.")
    if change != "arrive":
        fact(memory, "two", "PostgreSQL replaced Redis in 2026.")
    store = memory[0]

    class Changed(MockJudge):
        async def ask(self, *args, **kwargs):
            result = await super().ask(*args, **kwargs)
            if change == "arrive":
                fact(memory, "two", "PostgreSQL replaced Redis in 2026.")
            elif change == "forget":
                store.delete("two")
            elif change == "source":
                store._conn.execute(
                    "UPDATE fact_evidence SET excerpt='Only a proposal' WHERE fact_id='two'"
                )
                store._conn.commit()
            else:
                chunk = store.get("two")
                if change == "correct":
                    chunk = chunk.reconsolidate("Redis is still current.")
                else:
                    chunk.metadata["scope_id"] = "foreign"
                store.store(chunk)
            return result

    judge = Changed(scripted())
    result = await evaluate(memory, judge, ("one", "two"))
    assert len(judge.calls) == 1
    assert result.status == "partial" and not result.supported
    assert all(d.disposition == "stale" for d in result.decisions)


@pytest.mark.asyncio
async def test_scoped_comparison_payload_is_isolated_between_candidate_calls(memory):
    fact(memory, "one")
    fact(memory, "two")
    fact(memory, "foreign", "A private fact.", scope_id="foreign")
    fact(memory, "capture", "An unaccepted guess.", source="captured_passthrough")
    seen = []

    class Mutating(MockJudge):
        async def ask(self, state, *args, **kwargs):
            seen.append(deepcopy(state))
            result = await super().ask(state, *args, **kwargs)
            state["comparison_candidates"].clear()
            return result

    result = await evaluate(memory, Mutating(scripted()), ("one", "foreign", "two", "capture"))
    assert result.status == "assessed" and len(result.supported) == 2
    assert len(seen) == 2 and seen[0] == seen[1]
    assert len(seen[0]["comparison_candidates"]) == 2
    assert seen[0]["reference_date"] == "2026-11-06"


@pytest.mark.asyncio
async def test_three_checks_require_explicit_date_before_provider_work(memory):
    fact(memory, "one")
    store, registry, context = memory
    judge = MockJudge(scripted())
    with pytest.raises(ValueError, match="reference_date"):
        await assess_answer_support(
            "query",
            ["one"],
            store=store,
            registry=registry,
            context=context,
            judge=judge,
            config=AnswerSupportConfig(protocol="checked_v4"),
        )
    assert not judge.calls


@pytest.mark.asyncio
async def test_total_request_limit_rejects_before_any_component_call(memory):
    fact(memory, "one")
    judge = MockJudge(scripted())
    result = await evaluate(memory, judge, max_request_bytes=1)
    assert result.status == "unavailable" and not judge.calls
    assert result.error_type == "JudgmentRequestTooLarge"


@pytest.mark.asyncio
async def test_component_timeout_does_not_count_as_abstention(memory):
    fact(memory, "one")

    class Slow(MockJudge):
        async def ask(self, *args, **kwargs):
            await asyncio.sleep(1)
            return await super().ask(*args, **kwargs)

    result = await evaluate(memory, Slow(scripted()), timeout_seconds=0.01)
    assert result.status == "unavailable" and not result.supported
    assert result.error_type == "TimeoutError"
