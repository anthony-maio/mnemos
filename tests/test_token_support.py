import asyncio
import json
import math

import pytest

from benchmarks.local_memory import token_support
from benchmarks.local_memory.token_support import TokenSourceSelector
from mnemos.utils.reliability import MnemosTransientError
from tests.test_answer_support import fact, memory  # noqa: F401
from tests.test_support_intersection import Selector, assess


def reply(choice="A", probability=0.95):
    return {
        "done": True,
        "response": choice,
        "logprobs": [
            {
                "token": choice,
                "logprob": math.log(probability),
                "top_logprobs": [
                    {"token": choice, "logprob": math.log(probability)},
                    {"token": "B" if choice == "A" else "A", "logprob": math.log(1 - probability)},
                ],
            }
        ],
    }


@pytest.mark.asyncio
async def test_full_cohort_and_source_witnesses_preserved_without_short_circuit(memory):
    fact(memory, "one", "The inspection was proposed, not approved.")
    fact(memory, "two", "The current checklist is docs/current.md.")
    records = []

    async def predict(prompt):
        shared = json.loads(prompt.split("DATA:\n", 1)[1].split("\nTARGET:\n", 1)[0])
        assert [c["fact_id"] for c in shared["comparison_candidates"]] == ["one", "two"]
        assert shared["reference_date"] == "2026-11-07"
        target = json.loads(prompt.split("\nTARGET:\n")[1].split("\nANSWER:")[0])
        return reply("B" if target["fact_id"] == "one" else "A")

    result = await assess(
        memory, Selector(("one", "two")), TokenSourceSelector(predict, records.append)
    )
    assert result.status == "assessed"
    assert [c.fact_id for c in result.supported] == ["two"]
    assert result.components[1].selected[0].evidence_ids == ("two-evidence",)
    assert len(records) == 2 and all(r["status"] == "assessed" for r in records)


@pytest.mark.asyncio
@pytest.mark.parametrize("probability,accepted", [(0.89, False), (0.9, True), (0.99, True)])
async def test_fixed_prospective_rule_without_normalization(memory, probability, accepted):
    fact(memory, "one")

    async def predict(_):
        return reply(probability=probability)

    result = await assess(memory, Selector(("one",)), TokenSourceSelector(predict), ids=("one",))
    assert bool(result.supported) == accepted


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing_label", "provider", "cancel", "stale"])
async def test_failure_is_not_successful_abstention(memory, failure):
    fact(memory, "one")
    records = []

    async def predict(_):
        if failure == "provider":
            raise MnemosTransientError("offline")
        if failure == "cancel":
            raise asyncio.CancelledError()
        if failure == "stale":
            memory[0].delete("one")
        raw = reply()
        if failure == "missing_label":
            raw["logprobs"][0]["top_logprobs"].pop()
        return raw

    call = assess(
        memory, Selector(("one",)), TokenSourceSelector(predict, records.append), ids=("one",)
    )
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await call
    else:
        result = await call
        assert result.status == ("partial" if failure == "stale" else "unavailable")
        assert not result.supported
    assert records[0]["status"] == "unavailable" and records[0]["error_type"]


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["MAX_REQUESTS", "MAX_TOTAL_PROMPT_BYTES"])
async def test_whole_request_budget_rejects_before_provider_work(memory, monkeypatch, limit):
    fact(memory, "one")
    fact(memory, "two")
    monkeypatch.setattr(token_support, limit, 1)

    async def predict(_):
        raise AssertionError("Provider must not run")

    result = await assess(memory, Selector(("one", "two")), TokenSourceSelector(predict))
    assert result.status == "unavailable" and not result.supported


@pytest.mark.asyncio
async def test_foreign_fact_never_enters_model_prompt(memory):
    fact(memory, "one")
    fact(memory, "two", "PRIVATE FOREIGN SENTINEL", scope_id="other")
    calls = []

    async def predict(prompt):
        calls.append(prompt)
        assert "PRIVATE FOREIGN SENTINEL" not in prompt
        return reply()

    result = await assess(memory, Selector(("one",)), TokenSourceSelector(predict))
    assert result.status == "assessed" and len(calls) == 1
    assert result.decisions[1].disposition == "ineligible"


@pytest.mark.asyncio
async def test_later_failure_discards_earlier_positive_witness(memory):
    fact(memory, "one")
    fact(memory, "two")
    records = []

    async def predict(_):
        if records:
            raise MnemosTransientError("second request failed")
        return reply()

    result = await assess(
        memory, Selector(("one", "two")), TokenSourceSelector(predict, records.append)
    )
    assert records[0]["selected"] is True
    assert records[1]["status"] == "unavailable"
    assert result.status == "unavailable" and not result.supported
