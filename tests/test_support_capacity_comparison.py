"""Offline paired-input and failure-accounting checks; no models run here."""

import copy

import pytest

from benchmarks.local_memory.answer_support_comparison import SupportTreatment, _seed_case
from benchmarks.local_memory import support_capacity_comparison as capacity
from mnemos.answer_support import AnswerSupportConfig
from mnemos.judgment import MockJudge
from mnemos.utils.reliability import MnemosTransientError
from tests.test_answer_support_comparison import FixedEmbedding


@pytest.mark.asyncio
async def test_models_share_frozen_candidates_without_retrieval_or_source_mutation(tmp_path):
    case = {
        "id": "ownership",
        "facts": [
            {"id": "local", "content": "Nora owns the billing API."},
            {"id": "foreign", "content": "River owns search.", "scope": "foreign"},
        ],
        "queries": [
            {"id": "owner", "query": "Who owns the billing API?", "expected_fact_ids": ["local"]}
        ],
    }
    service, context = _seed_case(tmp_path / "accepted", case, FixedEmbedding())
    try:
        before = [f.model_dump() for f in service.engine.store.get_all()]
        inputs = await capacity.freeze_inputs(service, context, case)
        assert len(inputs) == 4
        original = inputs[0].record
        original["baseline_ids"].append("mutated-copy")
        assert "mutated-copy" not in inputs[0].record["baseline_ids"]
        select = MockJudge({"support:*": "evidence:0"})
        decline = MockJudge({"support:*": "related"})
        policies = [
            SupportTreatment("small", select, AnswerSupportConfig(batching="candidate")),
            SupportTreatment("large", decline, AnswerSupportConfig(batching="candidate")),
        ]
        rows = [
            await capacity.evaluate_input(entry, policy, policy.name)
            for policy in policies
            for entry in inputs
        ]
        capacity.verify_pairing(rows, {"small", "large"})
        assert all("foreign" not in r["baseline_ids"] for r in rows)
        assert all(r["supported_ids"] == (["local"] if r["model"] == "small" else []) for r in rows)
        assert all(
            state == {"query": case["queries"][0]["query"]}
            for state, _, _ in select.calls + decline.calls
        )
        assert [f.model_dump() for f in service.engine.store.get_all()] == before
        capacity.verify_stores(inputs)
        for bad in [rows[:-1], rows + [rows[0]]]:
            with pytest.raises(ValueError, match="treatment group"):
                capacity.verify_pairing(bad, {"small", "large"})
        changed = copy.deepcopy(rows)
        changed[0]["input_sha256"] = "changed"
        with pytest.raises(ValueError, match="different frozen inputs"):
            capacity.verify_pairing(changed, {"small", "large"})
        original_fact = service.engine.store.get("local")
        assert service.engine.store.update(
            "local", original_fact.model_copy(update={"content": "Changed"})
        )
        with pytest.raises(ValueError, match="store changed"):
            capacity.verify_stores(inputs)
    finally:
        await service.engine.aclose()


@pytest.mark.asyncio
async def test_capacity_retains_failed_observation_without_abstention_credit(tmp_path):
    case = {
        "id": "failure",
        "facts": [{"id": "local", "content": "The release is unapproved."}],
        "queries": [{"id": "unknown", "query": "Who approved release?", "expected_fact_ids": []}],
    }
    service, context = _seed_case(tmp_path / "accepted", case, FixedEmbedding())

    class FailedJudge(MockJudge):
        async def ask(self, *args, **kwargs):
            raise MnemosTransientError("synthetic outage")

    try:
        inputs = await capacity.freeze_inputs(service, context, case)
        policy = SupportTreatment(
            "failed", FailedJudge(), AnswerSupportConfig(batching="candidate")
        )
        row = await capacity.evaluate_input(inputs[0], policy, "failed-model")
        assert row["support_status"] == "unavailable"
        assert row["support_error_type"] == "MnemosTransientError"
        assert not row["support_grade"]["correct_abstention"]
        assert row["model"] == "failed-model" and row["input_sha256"]
    finally:
        await service.engine.aclose()
