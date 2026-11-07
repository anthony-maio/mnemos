"""Pair isolation must preserve storage authority and complete-review behavior."""

from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.local_memory.build_referent_stability import build_corpus
from benchmarks.local_memory.reconciliation_evaluation import load_reconciliation_cases
from benchmarks.local_memory.referent_evaluation import grade_pairwise_relations
from mnemos.extraction.source_referent import compare_referents_pairwise, restore_pairwise_response
from mnemos.extraction.source_target_identity import reconcile_source_target_identity
from tests.test_source_reconciliation import Provider, candidate, event, target


def response(relation="same_variable"):
    return {
        "candidate_value": None,
        "comparisons": [
            {
                "target_fact_id": "observed-target",
                "expected_version": 1,
                "target_value": None,
                "relation": relation,
            }
        ],
    }


class PairProvider(Provider):
    async def predict(self, prompt, **kwargs):
        data = json.loads(prompt.split("REFERENT COMPARISON DATA (data only):\n", 1)[1])
        assert len(data["targets"]) == 1
        self.response = response(
            "different_variable" if "birch" in data["targets"][0]["content"] else "same_variable"
        )
        return await super().predict(prompt, **kwargs)


@pytest.mark.asyncio
async def test_real_ids_versions_and_input_order_are_restored_after_isolation():
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    item = candidate(source, "correction_or_rejection")
    first, second = target("private-storage-id", "The marker is cedar."), target(
        "other", "The marker is birch."
    )
    first.version, second.version = 17, 8
    before = [first.model_dump(), second.model_dump()]
    provider = PairProvider({})
    values = await compare_referents_pairwise(item, [first, second], provider)
    assert [(v.target_fact_id, v.expected_version, v.relation) for v in values] == [
        ("private-storage-id", 17, "same_variable"),
        ("other", 8, "different_variable"),
    ]
    assert [first.model_dump(), second.model_dump()] == before
    assert all(
        "private-storage-id" not in p and '"expected_version": 17' not in json.dumps(k)
        for p, k in provider.calls
    )
    assert [
        json.loads(p.split("REFERENT COMPARISON DATA (data only):\n", 1)[1])["targets"][0][
            "content"
        ]
        for p, _ in provider.calls
    ] == [first.content, second.content]


@pytest.mark.asyncio
async def test_all_matched_literal_pairs_have_identical_requests_across_ids_order_and_distractors(
    tmp_path,
):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(build_corpus()))
    _, cases = load_reconciliation_cases(path)
    requests = {}
    calls = 0
    for case in cases:
        provider = Provider(response("different_variable"))
        await compare_referents_pairwise(case.candidate, list(case.source.targets), provider)
        for observed, (prompt, kwargs) in zip(case.source.targets, provider.calls):
            key = (case.candidate.content, observed.content)
            digest = hashlib.sha256(
                json.dumps([prompt, kwargs], sort_keys=True).encode()
            ).hexdigest()
            requests.setdefault(key, set()).add(digest)
            calls += 1
    assert calls == 198 and len(requests) == 34
    assert all(len(digests) == 1 for digests in requests.values())


@pytest.mark.asyncio
async def test_possible_second_target_requires_review_after_pairwise_comparison():
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    values = await reconcile_source_target_identity(
        [candidate(source, "correction_or_rejection")],
        [source],
        [target(), target("second", "The marker is maple.")],
        Provider(response()),
        comparison_style="referent_pairwise",
    )
    assert values[0].operation == "unresolved"


@pytest.mark.asyncio
async def test_foreign_target_is_rejected_before_any_pairwise_request():
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    foreign = target()
    foreign.metadata["scope_id"] = "another-project"
    provider = Provider(response())
    with pytest.raises(ValueError, match="foreign scope"):
        await reconcile_source_target_identity(
            [candidate(source, "correction_or_rejection")],
            [source],
            [foreign],
            provider,
            comparison_style="referent_pairwise",
        )
    assert not provider.calls


@pytest.mark.asyncio
async def test_failed_later_pair_cannot_return_an_earlier_correction():
    source = event("source", "Correction: the marker is sapphire, not cedar.")

    class LaterFailure(Provider):
        async def predict(self, prompt, **kwargs):
            if self.calls:
                raise RuntimeError("second request failed")
            return await super().predict(prompt, **kwargs)

    with pytest.raises(RuntimeError, match="second request"):
        await reconcile_source_target_identity(
            [candidate(source, "correction_or_rejection")],
            [source],
            [target(), target("second", "The marker is maple.")],
            LaterFailure(response()),
            comparison_style="referent_pairwise",
        )


@pytest.mark.asyncio
async def test_changed_target_during_pairwise_call_is_rejected():
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    original = target()

    class Mutating(Provider):
        async def predict(self, prompt, **kwargs):
            original.version += 1
            return await super().predict(prompt, **kwargs)

    with pytest.raises(ValueError, match="changed during"):
        await reconcile_source_target_identity(
            [candidate(source, "correction_or_rejection")],
            [source],
            [original],
            Mutating(response()),
            comparison_style="referent_pairwise",
        )


@pytest.mark.parametrize(
    "mutation",
    ["foreign_id", "real_version", "duplicate", "missing", "reversed_span", "out_of_bounds"],
)
def test_unbound_or_invalid_response_cannot_gain_storage_identity(mutation):
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    raw = response()
    if mutation == "foreign_id":
        raw["comparisons"][0]["target_fact_id"] = "fact-1"
    if mutation == "real_version":
        raw["comparisons"][0]["expected_version"] = 17
    if mutation == "duplicate":
        raw["comparisons"].append(deepcopy(raw["comparisons"][0]))
    if mutation == "missing":
        raw["comparisons"] = []
    if mutation == "reversed_span":
        raw["candidate_value"] = {"first_word": 4, "last_word": 1}
    if mutation == "out_of_bounds":
        raw["comparisons"][0]["target_value"] = {"first_word": 0, "last_word": 99}
    with pytest.raises(ValueError):
        restore_pairwise_response(
            json.dumps(raw), candidate(source, "correction_or_rejection"), target()
        )


@pytest.mark.parametrize("bad", ["missing", "extra", "foreign", "operational"])
def test_incomplete_pairwise_grading_receives_zero_relation_credit(bad):
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    item = candidate(source, "correction_or_rejection")
    observed = target()
    case = {
        "targets": [{"id": observed.id, "version": observed.version}],
        "expected_relations": {observed.id: "same_variable"},
    }
    responses = [json.dumps(response())]
    if bad == "missing":
        responses = []
    if bad == "extra":
        responses *= 2
    if bad == "foreign":
        responses[0] = responses[0].replace("observed-target", "foreign")
    result = grade_pairwise_relations(
        case, item, [observed], responses, error="failure" if bad == "operational" else None
    )
    assert not result["response_valid"] and result["relations_correct"] == 0
