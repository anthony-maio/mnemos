"""Literal identity witnesses are required, but are not semantic acceptance tests."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json

from jsonschema import Draft202012Validator
import pytest

from benchmarks.local_memory.build_referent_stability import build_corpus
from benchmarks.local_memory.reconciliation_evaluation import load_reconciliation_cases
from benchmarks.local_memory.referent_evaluation import grade_pairwise_relations
from mnemos.extraction.source_identity_spans import (
    DATA_MARKER,
    compare_identity_spans,
    comparison_from_identity_spans,
    identity_span_schema,
    validate_identity_spans,
)
from mnemos.extraction.source_target_identity import reconcile_source_target_identity
from tests.test_source_reconciliation import Provider, candidate, event, target


def span(index):
    return {"first_word": index, "last_word": index}


def fixture():
    source = event("source", "Correction: the Cedar production schedule is Tuesday.")
    observed = target("storage-private", "The Cedar production schedule is Monday.")
    observed.version = 19
    item = candidate(source, "correction_or_rejection")
    raw = {
        "subject": {"relation": "same", "candidate_span": span(2), "target_span": span(1)},
        "attribute": {"relation": "same", "candidate_span": span(4), "target_span": span(3)},
        "qualifiers": {"relation": "same", "candidate_span": span(3), "target_span": span(2)},
    }
    return source, item, observed, raw


@pytest.mark.parametrize("side", ["candidate_span", "target_span"])
@pytest.mark.parametrize("dimension", ["subject", "attribute", "qualifiers"])
@pytest.mark.parametrize("relation", ["same", "different"])
def test_claimed_match_or_difference_cannot_use_one_missing_witness(side, dimension, relation):
    _, item, observed, raw = fixture()
    raw[dimension][side] = None
    raw[dimension]["relation"] = relation
    with pytest.raises(ValueError, match="evidence on both sides"):
        comparison_from_identity_spans(json.dumps(raw), item, observed)


@pytest.mark.parametrize("dimension", ["subject", "attribute", "qualifiers"])
def test_missing_witness_can_be_reported_as_unspecified(dimension):
    _, item, observed, raw = fixture()
    raw[dimension].update(relation="unspecified", candidate_span=None)
    assert (
        comparison_from_identity_spans(json.dumps(raw), item, observed).relation
        == "insufficient_identity"
    )


def test_qualifiers_absent_on_both_sides_do_not_force_an_unresolved_identity():
    _, item, observed, raw = fixture()
    raw["qualifiers"].update(candidate_span=None, target_span=None)
    assert (
        comparison_from_identity_spans(json.dumps(raw), item, observed).relation == "same_variable"
    )
    # A model can omit a qualifier that is actually present. Structural checks
    # cannot detect that semantic error; real-model relation scoring must do so.


@pytest.mark.parametrize(
    "bad_span",
    [
        {"first_word": True, "last_word": 2},
        {"first_word": -1, "last_word": 2},
        {"first_word": 3, "last_word": 2},
        {"first_word": 0, "last_word": 99},
    ],
)
def test_bad_span_locations_fail_instead_of_becoming_review(bad_span):
    _, item, observed, raw = fixture()
    raw["qualifiers"]["candidate_span"] = bad_span
    with pytest.raises(ValueError):
        validate_identity_spans(json.dumps(raw), item, observed)


def test_schema_has_independent_source_bounds_and_no_storage_authority():
    _, item, observed, raw = fixture()
    validator = Draft202012Validator(identity_span_schema(item, observed))
    assert validator.is_valid(raw)
    forged = {**raw, "target_fact_id": "other-storage"}
    assert not validator.is_valid(forged)
    with pytest.raises(ValueError):
        validate_identity_spans(json.dumps(forged), item, observed)
    raw["subject"]["target_span"] = span(6)
    assert not validator.is_valid(raw)  # Target ends at word 5; candidate ends at 6.


@pytest.mark.asyncio
async def test_complete_evidence_restores_runtime_identity_without_rewriting_source():
    source, item, observed, raw = fixture()
    provider = Provider(raw)
    result = await reconcile_source_target_identity(
        [item], [source], [observed], provider, comparison_style="identity_spans"
    )
    assert result[0].operation == "correct"
    assert result[0].target_fact_id == observed.id and result[0].expected_version == 19
    assert result[0].content == source.content
    payload = json.loads(provider.calls[0][0].split(DATA_MARKER, 1)[1])
    assert set(payload) == {"candidate", "target"}
    assert set(payload["target"]) == {"content", "words"}
    assert observed.id not in provider.calls[0][0]


@pytest.mark.asyncio
async def test_missing_environment_requirement_keeps_review_instead_of_correcting():
    source, item, observed, raw = fixture()
    raw["qualifiers"].update(relation="unspecified", candidate_span=None)
    item = replace(item, kind="continuing_requirement", revises_existing_requirement=True)
    result = await reconcile_source_target_identity(
        [item], [source], [observed], Provider(raw), comparison_style="identity_spans"
    )
    assert result[0].operation == "unresolved" and result[0].target_fact_id is None


@pytest.mark.asyncio
async def test_later_invalid_pair_cannot_return_an_earlier_valid_correction():
    source, item, observed, raw = fixture()

    class Later(Provider):
        async def predict(self, prompt, **kwargs):
            if self.calls:
                self.response = deepcopy(raw)
                self.response["qualifiers"]["candidate_span"] = None
            return await super().predict(prompt, **kwargs)

    with pytest.raises(ValueError, match="evidence on both sides"):
        await reconcile_source_target_identity(
            [item],
            [source],
            [observed, target("second", "The Cedar staging schedule is Monday.")],
            Later(raw),
            comparison_style="identity_spans",
        )


@pytest.mark.asyncio
async def test_changed_target_is_rejected_after_span_comparison():
    source, item, observed, raw = fixture()

    class Mutating(Provider):
        async def predict(self, prompt, **kwargs):
            observed.version += 1
            return await super().predict(prompt, **kwargs)

    with pytest.raises(ValueError, match="changed during"):
        await reconcile_source_target_identity(
            [item], [source], [observed], Mutating(raw), comparison_style="identity_spans"
        )


@pytest.mark.asyncio
async def test_each_literal_pair_has_identical_request_across_all_matched_views(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(build_corpus()))
    _, cases = load_reconciliation_cases(path)
    requests = {}
    count = 0
    for case in cases:
        raw = {
            name: {"relation": "unspecified", "candidate_span": None, "target_span": None}
            for name in ("subject", "attribute", "qualifiers")
        }
        provider = Provider(raw)
        await compare_identity_spans(case.candidate, list(case.source.targets), provider)
        for observed, (prompt, kwargs) in zip(case.source.targets, provider.calls):
            key = (case.candidate.content, observed.content)
            requests.setdefault(key, set()).add(
                hashlib.sha256(json.dumps([prompt, kwargs], sort_keys=True).encode()).hexdigest()
            )
            count += 1
    assert count == 198 and len(requests) == 34 and all(len(v) == 1 for v in requests.values())


def test_invalid_dimension_receives_no_partial_relation_credit():
    _, item, observed, raw = fixture()
    raw["subject"]["candidate_span"] = None
    case = {
        "targets": [{"id": observed.id, "version": 19}],
        "expected_relations": {observed.id: "same_variable"},
    }
    result = grade_pairwise_relations(
        case, item, [observed], [json.dumps(raw)], error=None, identity_spans=True
    )
    assert not result["response_valid"] and result["relations_correct"] == 0
