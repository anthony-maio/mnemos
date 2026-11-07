"""A correct review operation cannot conceal a wrong semantic relation."""

from copy import deepcopy
import json

import pytest

from benchmarks.local_memory.referent_evaluation import (
    CapturedReferents,
    grade_relations,
    load_relation_annotations,
    score_stability,
)
from benchmarks.local_memory.build_referent_stability import build_corpus
from benchmarks.local_memory.reconciliation_evaluation import (
    load_reconciliation_cases,
    grade_reconciliation,
)
from mnemos.extraction.source_target_identity import reconcile_source_target_identity
from tests.test_source_reconciliation import Provider
from pathlib import Path
from jsonschema import Draft202012Validator


def raw_response(relation="insufficient_identity", target_id="target-a", version=4):
    return json.dumps(
        {
            "candidate_value": None,
            "comparisons": [
                {
                    "target_fact_id": target_id,
                    "expected_version": version,
                    "target_value": None,
                    "relation": relation,
                }
            ],
        }
    )


def raw_case(case_id="case/base", variant="base", target_id="target-a"):
    return {
        "id": case_id,
        "base_id": "case",
        "variant": variant,
        "events": [{"content": "The count must be sixteen."}],
        "selections": [{"kind": "continuing_requirement", "revises_existing_requirement": True}],
        "targets": [{"id": target_id, "version": 4, "content": "There are twelve workers."}],
        "expected_relations": {target_id: "insufficient_identity"},
        "logical_target_ids": {target_id: "workers"},
    }


def test_wrong_intermediate_relation_does_not_receive_review_credit() -> None:
    grade = grade_relations(raw_case(), [raw_response("different_variable")], error=None)
    assert grade["response_valid"] and not grade["relation_case_exact"]
    assert grade["relations_correct"] == 0 and grade["relations_expected"] == 1


def test_correct_relation_and_complete_target_coverage_pass() -> None:
    grade = grade_relations(raw_case(), [raw_response()], error=None)
    assert grade["relation_case_exact"] and grade["relations_correct"] == 1


@pytest.mark.parametrize(
    "responses,error",
    [
        ([], None),
        (["not json"], None),
        ([raw_response()] * 2, None),
        ([raw_response("other")], None),
        ([raw_response(target_id="missing")], None),
        ([raw_response(version=99)], None),
        ([raw_response()], "Runtime failure"),
        ([json.dumps({"comparisons": [json.loads(raw_response())["comparisons"][0]] * 2})], None),
    ],
)
def test_missing_invalid_stale_duplicate_or_failed_responses_cannot_pass(responses, error) -> None:
    grade = grade_relations(raw_case(), responses, error=error)
    assert not grade["relation_case_exact"] and grade["relations_correct"] == 0


@pytest.mark.parametrize(
    "change", ["missing", "unknown", "duplicate_logical", "changed_text", "changed_label"]
)
def test_annotation_validation_refuses_inconsistent_ground_truth(change) -> None:
    first, second = raw_case(), raw_case("case/renamed", "renamed", "other-id")
    if change == "missing":
        second["expected_relations"] = {}
    elif change == "unknown":
        second["expected_relations"]["other-id"] = "unknown"
    elif change == "duplicate_logical":
        second["targets"].append({"id": "third", "content": "Another target.", "version": 4})
        second["expected_relations"]["third"] = "different_variable"
        second["logical_target_ids"]["third"] = "workers"
    elif change == "changed_text":
        second["targets"][0]["content"] = "Changed text."
    else:
        second["expected_relations"]["other-id"] = "different_variable"
    with pytest.raises(ValueError):
        load_relation_annotations({"cases": [first, second]})


def stability_inputs():
    cases = [raw_case(), raw_case("case/renamed", "renamed", "other-id")]
    annotations = load_relation_annotations({"cases": cases})
    rows = [
        {
            "case_id": case["id"],
            "arm": "referent_v5",
            "repetition": 0,
            "error": None,
            "result": [
                {"operation": "unresolved", "target_fact_id": None, "expected_version": None}
            ],
            "relation_grade": grade_relations(
                case, [raw_response(target_id=case["targets"][0]["id"])], error=None
            ),
        }
        for case in cases
    ]
    return annotations, rows


def test_stability_compares_logical_targets_after_id_renaming() -> None:
    annotations, rows = stability_inputs()
    result = score_stability(rows, annotations, expected_arms=["referent_v5"], repetitions=1)
    assert result == {
        "groups_expected": 1,
        "groups_complete": 1,
        "operation_stable": 1,
        "relations_stable": 1,
        "both_stable": 1,
    }


def test_stable_operation_does_not_hide_relation_instability() -> None:
    annotations, rows = stability_inputs()
    rows[1]["relation_grade"]["observed_relations"]["other-id"] = "different_variable"
    result = score_stability(rows, annotations, expected_arms=["referent_v5"], repetitions=1)
    assert (
        result["operation_stable"] == 1
        and result["relations_stable"] == 0
        and result["both_stable"] == 0
    )


@pytest.mark.parametrize("change", ["missing", "duplicate", "failed", "empty"])
def test_incomplete_groups_remain_in_stability_denominator(change) -> None:
    annotations, rows = stability_inputs()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif change == "failed":
        rows[0]["error"] = "failure"
    else:
        rows = []
    result = score_stability(rows, annotations, expected_arms=["referent_v5"], repetitions=1)
    assert result["groups_expected"] == 1 and result["both_stable"] == 0


def test_frozen_corpus_matches_authoring_source_and_preserves_relation_labels() -> None:
    path = (
        Path(__file__).parents[1]
        / "benchmarks/local_memory/fixtures/referent-stability-development-v1.json"
    )
    data = json.loads(path.read_text())
    assert data == build_corpus()
    annotations = load_relation_annotations(data)
    assert len(annotations) == 72
    assert len({case["base_id"] for case in annotations.values()}) == 12
    assert len({case["cluster_id"] for case in annotations.values()}) == 5
    assert sum(len(case["targets"]) for case in annotations.values()) == 198
    for base in {case["base_id"] for case in annotations.values()}:
        assert (
            len({case["variant"] for case in annotations.values() if case["base_id"] == base}) == 6
        )


@pytest.mark.asyncio
async def test_every_labeled_view_is_mechanically_eligible_and_scores_both_layers() -> None:
    path = (
        Path(__file__).parents[1]
        / "benchmarks/local_memory/fixtures/referent-stability-development-v1.json"
    )
    data, cases = load_reconciliation_cases(path)
    annotations = load_relation_annotations(data)
    rows = []
    for case in cases:
        annotation = annotations[case.source.case_id]
        provider = Provider(
            {
                "candidate_value": None,
                "comparisons": [
                    {
                        "target_fact_id": t["id"],
                        "expected_version": t["version"],
                        "target_value": None,
                        "relation": annotation["expected_relations"][t["id"]],
                    }
                    for t in annotation["targets"]
                ],
            }
        )
        captured = CapturedReferents(provider)
        result = await reconcile_source_target_identity(
            [case.candidate],
            list(case.source.events),
            list(case.source.targets),
            captured,
            comparison_style="referent",
        )
        assert len(provider.calls) == len(captured.responses) == 1
        assert captured.responses[0] == json.dumps(provider.response)
        Draft202012Validator(provider.calls[0][1]["format"]).validate(provider.response)
        assert grade_reconciliation(case, result, error=None)["case_exact"]
        grade = grade_relations(annotation, captured.responses, error=None)
        assert grade["relation_case_exact"]
        rows.append(
            {
                "case_id": case.source.case_id,
                "arm": "referent_v5",
                "repetition": 0,
                "error": None,
                "result": [r.model_dump() for r in result],
                "relation_grade": grade,
            }
        )
    assert (
        score_stability(rows, annotations, expected_arms=["referent_v5"], repetitions=1)[
            "both_stable"
        ]
        == 12
    )
