from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchmarks.local_memory.joint_support_corpus import assemble
from benchmarks.local_memory.joint_support_critical import apply_critical_registry
from benchmarks.local_memory.joint_support_grading import grade_selections


def fixture():
    rows, labels = assemble()
    registry = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "benchmarks/local_memory/joint_support_critical_registry.json"
        ).read_text()
    )
    labels, _ = apply_critical_registry(rows, labels, registry)
    observations = {
        e["id"]: dict(
            status="assessed",
            selected_ids=[t["id"] for t in labels[e["id"]]["targets"] if t["supported"]],
        )
        for e in rows
        if e["split"] == "development"
    }
    return rows, labels, observations


def test_exact_gold_has_full_credit_but_is_not_acceptance():
    rows, labels, observations = fixture()
    result = grade_selections(rows, labels, observations, "development")
    assert all(v == 1 for v in result["metrics"].values())
    assert result["status"] == "diagnostic_thresholds_met"
    assert not result["fresh_acceptance"] and not result["full_retrieval_timing"]
    assert result["split"] == "development"
    assert result["counts"]["distinct_critical_facts"] == 2
    assert set(result["critical_facts"]) == {"pass-rule", "build-no-both"}
    for outcome in result["critical_facts"].values():
        assert outcome["applicable_cases"] == outcome["retained_cases"]
        assert not outcome["missing_cases"] and not outcome["unavailable_cases"]


def test_all_abstention_cannot_pass_supported_or_critical_retention():
    rows, labels, observations = fixture()
    for observation in observations.values():
        observation["selected_ids"] = []
    result = grade_selections(rows, labels, observations, "development")
    assert result["metrics"]["precision"] is None
    assert result["metrics"]["critical_retention"] == 0
    assert result["metrics"]["supported_retention"] == 0
    assert result["status"] == "diagnostic_failed"


def test_unavailable_never_counts_as_correct_abstention():
    rows, labels, observations = fixture()
    for observation in observations.values():
        observation.update(status="unavailable", selected_ids=[])
    result = grade_selections(rows, labels, observations, "development")
    assert result["metrics"]["abstention"] == 0
    assert result["counts"]["invalid_or_unavailable"] == len(observations)
    for outcome in result["critical_facts"].values():
        assert (
            outcome["applicable_cases"] == outcome["missing_cases"] == outcome["unavailable_cases"]
        )
        assert not outcome["retained_cases"]


@pytest.mark.parametrize("selected", [["unknown"], ["unknown", "unknown"], "unknown", [None]])
def test_invalid_selections_fail_closed(selected):
    rows, labels, observations = fixture()
    observations[next(iter(observations))]["selected_ids"] = selected
    result = grade_selections(rows, labels, observations, "development")
    assert result["counts"]["invalid_or_unavailable"] == 1
    assert result["status"] == "diagnostic_failed"


def test_missing_observation_cannot_shrink_denominator():
    rows, labels, observations = fixture()
    observations.pop(next(iter(observations)))
    with pytest.raises(ValueError, match="exactly one"):
        grade_selections(rows, labels, observations, "development")


def test_unsupported_selection_lowers_precision_and_preserves_reason():
    rows, labels, observations = fixture()
    for identity, observation in observations.items():
        negative = next((t for t in labels[identity]["targets"] if not t["supported"]), None)
        if negative:
            observation["selected_ids"].append(negative["id"])
            break
    before = deepcopy((rows, labels, observations))
    result = grade_selections(rows, labels, observations, "development")
    assert result["counts"]["unsupported"] == 1
    assert result["metrics"]["precision"] < 1
    assert result["false_support_by_reason"][negative["reason"]] == 1
    assert (rows, labels, observations) == before
