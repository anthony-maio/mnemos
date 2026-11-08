from dataclasses import replace
import json

import pytest

from benchmarks.local_memory.condition_query_probe import (
    build_condition_cases,
    prepare_condition_probe,
)


def test_cohort_covers_interpretation_failures_without_grading_in_provider_input():
    cases = build_condition_cases("authored-test-interpreter-v1")
    execution, grading = prepare_condition_probe(cases)
    assert len(execution["requests"]) == len(grading["labels"]) == 16
    assert len({case.category for case in cases}) == 16
    for row, label in zip(execution["requests"], grading["labels"], strict=True):
        assert row["key"] == label["key"]
        assert set(row) == {"key", "owner_request", "prompt", "schema"}
        model_input = json.loads(row["prompt"].split("\nINPUT\n", 1)[1])
        assert set(model_input) == {"question", "predicates"}
        assert model_input["question"] == row["owner_request"]["query"]
        assert set(label["expected"]) == set(model_input["predicates"])
        assert "expected" not in model_input


def test_same_question_with_all_or_any_prerequisites_keeps_query_interpretation_constant():
    cases = {case.key: case for case in build_condition_cases("test-v1")}
    conjunction = cases["11-one-prerequisite"]
    alternative = cases["12-alternative"]
    assert conjunction.request.query == alternative.request.query
    assert conjunction.request.predicates == alternative.request.predicates
    assert conjunction.expected == alternative.expected
    assert conjunction.expected_applicability == "unresolved"
    assert alternative.expected_applicability == "applies"


def test_unknown_under_negation_and_unspecified_cases_are_not_positive_examples():
    cases = {case.key: case for case in build_condition_cases("test-v1")}
    for key in [
        "03-unstated",
        "04-phrase",
        "05-report",
        "06-question",
        "08-reported-hypothetical",
        "09-foreign",
        "13-negated-missing",
        "15-instruction",
    ]:
        assert all(
            item == {"state": "not_established", "quote": None}
            for item in cases[key].expected.values()
        )
        assert cases[key].expected_applicability == "unresolved"
    assert cases["07-hypothetical"].expected_applicability == "applies"
    assert cases["14-negated-false"].expected_applicability == "applies"


def test_invalid_expected_applicability_and_duplicate_cases_fail_preparation():
    cases = build_condition_cases("test-v1")
    with pytest.raises(ValueError):
        prepare_condition_probe((cases[0], cases[0]))
    with pytest.raises(ValueError):
        prepare_condition_probe((replace(cases[0], expected_applicability="excluded"),))
