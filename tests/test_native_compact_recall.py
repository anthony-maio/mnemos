"""Fixture grading tests; these do not establish native model performance."""

import json

import pytest

from benchmarks.native_hosts.compact_recall import (
    EXPECTED,
    arguments,
    grade_answer,
    prompt,
    validate_usage,
)


def test_treatment_changes_only_the_requested_response_format():
    assert prompt("full").replace(
        '"response_format": "full"', '"response_format": "compact"'
    ) == prompt("compact")
    full, compact = arguments("full"), arguments("compact")
    full.pop("response_format")
    compact.pop("response_format")
    assert full == compact


def test_all_complete_qualified_answers_pass():
    result = grade_answer(json.dumps(EXPECTED))
    assert result["passed"] and result["correct_fields"] == 8


@pytest.mark.parametrize(
    "key,value",
    [
        ("production_log_days", 7),
        ("export_archive_days", 7),
        ("production_release_reviewers", 2),
        ("production_release_conflict", False),
        ("production_deployment_queue", "blue"),
        ("export_archive_encrypted", 1),
    ],
)
def test_dropped_qualifiers_conflicts_and_boolean_coercion_fail(key, value):
    result = grade_answer(json.dumps({**EXPECTED, key: value}))
    assert not result["passed"] and result["correct_fields"] == 7


@pytest.mark.parametrize("text", ["{}", "[]", '{"a": 1, "a": 2}', "```json\n{}\n```"])
def test_missing_or_malformed_answers_fail(text):
    assert not grade_answer(text)["passed"]


def test_extra_answer_key_cannot_pass():
    result = grade_answer(json.dumps({**EXPECTED, "extra": "unsupported"}))
    assert result["correct_fields"] == 8 and not result["passed"]


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"input_tokens": 1, "cached_input_tokens": 2, "output_tokens": 0},
        {"input_tokens": True, "cached_input_tokens": 0, "output_tokens": 0},
    ],
)
def test_incomplete_or_inconsistent_usage_is_rejected(usage):
    with pytest.raises(ValueError):
        validate_usage(usage)


def test_native_usage_preserves_cached_counts_separately():
    usage = {"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 20}
    assert validate_usage(usage) == usage
