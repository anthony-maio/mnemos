"""Frozen synthetic recall fixture and grading for the native compact screen."""

from __future__ import annotations

import json
from typing import Any

PROFILE = "native-compact-v1"
PROJECT_ID = "native-compact-project"
MODEL = "gpt-6-astra"
FACTS = (
    "Cedar production retains logs for 30 days; staging retains logs for 7 days. "
    "This rule does not apply to exported archives.",
    "Cedar exported archives must remain encrypted and are deleted after 90 days. "
    "Do not apply the staging log rule to exports.",
    "Cedar production release approval requires two reviewers.",
    "Cedar production release approval requires three reviewers.",
    "Cedar staging deployment uses the blue queue. "
    "This statement does not specify the production deployment queue.",
)
FOREIGN = "Cedar FORBIDDEN_FOREIGN_PROJECT production uses the violet queue."
EXPECTED = {
    "production_log_days": 30,
    "staging_log_days": 7,
    "export_archive_days": 90,
    "export_archive_encrypted": True,
    "production_release_reviewers": None,
    "production_release_conflict": True,
    "staging_deployment_queue": "blue",
    "production_deployment_queue": None,
}
ORDER = ("full", "compact", "compact", "full", "full", "compact")


def arguments(response_format: str) -> dict[str, Any]:
    if response_format not in {"full", "compact"}:
        raise ValueError("Unknown frozen format")
    return {
        "query": "Cedar logs exported archives release approval deployment queue",
        "top_k": 5,
        "reconsolidate": False,
        "current_scope": "project",
        "scope_id": PROJECT_ID,
        "allowed_scopes": ["project"],
        "view": "memory",
        "response_format": response_format,
    }


def prompt(response_format: str) -> str:
    return (
        "Use mnemos_retrieve exactly once with these exact arguments: "
        + json.dumps(arguments(response_format), sort_keys=True)
        + ". Use only the returned memory as evidence. Do not call any other memory tool. "
        "Answer with one JSON object, with no markdown or commentary, using these keys: "
        "production_log_days, staging_log_days, export_archive_days, "
        "export_archive_encrypted, production_release_reviewers, "
        "production_release_conflict, staging_deployment_queue, production_deployment_queue. "
        "Day and reviewer counts are integers, encryption and conflict flags are booleans, "
        "and queue names are strings. Use null for a value that is unsupported or conflicted. "
        "Preserve the distinction between staging, production and exported archives."
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate answer key")
        result[key] = value
    return result


def grade_answer(text: str) -> dict[str, Any]:
    """Exact typed field agreement; malformed/extra output cannot pass."""
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
        if not isinstance(value, dict):
            raise ValueError("Answer is not an object")
    except (ValueError, TypeError):
        return {
            "valid_json_object": False,
            "correct_fields": 0,
            "total_fields": len(EXPECTED),
            "passed": False,
        }
    correct = {
        key: key in value and type(value[key]) is type(expected) and value[key] == expected
        for key, expected in EXPECTED.items()
    }
    return {
        "valid_json_object": True,
        "exact_keys": set(value) == set(EXPECTED),
        "fields": correct,
        "correct_fields": sum(correct.values()),
        "total_fields": len(EXPECTED),
        "passed": set(value) == set(EXPECTED) and all(correct.values()),
    }


def validate_usage(usage: Any) -> dict[str, int]:
    fields = ("input_tokens", "cached_input_tokens", "output_tokens")
    if not isinstance(usage, dict) or any(
        type(usage.get(k)) is not int or usage[k] < 0 for k in fields
    ):
        raise ValueError("Native terminal usage is incomplete")
    if usage["cached_input_tokens"] > usage["input_tokens"]:
        raise ValueError("Cached input exceeds total input")
    return {k: usage[k] for k in fields}
