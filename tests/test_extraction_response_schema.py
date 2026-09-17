"""Complete decision alternatives for structured extraction responses."""

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from mnemos.contracts import ExtractionResponse

_FACT = {
    "content": "The user uses SQLite.",
    "evidence_ids": ["source-1"],
    "operation": "add",
}


def _schema() -> dict[str, object]:
    schema = ExtractionResponse.model_json_schema()
    assert isinstance(schema.get("oneOf"), list)
    return schema


def test_decision_schema_has_complete_non_overlapping_alternatives() -> None:
    schema = _schema()
    branches = schema["oneOf"]
    assert len(branches) == 2

    facts_branch = next(
        branch for branch in branches if branch["properties"]["decision"]["enum"] == ["facts"]
    )
    no_facts_branch = next(
        branch
        for branch in branches
        if branch["properties"]["decision"]["enum"] == ["no_durable_facts"]
    )

    assert facts_branch["properties"]["facts"]["minItems"] == 1
    assert no_facts_branch["properties"]["facts"]["maxItems"] == 0
    assert no_facts_branch["required"] == ["decision", "facts", "reason"]
    assert r"\S" in no_facts_branch["properties"]["reason"]["pattern"]

    validator = Draft202012Validator(schema)
    assert validator.is_valid({"decision": "facts", "facts": [_FACT]})
    assert validator.is_valid(
        {"decision": "no_durable_facts", "facts": [], "reason": "No assertion."}
    )
    assert not validator.is_valid({"decision": "facts", "facts": []})
    assert not validator.is_valid({"decision": "no_durable_facts", "facts": []})
    assert not validator.is_valid({"decision": "no_durable_facts", "facts": [], "reason": "   "})


@pytest.mark.parametrize(
    "payload",
    [
        {"decision": "facts", "facts": []},
        {"decision": "no_durable_facts", "facts": []},
        {"decision": "no_durable_facts", "facts": [], "reason": "   "},
        {"decision": "no_durable_facts", "facts": [_FACT], "reason": "unsupported"},
    ],
)
def test_runtime_rejects_invalid_decision_payloads(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ExtractionResponse.model_validate(payload)


def test_runtime_accepts_valid_facts_and_no_facts_payloads() -> None:
    facts = ExtractionResponse.model_validate({"decision": "facts", "facts": [_FACT]})
    no_facts = ExtractionResponse.model_validate(
        {"decision": "no_durable_facts", "facts": [], "reason": "No assertion."}
    )
    assert facts.decision == "facts"
    assert no_facts.decision == "no_durable_facts"
