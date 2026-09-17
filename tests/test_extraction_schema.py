"""Exercise each decoder alternative without relying on sibling constraints."""

import pytest
from jsonschema import Draft202012Validator

from mnemos.contracts import ExtractionResponse


@pytest.mark.parametrize("operation", ["add", "support", "correct", "conflict"])
def test_standalone_decoder_branch_requires_grounded_fact(operation: str) -> None:
    schema = ExtractionResponse.model_json_schema()
    branch = next(
        item
        for item in schema["$defs"]["FactProposal"]["oneOf"]
        if item["properties"]["operation"]["enum"] == [operation]
    )
    Draft202012Validator.check_schema(branch)
    validator = Draft202012Validator(branch)
    fact = {"operation": operation, "content": "The cache uses Redis.", "evidence_ids": ["e1"]}
    if operation in {"support", "correct"}:
        fact["target_fact_id"] = "fact-1"
    if operation == "correct":
        fact["expected_version"] = 1
    assert validator.is_valid(fact)
    # This is the shape emitted repeatedly by the real failed development run.
    assert not validator.is_valid({"operation": operation})
    for field in ("content", "evidence_ids"):
        assert not validator.is_valid({key: value for key, value in fact.items() if key != field})
    assert not validator.is_valid({**fact, "content": ""})
    assert not validator.is_valid({**fact, "evidence_ids": []})
    assert not validator.is_valid({**fact, "unexpected": True})
    if operation in {"support", "correct"}:
        assert not validator.is_valid({**fact, "target_fact_id": None})
        assert not validator.is_valid({**fact, "target_fact_id": ""})
    if operation == "correct":
        assert not validator.is_valid({**fact, "expected_version": None})
        assert not validator.is_valid({**fact, "expected_version": 0})
    # The emitted full schema and runtime validator agree on the valid example.
    response = {"decision": "facts", "facts": [fact]}
    Draft202012Validator(schema).validate(response)
    ExtractionResponse.model_validate(response)
