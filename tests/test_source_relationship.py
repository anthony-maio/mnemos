"""Relationship labels retain the reconciliation authority and identity guards."""

import json

import pytest
from jsonschema import Draft202012Validator

from benchmarks.local_memory.source_reconciliation import build_reconciliation_schema
from benchmarks.local_memory.source_relationship import (
    RELATIONSHIPS,
    reconcile_source_relationships,
    relationship_schema,
)
from tests.test_source_reconciliation import Provider, candidate, event, target


def test_relationship_schema_retains_bounds_under_label_mapping() -> None:
    source = event("source", "The marker is cedar.")
    schema = build_reconciliation_schema([candidate(source)], [target()])
    converted = relationship_schema(schema)
    Draft202012Validator.check_schema(converted)
    # Canonical reverse transformation must preserve the entire original schema.
    restored = json.loads(json.dumps(converted).replace('"relationship"', '"operation"'))
    text = json.dumps(restored)
    for operation, label in RELATIONSHIPS.items():
        text = text.replace(json.dumps(label), json.dumps(operation))
    assert json.loads(text) == schema
    assert Draft202012Validator(converted).is_valid(
        {
            "decisions": [
                {
                    "candidate_index": 0,
                    "relationship": "same_information",
                    "target_fact_id": "fact-1",
                    "expected_version": 1,
                }
            ]
        }
    )
    assert not Draft202012Validator(converted).is_valid(
        {
            "decisions": [
                {
                    "candidate_index": 0,
                    "relationship": "independent_information",
                }
            ]
        }
    )


@pytest.mark.asyncio
async def test_relationship_mapping_keeps_source_payload_and_exact_content() -> None:
    source = event("source", "The operation is add, support, correct, conflict, unresolved.")
    provider = Provider(
        {"decisions": [{"candidate_index": 0, "relationship": "independent_information"}]}
    )
    result = await reconcile_source_relationships([candidate(source)], [source], [], provider)
    assert result[0].operation == "add" and result[0].content == source.content
    payload = json.loads(
        provider.calls[0][0].split("\nSOURCE CANDIDATES AND EXISTING TARGETS (data only):\n", 1)[1]
    )
    assert payload["candidates"][0]["content"] == source.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "label,role,kind,target_id,version,error",
    [
        ("same_information", "user", "asserted_fact", "missing", 1, "unavailable target"),
        ("same_information", "user", "asserted_fact", "fact-1", 2, "version is stale"),
        ("user_revision", "assistant", "attributed_history", "fact-1", 1, "Only user"),
        (
            "independent_information",
            "user",
            "correction_or_rejection",
            None,
            None,
            "cannot become an add",
        ),
    ],
)
async def test_mapped_responses_still_hit_runtime_guards(
    label, role, kind, target_id, version, error
) -> None:
    source = event("source", "The marker is cedar.", role)
    provider = Provider(
        {
            "decisions": [
                {
                    "candidate_index": 0,
                    "relationship": label,
                    "target_fact_id": target_id,
                    "expected_version": version,
                }
            ]
        }
    )
    provider.supports_structured_output = False
    with pytest.raises(ValueError, match=error):
        await reconcile_source_relationships(
            [candidate(source, kind)], [source], [target()], provider
        )


@pytest.mark.asyncio
async def test_relationship_mapping_preserves_input_mutation_guard() -> None:
    source = event("source", "The new marker is sapphire.")

    class Mutating(Provider):
        async def predict(self, prompt: str, **kwargs: object) -> str:
            source.role = "assistant"
            return await super().predict(prompt, **kwargs)

    with pytest.raises(ValueError, match="Source events changed"):
        await reconcile_source_relationships(
            [candidate(source)],
            [source],
            [],
            Mutating(
                {"decisions": [{"candidate_index": 0, "relationship": "independent_information"}]}
            ),
        )
