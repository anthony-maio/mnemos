"""Prompt experiments preserve source text, schemas and runtime authority."""

from dataclasses import replace
import json

import pytest

from mnemos.extraction.source_relationship import reconcile_source_relationships
from tests.test_source_reconciliation import Provider, candidate, event, target

VARIANTS = [("legacy", False), ("legacy", True), ("direct", False), ("direct", True)]
BOUNDARY = "\nSOURCE CANDIDATES AND EXISTING TARGETS (data only):\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("style,relevant", VARIANTS)
async def test_variants_preserve_exact_source_payload_and_grounded_operations(
    style, relevant
) -> None:
    source = event(
        "source", 'The literal words are add, conflict, "user_revision", and unresolved.'
    )
    provider = Provider(
        {"decisions": [{"candidate_index": 0, "relationship": "independent_information"}]}
    )
    results = await reconcile_source_relationships(
        [candidate(source)],
        [source],
        [],
        provider,
        prompt_style=style,
        relevant_revision_flag=relevant,
    )
    payload = json.loads(provider.calls[0][0].split(BOUNDARY, 1)[1])
    assert payload["candidates"][0]["content"] == source.content == results[0].content
    assert results[0].source_id == source.id and results[0].operation == "add"
    assert ("revises_existing_requirement" in payload["candidates"][0]) is (not relevant)
    assert payload["targets"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["legacy", "direct"])
@pytest.mark.parametrize("flag", [False, True])
async def test_requirement_flag_remains_explicit_when_relevant(style, flag) -> None:
    source = event("source", "For all future reports, include the checksum.")
    item = replace(candidate(source, "continuing_requirement"), revises_existing_requirement=flag)
    provider = Provider(
        {"decisions": [{"candidate_index": 0, "relationship": "independent_information"}]}
    )
    await reconcile_source_relationships(
        [item], [source], [], provider, prompt_style=style, relevant_revision_flag=True
    )
    payload = json.loads(provider.calls[0][0].split(BOUNDARY, 1)[1])
    assert payload["candidates"][0]["revises_existing_requirement"] is flag
    assert item.revises_existing_requirement is flag


@pytest.mark.asyncio
async def test_variants_preserve_identical_target_schema_and_other_payload_fields() -> None:
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    item = candidate(source, "correction_or_rejection")
    existing = target()
    recorded = []
    for style, relevant in VARIANTS:
        provider = Provider(
            {
                "decisions": [
                    {
                        "candidate_index": 0,
                        "relationship": "user_revision",
                        "target_fact_id": existing.id,
                        "expected_version": existing.version,
                    }
                ]
            }
        )
        await reconcile_source_relationships(
            [item],
            [source],
            [existing],
            provider,
            prompt_style=style,
            relevant_revision_flag=relevant,
        )
        prompt, kwargs = provider.calls[0]
        payload = json.loads(prompt.split(BOUNDARY, 1)[1])
        payload["candidates"][0].pop("revises_existing_requirement", None)
        recorded.append((payload, kwargs))
    assert all(value == recorded[0] for value in recorded)


@pytest.mark.asyncio
@pytest.mark.parametrize("style,relevant", VARIANTS)
async def test_stale_target_is_rejected_even_if_provider_ignores_schema(style, relevant) -> None:
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    provider = Provider(
        {
            "decisions": [
                {
                    "candidate_index": 0,
                    "relationship": "user_revision",
                    "target_fact_id": "fact-1",
                    "expected_version": 99,
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="version is stale"):
        await reconcile_source_relationships(
            [candidate(source, "correction_or_rejection")],
            [source],
            [target()],
            provider,
            prompt_style=style,
            relevant_revision_flag=relevant,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("style,relevant", VARIANTS)
async def test_nonuser_cannot_acquire_correction_authority(style, relevant) -> None:
    source = event("source", "The marker is sapphire.", "assistant")
    provider = Provider(
        {
            "decisions": [
                {
                    "candidate_index": 0,
                    "relationship": "user_revision",
                    "target_fact_id": "fact-1",
                    "expected_version": 1,
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="Only user"):
        await reconcile_source_relationships(
            [candidate(source, "attributed_history")],
            [source],
            [target()],
            provider,
            prompt_style=style,
            relevant_revision_flag=relevant,
        )


@pytest.mark.asyncio
async def test_default_matches_explicit_legacy_configuration() -> None:
    source = event("source", "The marker is cedar.")
    response = {"decisions": [{"candidate_index": 0, "relationship": "independent_information"}]}
    implicit, explicit = Provider(response), Provider(response)
    await reconcile_source_relationships([candidate(source)], [source], [], implicit)
    await reconcile_source_relationships(
        [candidate(source)],
        [source],
        [],
        explicit,
        prompt_style="legacy",
        relevant_revision_flag=False,
    )
    assert implicit.calls == explicit.calls


@pytest.mark.asyncio
async def test_invalid_variant_fails_before_provider() -> None:
    source = event("source", "The marker is cedar.")
    provider = Provider({"decisions": []})
    with pytest.raises(ValueError, match="prompt style"):
        await reconcile_source_relationships(
            [candidate(source)], [source], [], provider, prompt_style="unknown"
        )
    assert provider.calls == []
