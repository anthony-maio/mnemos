"""Exact duplicate user facts require review before selecting an identity."""

from dataclasses import replace

import pytest

from mnemos.extraction.judged_reconciliation import reconcile_source_candidates_with_judge
from mnemos.extraction.source_reconciliation import (
    build_reconciliation_schema,
    reconcile_source_candidates,
)
from tests.test_source_reconciliation import Provider, candidate, event, target
from mnemos.judgment import MockJudge

KINDS = [
    ("asserted_fact", False),
    ("user_preference", False),
    ("continuing_requirement", False),
    ("continuing_requirement", True),
    ("explicitly_retained_uncertainty", False),
    ("correction_or_rejection", False),
]


def inputs(kind="asserted_fact", revision=False, role="user"):
    text = "The marker is cedar."
    if kind == "explicitly_retained_uncertainty":
        text = "Remember that the marker might be cedar."
    source = event("duplicate-source", text, role)
    item = replace(
        candidate(source, kind, text if kind == "explicitly_retained_uncertainty" else None),
        revises_existing_requirement=revision,
    )
    targets = [
        target("first", text),
        target("second", "  " + text.upper() + "  "),
        target("unrelated", "The export format is JSON."),
    ]
    return source, item, targets


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,revision", KINDS)
async def test_duplicate_user_content_has_only_an_unresolved_branch(kind, revision) -> None:
    source, item, targets = inputs(kind, revision)
    schema = build_reconciliation_schema([item], targets)
    branches = schema["properties"]["decisions"]["items"]["oneOf"]
    assert len(branches) == 1
    assert branches[0]["properties"]["operation"]["enum"] == ["unresolved"]
    provider = Provider(
        {
            "decisions": [
                {
                    "candidate_index": 0,
                    "operation": "unresolved",
                    "reason": "Two observed fact identities match.",
                }
            ]
        }
    )
    result = await reconcile_source_candidates([item], [source], targets, provider)
    assert result[0].operation == "unresolved" and result[0].target_fact_id is None
    assert result[0].content == source.content


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize(
    "kind,revision,operation,target_id",
    [
        ("asserted_fact", False, "conflict", None),
        ("user_preference", False, "conflict", "unrelated"),
        ("continuing_requirement", True, "correct", "unrelated"),
        ("correction_or_rejection", False, "correct", "unrelated"),
    ],
)
async def test_provider_cannot_route_duplicate_user_content_around_review(
    kind, revision, operation, target_id, structured
) -> None:
    source, item, targets = inputs(kind, revision)
    provider = Provider(
        {
            "decisions": [
                {
                    "candidate_index": 0,
                    "operation": operation,
                    "target_fact_id": target_id,
                    "expected_version": 1 if target_id else None,
                    "reason": "A model-selected alternative.",
                }
            ]
        }
    )
    provider.supports_structured_output = structured
    with pytest.raises(ValueError, match="Duplicate exact user content requires unresolved"):
        await reconcile_source_candidates([item], [source], targets, provider)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["add", "conflict"])
async def test_attributed_history_keeps_its_separate_authority(operation) -> None:
    source, item, targets = inputs("attributed_history", role="assistant")
    provider = Provider(
        {
            "decisions": [
                {
                    "candidate_index": 0,
                    "operation": operation,
                    "reason": "Attribution remains separate from a direct user fact.",
                }
            ]
        }
    )
    result = await reconcile_source_candidates([item], [source], targets, provider)
    assert result[0].source_role == "assistant" and result[0].operation == operation


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["add", "conflict"])
async def test_unrelated_duplicate_targets_do_not_force_review(operation) -> None:
    _, _, targets = inputs()
    source = event("new-source", "The release channel is stable.")
    provider = Provider(
        {
            "decisions": [
                {
                    "candidate_index": 0,
                    "operation": operation,
                    "reason": "Candidate does not match the duplicate content.",
                }
            ]
        }
    )
    result = await reconcile_source_candidates([candidate(source)], [source], targets, provider)
    assert result[0].operation == operation


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,revision", KINDS)
async def test_judged_path_resolves_known_identity_ambiguity_without_a_model_call(
    kind, revision
) -> None:
    source, item, targets = inputs(kind, revision)
    judge = MockJudge({"referent:c0": "t2", "relation:c0:t2": "different_on_same_subject"})
    result = await reconcile_source_candidates_with_judge([item], [source], targets, judge)
    assert result.staged[0].operation == "unresolved"
    assert result.staged[0].reason_code == "multiple_matching_targets"
    assert judge.calls == []
