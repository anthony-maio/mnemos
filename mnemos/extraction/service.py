"""Typed service adapter for opt-in source-bound extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from ..contracts import FactProposal
from ..memory_safety import MemoryWriteFirewall, bound_source_firewall
from ..source_review import SourceReviewReference
from ..types import EpisodicEvent, MemoryChunk
from ..utils.llm import LLMProvider
from .source_pipeline import run_source_pipeline
from .source_relationship import RelationshipProvider


@dataclass(frozen=True)
class SourceExtractionResult:
    proposals: list[FactProposal]
    reviews: list[SourceReviewReference]
    deferred_corrections: int
    oversized_candidates: int = 0


async def extract_source_bound(
    events: list[EpisodicEvent],
    targets: list[MemoryChunk],
    provider: LLMProvider,
    *,
    allow_corrections: bool,
    firewall: MemoryWriteFirewall | None = None,
) -> SourceExtractionResult:
    # Source checks must apply the policy capture used, or content stored
    # under a non-default policy can never pass validation.
    with bound_source_firewall(firewall or MemoryWriteFirewall()):
        return await _extract_source_bound(
            events, targets, provider, allow_corrections=allow_corrections
        )


async def _extract_source_bound(
    events: list[EpisodicEvent],
    targets: list[MemoryChunk],
    provider: LLMProvider,
    *,
    allow_corrections: bool,
) -> SourceExtractionResult:
    result = await run_source_pipeline(
        events,
        targets,
        selection_provider=provider,
        reconciliation_provider=RelationshipProvider(provider),
    )
    proposals = []
    reviews = list(result.review_references)
    accepted = [
        item for item in result.reconciliations if item.operation in {"add", "support", "correct"}
    ]
    if len(accepted) != len(result.proposals):
        raise ValueError("Source reconciliation proposal accounting changed")
    deferred = 0
    for staged, proposal in zip(accepted, result.proposals):
        if proposal.operation != "correct" or allow_corrections:
            proposals.append(proposal)
            continue
        # Disabling reconsolidation must not silently authorize a correction or
        # promote the conflicting new rule into an accepted fact.
        candidate = result.candidates[staged.candidate_index]
        review = SourceReviewReference(
            candidate_index=staged.candidate_index,
            source_id=candidate.source_id,
            source_fingerprint=candidate.source_fingerprint,
            source_content_sha256=candidate.source_content_sha256,
            source_role=candidate.source_role,
            source_channel=candidate.source_channel,
            occurred_at=candidate.occurred_at,
            user_profile_id=candidate.profile_id,
            scope=cast(Literal["project", "workspace", "global"], candidate.scope),
            scope_id=candidate.scope_id,
            start=candidate.start,
            end=candidate.end,
            source_kind=candidate.kind,
            operation="conflict",
            target_fact_id=staged.target_fact_id,
            expected_version=staged.expected_version,
            revises_existing_requirement=candidate.revises_existing_requirement,
        )
        review.resolve_source(
            next(event for event in events if event.id == review.source_id),
            target=next((target for target in targets if target.id == review.target_fact_id), None),
        )
        reviews.append(review)
        deferred += 1
    return SourceExtractionResult(
        proposals=proposals,
        reviews=reviews,
        deferred_corrections=deferred,
        oversized_candidates=result.oversized_candidates,
    )
