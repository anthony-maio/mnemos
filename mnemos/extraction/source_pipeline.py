"""Write-free source classification and reconciliation orchestration.

This module composes the experimental contracts without crossing into storage or
production service behavior. Classification chooses provenance-bound excerpts;
reconciliation chooses guarded operations against an observed target snapshot.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .source_classification import ClassifiedInventory
from .source_instruction_ablation import classify_source_memories_without_uncertainty_example
from .source_reconciliation import StagedReconciliation, reconcile_source_candidates
from .source_selection import ExtractiveMemory, Kind
from mnemos.config import JudgmentConfig
from mnemos.contracts import FACT_CONTENT_LIMIT, FactProposal
from mnemos.judgment import Judge
from mnemos.source_review import SourceReviewReference
from mnemos.types import CognitiveState, EpisodicEvent, MemoryChunk
from mnemos.utils.llm import LLMProvider


class ReviewOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_index: int = Field(ge=0, strict=True)
    source_id: str = Field(min_length=1)
    source_role: str = Field(min_length=1)
    source_kind: Kind
    content: str = Field(min_length=1)
    operation: Literal["conflict", "unresolved"]
    target_fact_id: str | None = None
    expected_version: int | None = Field(default=None, ge=1, strict=True)
    reason: str = Field(min_length=1)
    revises_existing_requirement: bool = Field(default=False, strict=True)


class SourcePipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inventory: ClassifiedInventory
    candidates: list[ExtractiveMemory]
    reconciliations: list[StagedReconciliation]
    proposals: list[FactProposal]
    reviews: list[ReviewOutcome]
    review_references: list[SourceReviewReference] = Field(default_factory=list)
    oversized_candidates: int = Field(default=0, ge=0)
    # Judged path only: per-call accounting and the affective state judged
    # alongside classification, so the service need not ask again.
    judgment_calls: list[dict[str, Any]] = Field(default_factory=list)
    cognitive_state: CognitiveState | None = None


def _snapshot_events(events: list[EpisodicEvent]) -> list[EpisodicEvent]:
    return [event.model_copy(deep=True) for event in events]


def _snapshot_targets(targets: list[MemoryChunk]) -> list[MemoryChunk]:
    return [target.model_copy(deep=True) for target in targets]


def _validate_targets(events: list[EpisodicEvent], targets: list[MemoryChunk]) -> None:
    partitions = {
        (event.metadata.get("user_profile_id"), event.scope, event.scope_id) for event in events
    }
    seen: set[str] = set()
    for target in targets:
        if not target.id or target.id in seen:
            raise ValueError("Target snapshot contains a duplicate or blank ID")
        seen.add(target.id)
        partition = (
            target.metadata.get("user_profile_id"),
            target.metadata.get("scope"),
            target.metadata.get("scope_id"),
        )
        if partition not in partitions:
            raise ValueError("Target snapshot contains a foreign scope or profile")
        if target.version < 1:
            raise ValueError("Target snapshot contains an invalid version")


def _to_outcome(staged: StagedReconciliation) -> ReviewOutcome:
    if staged.operation not in {"conflict", "unresolved"} or not staged.reason:
        raise ValueError("Review outcomes require conflict or unresolved reasons")
    return ReviewOutcome(
        candidate_index=staged.candidate_index,
        source_id=staged.source_id,
        source_role=staged.source_role,
        source_kind=staged.source_kind,
        content=staged.content,
        operation=staged.operation,
        target_fact_id=staged.target_fact_id,
        expected_version=staged.expected_version,
        reason=staged.reason,
        revises_existing_requirement=staged.revises_existing_requirement,
    )


def _to_proposal(staged: StagedReconciliation, candidates: list[ExtractiveMemory]) -> FactProposal:
    candidate = candidates[staged.candidate_index]
    if staged.operation not in {"add", "support", "correct"}:
        raise ValueError("Only accepted reconciliation operations become proposals")
    if candidate.content != staged.content:
        raise ValueError("Reconciliation content is not the selected source content")
    return FactProposal(
        content=candidate.content,
        evidence_ids=[candidate.source_id],
        operation=staged.operation,
        target_fact_id=staged.target_fact_id,
        expected_version=staged.expected_version,
    )


async def run_source_pipeline(
    events: list[EpisodicEvent],
    targets: list[MemoryChunk],
    *,
    selection_provider: LLMProvider | None = None,
    reconciliation_provider: LLMProvider | None = None,
    judge: Judge | None = None,
    judgment_config: JudgmentConfig | None = None,
) -> SourcePipelineResult:
    """Run classification then reconciliation without writing any state.

    With ``judge`` set, both stages ask typed questions instead of prompting a
    generative model; the providers are then unused. Either way the same
    renderer and the same decision validator sit between the model and any
    proposal.
    """

    if judge is None and (selection_provider is None or reconciliation_provider is None):
        raise ValueError("run_source_pipeline needs a judge or both providers")
    original_events = _snapshot_events(events)
    original_targets = _snapshot_targets(targets)
    _validate_targets(original_events, original_targets)

    classified_events = _snapshot_events(original_events)
    judgment_calls: list[dict[str, Any]] = []
    cognitive_state: CognitiveState | None = None
    if judge is not None:
        from .judged_classification import classify_source_memories_with_judge

        judged = await classify_source_memories_with_judge(
            classified_events, judge, config=judgment_config, include_affective=True
        )
        inventory, candidates = judged.inventory, judged.memories
        judgment_calls.extend(judged.trace.calls)
        cognitive_state = judged.cognitive_state
    else:
        assert selection_provider is not None
        inventory, candidates = await classify_source_memories_without_uncertainty_example(
            classified_events, selection_provider
        )
    if events != original_events or targets != original_targets:
        raise ValueError("Pipeline inputs changed during classification")
    # A slice longer than any fact can hold can never become one. Leave it out
    # before reconciliation rather than failing every other candidate with it.
    fitting = [
        candidate for candidate in candidates if len(candidate.content) <= FACT_CONTENT_LIMIT
    ]
    oversized = len(candidates) - len(fitting)
    candidates = fitting

    if not candidates:
        return SourcePipelineResult(
            inventory=inventory,
            candidates=[],
            reconciliations=[],
            proposals=[],
            reviews=[],
            oversized_candidates=oversized,
            judgment_calls=judgment_calls,
            cognitive_state=cognitive_state,
        )

    reconciliation_events = _snapshot_events(original_events)
    reconciliation_targets = _snapshot_targets(original_targets)
    if judge is not None:
        from .judged_reconciliation import reconcile_source_candidates_with_judge

        judged_reconciliation = await reconcile_source_candidates_with_judge(
            candidates, reconciliation_events, reconciliation_targets, judge, config=judgment_config
        )
        staged = judged_reconciliation.staged
        judgment_calls.extend(judged_reconciliation.trace.calls)
    else:
        assert reconciliation_provider is not None
        staged = await reconcile_source_candidates(
            candidates,
            reconciliation_events,
            reconciliation_targets,
            reconciliation_provider,
        )
    if events != original_events or targets != original_targets:
        raise ValueError("Pipeline inputs changed during reconciliation")

    proposals = [
        _to_proposal(item, candidates)
        for item in staged
        if item.operation in {"add", "support", "correct"}
    ]
    reviews = [_to_outcome(item) for item in staged if item.operation in {"conflict", "unresolved"}]
    references = []
    for review in reviews:
        candidate = candidates[review.candidate_index]
        reference = SourceReviewReference(
            candidate_index=review.candidate_index,
            source_id=candidate.source_id,
            source_fingerprint=candidate.source_fingerprint,
            source_content_sha256=candidate.source_content_sha256,
            source_role=candidate.source_role,
            source_channel=candidate.source_channel,
            occurred_at=candidate.occurred_at,
            user_profile_id=candidate.profile_id,
            scope=candidate.scope,  # type: ignore[arg-type]
            scope_id=candidate.scope_id,
            start=candidate.start,
            end=candidate.end,
            source_kind=candidate.kind,
            operation=review.operation,
            target_fact_id=review.target_fact_id,
            expected_version=review.expected_version,
            revises_existing_requirement=candidate.revises_existing_requirement,
        )
        excerpt = reference.resolve_source(
            next(event for event in original_events if event.id == reference.source_id),
            target=next(
                (target for target in original_targets if target.id == reference.target_fact_id),
                None,
            ),
        )
        if excerpt != review.content:
            raise ValueError("Review content differs from its source reference")
        references.append(reference)
    return SourcePipelineResult(
        inventory=inventory,
        candidates=candidates,
        reconciliations=staged,
        proposals=proposals,
        reviews=reviews,
        review_references=references,
        oversized_candidates=oversized,
        judgment_calls=judgment_calls,
        cognitive_state=cognitive_state,
    )


__all__ = ["ReviewOutcome", "SourcePipelineResult", "run_source_pipeline"]
