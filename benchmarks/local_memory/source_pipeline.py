"""Write-free source classification and reconciliation orchestration.

This module composes the experimental contracts without crossing into storage or
production service behavior. Classification chooses provenance-bound excerpts;
reconciliation chooses guarded operations against an observed target snapshot.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .source_classification import ClassifiedInventory
from .source_instruction_ablation import classify_source_memories_without_uncertainty_example
from .source_reconciliation import StagedReconciliation, reconcile_source_candidates
from .source_selection import ExtractiveMemory, Kind
from mnemos.contracts import FactProposal
from mnemos.types import EpisodicEvent, MemoryChunk
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
    selection_provider: LLMProvider,
    reconciliation_provider: LLMProvider,
) -> SourcePipelineResult:
    """Run classification then reconciliation without writing any state."""

    original_events = _snapshot_events(events)
    original_targets = _snapshot_targets(targets)
    _validate_targets(original_events, original_targets)

    classified_events = _snapshot_events(original_events)
    inventory, candidates = await classify_source_memories_without_uncertainty_example(
        classified_events, selection_provider
    )
    if events != original_events or targets != original_targets:
        raise ValueError("Pipeline inputs changed during classification")

    if not candidates:
        return SourcePipelineResult(
            inventory=inventory,
            candidates=[],
            reconciliations=[],
            proposals=[],
            reviews=[],
        )

    reconciliation_events = _snapshot_events(original_events)
    reconciliation_targets = _snapshot_targets(original_targets)
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
    return SourcePipelineResult(
        inventory=inventory,
        candidates=candidates,
        reconciliations=staged,
        proposals=proposals,
        reviews=reviews,
    )


__all__ = ["ReviewOutcome", "SourcePipelineResult", "run_source_pipeline"]
