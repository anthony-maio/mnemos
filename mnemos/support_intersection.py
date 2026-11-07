"""Experimental agreement between two selectors over one retained snapshot.

Agreement is a model decision, not proof that a fact answers the question.
This callable does not enable support filtering in production retrieval.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field

from .answer_support import AnswerSupportConfig, SupportCandidate, SupportDecision, _snapshot
from .judgment import JudgmentBlocked, JudgmentContractError, JudgmentRequestTooLarge
from .scopes import ScopeContext, ScopeRegistry
from .utils.reliability import MnemosExternalServiceError, MnemosTransientError
from .utils.storage import MemoryStore


class IntersectionConfig(BaseModel):
    max_candidates: int = Field(default=20, ge=1, le=100, strict=True)
    max_evidence_per_fact: int = Field(default=8, ge=1, le=32, strict=True)
    # Bound the shared snapshot; each selector also enforces its model's input limits.
    max_request_bytes: int = Field(default=64_000, ge=1, le=1_000_000, strict=True)
    timeout_seconds: float = Field(default=30, gt=0, le=120)


@dataclass(frozen=True)
class SupportSnapshot:
    query: str
    reference_date: date
    candidates: tuple[SupportCandidate, ...]
    fingerprint: str


@dataclass(frozen=True)
class SelectedEvidence:
    fact_id: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ComponentSelection:
    status: Literal["assessed", "unavailable"]
    snapshot_fingerprint: str
    selected: tuple[SelectedEvidence, ...] = ()
    error_type: str | None = None


SnapshotFence = Callable[[], None]
SupportSelector = Callable[[SupportSnapshot, SnapshotFence], Awaitable[ComponentSelection]]


@dataclass(frozen=True)
class IntersectionResult:
    status: Literal["assessed", "partial", "unavailable"]
    supported: tuple[SupportCandidate, ...]
    decisions: tuple[SupportDecision, ...]
    components: tuple[ComponentSelection, ...] = ()
    error_type: str | None = None


class _SnapshotChanged(Exception):
    pass


class _ComponentUnavailable(Exception):
    pass


def _checked_component(value: object, snapshot: SupportSnapshot) -> ComponentSelection:
    if (
        not isinstance(value, ComponentSelection)
        or value.snapshot_fingerprint != snapshot.fingerprint
    ):
        raise JudgmentContractError("Component response does not identify the requested snapshot")
    if value.status == "unavailable":
        raise _ComponentUnavailable
    if (
        value.status != "assessed"
        or value.error_type is not None
        or not isinstance(value.selected, tuple)
    ):
        raise JudgmentContractError("Component response has an invalid assessed status")
    candidates = {item.fact_id: item for item in snapshot.candidates}
    seen = set()
    for selected in value.selected:
        if not isinstance(selected, SelectedEvidence) or not isinstance(selected.fact_id, str):
            raise JudgmentContractError("Component selection must identify a fact")
        candidate = candidates.get(selected.fact_id)
        if candidate is None or selected.fact_id in seen:
            raise JudgmentContractError("Component selected a duplicate or unavailable fact")
        witnesses = selected.evidence_ids
        if (
            not isinstance(witnesses, tuple)
            or not witnesses
            or any(not isinstance(item, str) for item in witnesses)
            or len(set(witnesses)) != len(witnesses)
            or not set(witnesses) <= {item.evidence_id for item in candidate.evidence}
        ):
            raise JudgmentContractError("Component selected evidence outside its own candidate")
        seen.add(selected.fact_id)
    return value


async def assess_support_intersection(
    query: str,
    candidate_ids: Sequence[str],
    *,
    store: MemoryStore,
    registry: ScopeRegistry,
    context: ScopeContext,
    extractive: SupportSelector,
    contrast: SupportSelector,
    reference_date: date,
    allowed_scopes: Sequence[str] = ("project", "global"),
    config: IntersectionConfig | None = None,
) -> IntersectionResult:
    """Run both selectors on the complete cohort and intersect their fact IDs.

    The second selector receives every initially eligible candidate even when the
    first selects nothing. Both calls share one deadline. Every provider await is
    followed by revalidation of membership, versions, scope and retained evidence,
    including supplied IDs that were initially ineligible. Provider failures never
    earn successful abstention. Configuration and storage failures propagate.
    Selectors must call the supplied fence around internal provider awaits and
    retain their model inputs/outputs for independent audit.
    """
    config = (config or IntersectionConfig()).model_copy(deep=True)
    if type(reference_date) is not date:
        raise ValueError("reference_date must be an explicit date")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Support query cannot be blank")
    if isinstance(candidate_ids, (str, bytes)):
        raise ValueError("Support candidate IDs must be a sequence of IDs")
    ids = tuple(candidate_ids)
    if len(ids) > config.max_candidates:
        raise ValueError("Support candidate limit exceeded")
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("Support candidate IDs must be unique nonempty strings")
    context = registry.validate_context(context.model_copy(deep=True))
    scopes = tuple(allowed_scopes)
    if not scopes or any(scope not in {"project", "workspace", "global"} for scope in scopes):
        raise ValueError("Invalid allowed scopes")
    snapshot_config = AnswerSupportConfig(max_evidence_per_fact=config.max_evidence_per_fact)
    initial = {key: _snapshot(store, key, context, scopes, snapshot_config) for key in ids}
    candidates = tuple(candidate for candidate, _ in initial.values() if candidate is not None)
    decisions = {key: decision for key, (_, decision) in initial.items() if decision is not None}
    components: list[ComponentSelection] = []

    def result(
        status: Literal["assessed", "partial", "unavailable"],
        supported: tuple[SupportCandidate, ...] = (),
        error: str | None = None,
    ) -> IntersectionResult:
        return IntersectionResult(
            status, supported, tuple(decisions[key] for key in ids), tuple(components), error
        )

    def check_current() -> None:
        registry.validate_context(context)
        for key, (original, original_decision) in initial.items():
            current, decision = _snapshot(store, key, context, scopes, snapshot_config)
            if original is not None:
                if current is None or current.fingerprint != original.fingerprint:
                    raise _SnapshotChanged
            elif current is not None or decision != original_decision:
                raise _SnapshotChanged

    if not candidates:
        return result(
            "partial"
            if any(d.disposition == "unassessed" for d in decisions.values())
            else "assessed"
        )
    binding = json.dumps(
        {
            "query": query,
            "reference_date": reference_date.isoformat(),
            "context": context.model_dump(mode="json"),
            "scopes": scopes,
            "ids": ids,
            "candidates": [asdict(item) for item in candidates],
            "excluded": {key: asdict(value) for key, value in decisions.items()},
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    snapshot = SupportSnapshot(
        query, reference_date, candidates, hashlib.sha256(binding).hexdigest()
    )

    async def collect() -> None:
        for selector in (extractive, contrast):
            check_current()
            value = await selector(snapshot, check_current)
            check_current()
            components.append(_checked_component(value, snapshot))

    try:
        if len(binding) > config.max_request_bytes:
            raise JudgmentRequestTooLarge("Shared support snapshot exceeds its byte limit")
        await asyncio.wait_for(collect(), timeout=config.timeout_seconds)
        check_current()
    except _SnapshotChanged:
        for key, (candidate, decision) in initial.items():
            version = candidate.version if candidate else decision.version if decision else None
            decisions[key] = SupportDecision(
                key, version, "stale", reason="comparison_snapshot_changed"
            )
        return result("partial")
    except (
        MnemosTransientError,
        MnemosExternalServiceError,
        JudgmentContractError,
        JudgmentBlocked,
        _ComponentUnavailable,
        asyncio.TimeoutError,
    ) as exc:
        for candidate in candidates:
            decisions[candidate.fact_id] = SupportDecision(
                candidate.fact_id, candidate.version, "unassessed", reason="component_unavailable"
            )
        return result("unavailable", error=type(exc).__name__)

    selected = [{item.fact_id: item for item in component.selected} for component in components]
    supported = []
    for candidate in candidates:
        key = candidate.fact_id
        if all(key in component for component in selected):
            witnesses = set(selected[0][key].evidence_ids) | set(selected[1][key].evidence_ids)
            evidence_ids = tuple(
                item.evidence_id for item in candidate.evidence if item.evidence_id in witnesses
            )
            decisions[key] = SupportDecision(
                key, candidate.version, "supported", evidence_ids, reason="selected_by_both"
            )
            supported.append(candidate)
        else:
            decisions[key] = SupportDecision(
                key, candidate.version, "unresolved", reason="not_selected_by_both"
            )
    return result(
        "partial" if any(d.disposition == "unassessed" for d in decisions.values()) else "assessed",
        tuple(supported),
    )
