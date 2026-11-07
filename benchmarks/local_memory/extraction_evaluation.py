"""Independent exact-span grading for event-to-proposal extraction studies.

These scores are annotation proxies, not semantic factual-precision estimates.
They never query retrieval and never treat a failed empty operation as abstention.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from mnemos.extraction.source_pipeline import SourcePipelineResult
from mnemos.extraction.source_selection import source_units
from mnemos.types import EpisodicEvent, MemoryChunk

ACCEPTED = frozenset({"add", "support", "correct"})
OPERATIONS = ACCEPTED | {"conflict", "unresolved"}


@dataclass(frozen=True)
class ExpectedOutcome:
    source_id: str
    start: int
    end: int
    kind: str
    operations: tuple[str, ...]
    target_fact_id: str | None = None
    expected_version: int | None = None
    critical: bool = False


@dataclass(frozen=True)
class ExtractionCase:
    case_id: str
    cluster_id: str
    events: tuple[EpisodicEvent, ...]
    targets: tuple[MemoryChunk, ...]
    expected: tuple[ExpectedOutcome, ...]
    expected_preflight_error: str | None = None


def validate_case(case: ExtractionCase) -> None:
    if not case.case_id or not case.cluster_id or not case.events:
        raise ValueError("An extraction case needs identity, cluster and source events")
    sources = {event.id: event for event in case.events}
    targets = {target.id: target for target in case.targets}
    if len(sources) != len(case.events) or len(targets) != len(case.targets):
        raise ValueError("Source and target identities must be unique")
    if case.expected_preflight_error is not None and case.expected:
        raise ValueError("Preflight guard cases cannot carry quality expectations")
    seen = set()
    for expected in case.expected:
        source = sources.get(expected.source_id)
        if source is None or not 0 <= expected.start < expected.end <= len(source.content):
            raise ValueError("Expected source span is missing or outside the event")
        units = source_units(source.content)
        if expected.start not in {unit.start for unit in units} or expected.end not in {
            unit.end for unit in units
        }:
            raise ValueError("Expected span must follow the frozen source-unit boundaries")
        identity = (expected.source_id, expected.start, expected.end)
        if identity in seen:
            raise ValueError("Repeated expected source span")
        seen.add(identity)
        if not expected.operations or not set(expected.operations) <= OPERATIONS:
            raise ValueError("Expected operation is unknown")
        if expected.critical and not set(expected.operations) <= ACCEPTED:
            raise ValueError("A critical fact must require an accepted outcome")
        if expected.target_fact_id is not None:
            target = targets.get(expected.target_fact_id)
            if target is None or target.version != expected.expected_version:
                raise ValueError("Expected target version is absent from the snapshot")
        elif expected.expected_version is not None:
            raise ValueError("Expected version needs a target")
        if set(expected.operations) & {"support", "correct"} and expected.target_fact_id is None:
            raise ValueError("Expected support or correction needs a target")


def _outcomes(
    case: ExtractionCase, result: SourcePipelineResult
) -> tuple[list[dict[str, Any]], int]:
    sources = {event.id: event for event in case.events}
    targets = {target.id: target for target in case.targets}
    observed: list[dict[str, Any]] = []
    proposal_index = review_index = 0
    invalid = 0
    for staged in result.reconciliations:
        record: dict[str, Any] = staged.model_dump()
        valid = 0 <= staged.candidate_index < len(result.candidates)
        if valid:
            candidate = result.candidates[staged.candidate_index]
            source = sources.get(candidate.source_id)
            valid = source is not None
            if source is not None:
                valid = (
                    candidate.source_id == staged.source_id == source.id
                    and candidate.source_role == staged.source_role == source.role
                    and candidate.kind == staged.source_kind
                    and candidate.source_fingerprint == source.event_fingerprint
                    and candidate.source_content_sha256
                    == hashlib.sha256(source.content.encode()).hexdigest()
                    and candidate.source_channel == source.source
                    and candidate.occurred_at == source.occurred_at.isoformat()
                    and candidate.profile_id == source.metadata.get("user_profile_id")
                    and (candidate.scope, candidate.scope_id) == (source.scope, source.scope_id)
                    and 0 <= candidate.start < candidate.end <= len(source.content)
                    and candidate.content
                    == staged.content
                    == source.content[candidate.start : candidate.end]
                    and candidate.revises_existing_requirement
                    == staged.revises_existing_requirement
                )
            record.update(start=candidate.start, end=candidate.end)
            if staged.target_fact_id is not None:
                target = targets.get(staged.target_fact_id)
                valid = bool(
                    valid
                    and target is not None
                    and (
                        target.version == staged.expected_version
                        and target.metadata.get("user_profile_id") == candidate.profile_id
                        and target.metadata.get("scope") == candidate.scope
                        and target.metadata.get("scope_id") == candidate.scope_id
                    )
                )
            elif staged.expected_version is not None or staged.operation in {"support", "correct"}:
                valid = False
        if staged.operation in ACCEPTED:
            if proposal_index >= len(result.proposals):
                valid = False
            else:
                proposal = result.proposals[proposal_index]
                valid = valid and (
                    proposal.content == staged.content
                    and proposal.evidence_ids == [staged.source_id]
                    and proposal.operation == staged.operation
                    and proposal.target_fact_id == staged.target_fact_id
                    and proposal.expected_version == staged.expected_version
                )
            proposal_index += 1
        else:
            if review_index >= len(result.reviews):
                valid = False
            else:
                review = result.reviews[review_index]
                valid = valid and all(
                    getattr(review, name) == getattr(staged, name)
                    for name in (
                        "candidate_index",
                        "source_id",
                        "source_role",
                        "source_kind",
                        "content",
                        "operation",
                        "target_fact_id",
                        "expected_version",
                        "revises_existing_requirement",
                    )
                )
            review_index += 1
        record["provenance_valid"] = bool(valid)
        invalid += not valid
        observed.append(record)
    invalid += max(0, len(result.proposals) - proposal_index)
    invalid += max(0, len(result.reviews) - review_index)
    return observed, invalid


def grade_extraction(
    case: ExtractionCase,
    result: SourcePipelineResult | None,
    *,
    error: str | None,
    provider_calls: int,
) -> dict[str, Any]:
    validate_case(case)
    if (result is None) == (error is None):
        raise ValueError("Supply exactly one result or error")
    observed, invalid = _outcomes(case, result) if result is not None else ([], 0)
    included = case.expected_preflight_error is None
    ok = included and result is not None and not invalid
    # Matching is one to one. The expected spans are unique, so duplicate output
    # cannot turn one reference fact into two retained facts.
    matched: dict[int, int] = {}
    contained: set[int] = set()
    for actual_index, actual in enumerate(observed):
        if not ok or not actual["provenance_valid"]:
            continue
        for index, expected in enumerate(case.expected):
            same_outcome = (
                actual["source_id"] == expected.source_id
                and actual["source_kind"] == expected.kind
                and actual["operation"] in expected.operations
                and actual["target_fact_id"] == expected.target_fact_id
                and actual["expected_version"] == expected.expected_version
            )
            if (
                same_outcome
                and expected.critical
                and actual["start"] <= expected.start
                and actual["end"] >= expected.end
            ):
                contained.add(index)
            if (
                same_outcome
                and actual["start"] == expected.start
                and actual["end"] == expected.end
                and index not in matched
            ):
                matched[index] = actual_index
                break
    accepted = len(result.proposals) if result is not None else 0
    exact_accepted = sum(observed[index]["operation"] in ACCEPTED for index in matched.values())
    return {
        "included_in_quality": included,
        "guard_passed": bool(
            not included
            and result is None
            and provider_calls == 0
            and case.expected_preflight_error is not None
            and case.expected_preflight_error in (error or "")
        ),
        "operational_success": bool(ok),
        "case_exact": bool(ok and len(matched) == len(case.expected) == len(observed)),
        "correct_abstention": bool(ok and not case.expected and not observed),
        "expected_outcomes": len(case.expected),
        "exact_expected_outcomes": len(matched),
        "critical_expected": sum(item.critical for item in case.expected),
        "critical_retained": sum(case.expected[index].critical for index in matched),
        "critical_span_contained": len(contained),
        "accepted_outcomes": accepted,
        "exact_supported_accepted": exact_accepted,
        "unmatched_accepted": accepted - exact_accepted,
        "review_outcomes": len(result.reviews) if result is not None else 0,
        "invalid_provenance_outcomes": invalid,
        "matched_expected_indices": sorted(matched),
        "observed_outcomes": observed,
    }
