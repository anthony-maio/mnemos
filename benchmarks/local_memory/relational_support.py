"""Development-only evaluation of source-bound, fallible relation annotations.

This module does not extract annotations or assert their semantic correctness.
It has no production retrieval integration. Callers own scope and snapshot fences.
"""

from dataclasses import dataclass
from datetime import date
from typing import Literal, Mapping


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str

    def validate(self, source: str) -> None:
        if (type(self.start) is not int or type(self.end) is not int
                or not 0 <= self.start < self.end <= len(source)
                or not isinstance(self.text, str) or not self.text.strip()
                or source[self.start:self.end] != self.text):
            raise ValueError("Annotation span does not match retained source")


@dataclass(frozen=True)
class Claim:
    fact_id: str
    evidence_id: str
    subject: str
    relation: str
    value: str
    subject_span: Span
    relation_span: Span
    value_span: Span
    # Half-open intervals. None means no *annotated* bound, not timeless truth.
    valid_from: date | None = None
    valid_until: date | None = None
    temporal_span: Span | None = None
    applicability: Literal["bounded", "unbounded", "unresolved"] = "unresolved"


@dataclass(frozen=True)
class AnnotationSet:
    snapshot_fingerprint: str
    policy_identity: str
    claims: tuple[Claim, ...]
    # The producer must explicitly finish every candidate, including those
    # with no extracted claim. Completeness is still a fallible assessment.
    assessed_fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class QueryFrame:
    subject: str
    relation: str
    at: date


@dataclass(frozen=True)
class Decision:
    status: Literal["selected", "absent", "conflict", "unavailable"]
    fact_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()


def evaluate(
    frame: QueryFrame,
    annotations: AnnotationSet,
    *,
    sources: Mapping[str, Mapping[str, str]],
    snapshot_fingerprint: str,
    policy_identity: str,
) -> Decision:
    """Apply interval and relation rules to a complete, exact current cohort.

    `sources` maps eligible fact IDs to their own evidence IDs and excerpts.
    Canonical subject/relation/value keys come from the producer, never this
    evaluator or fixture labels. Exact spans are provenance, not entailment.
    The caller must recheck the live snapshot after this function returns.
    """
    if (not snapshot_fingerprint or not policy_identity
            or annotations.snapshot_fingerprint != snapshot_fingerprint
            or annotations.policy_identity != policy_identity):
        return Decision("unavailable")
    if (len(set(annotations.assessed_fact_ids)) != len(annotations.assessed_fact_ids)
            or set(annotations.assessed_fact_ids) != set(sources)):
        return Decision("unavailable")
    if type(frame.at) is not date or not frame.subject.strip() or not frame.relation.strip():
        raise ValueError("A query frame requires subject, relation and exact date")
    matches = []
    unresolved = False
    for claim in annotations.claims:
        own = sources.get(claim.fact_id)
        if own is None or claim.evidence_id not in own:
            raise ValueError("Claim cites unavailable or foreign evidence")
        if any(not isinstance(key, str) or not key.strip()
               for key in (claim.subject, claim.relation, claim.value)):
            raise ValueError("Canonical claim keys must be nonblank")
        excerpt = own[claim.evidence_id]
        for span in (claim.subject_span, claim.relation_span, claim.value_span):
            span.validate(excerpt)
        for boundary in (claim.valid_from, claim.valid_until):
            if boundary is not None and type(boundary) is not date:
                raise ValueError("Temporal boundaries must be exact dates")
        if (claim.valid_from is not None and claim.valid_until is not None
                and claim.valid_from >= claim.valid_until):
            raise ValueError("Empty or reversed validity interval")
        if claim.valid_from is not None or claim.valid_until is not None:
            if claim.temporal_span is None:
                raise ValueError("A temporal boundary requires a source span")
        if claim.temporal_span is not None:
            claim.temporal_span.validate(excerpt)
        if claim.applicability not in {"bounded", "unbounded", "unresolved"}:
            raise ValueError("Unknown temporal applicability")
        has_bound = claim.valid_from is not None or claim.valid_until is not None
        if ((claim.applicability == "bounded" and not has_bound)
                or (claim.applicability == "unbounded" and has_bound)):
            raise ValueError("Applicability contradicts annotated boundaries")
        if claim.subject == frame.subject and claim.relation == frame.relation:
            if claim.applicability == "unresolved":
                unresolved = True
                continue
        if (claim.subject == frame.subject and claim.relation == frame.relation
                and (claim.valid_from is None or claim.valid_from <= frame.at)
                and (claim.valid_until is None or frame.at < claim.valid_until)):
            matches.append(claim)
    if unresolved:
        return Decision("unavailable")
    if not matches:
        return Decision("absent")
    if len({claim.value for claim in matches}) > 1:
        return Decision("conflict")
    return Decision("selected", tuple(dict.fromkeys(c.fact_id for c in matches)),
                    tuple(dict.fromkeys(c.evidence_id for c in matches)))
