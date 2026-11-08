"""Development-only, in-memory reuse of fallible source assessments.

No provider, persistence, eligibility decisions, or production activation.
Callers supply a currently eligible snapshot and its live revision fence.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Literal

from mnemos.answer_support import SupportCandidate
from mnemos.support_intersection import SnapshotFence, SupportSnapshot

Disposition = Literal["supported", "unsupported", "unresolved"]


@dataclass(frozen=True)
class QualificationIdentity:
    policy_sha256: str
    model_sha256: str
    runtime_sha256: str

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("Qualification identity requires exact SHA-256 digests")


@dataclass(frozen=True)
class SourceAssessment:
    evidence_id: str
    disposition: Disposition
    cache_hit: bool


@dataclass(frozen=True)
class QualifiedCandidate:
    candidate: SupportCandidate
    assessments: tuple[SourceAssessment, ...]


def qualification_key(
    candidate: SupportCandidate, evidence_id: str, identity: QualificationIdentity
) -> str:
    if not isinstance(identity, QualificationIdentity):
        raise ValueError("Missing qualification identity")
    if (
        any(
            not isinstance(value, str) or not value.strip()
            for value in (candidate.fact_id, candidate.content, candidate.fingerprint)
        )
        or type(candidate.version) is not int
        or candidate.version < 1
        or candidate.scope not in {"project", "global", "session"}
        or (candidate.scope_id is not None and not isinstance(candidate.scope_id, str))
        or (candidate.scope != "global" and not candidate.scope_id)
    ):
        raise ValueError("Incomplete candidate identity")
    ids = [source.evidence_id for source in candidate.evidence]
    if len(set(ids)) != len(ids) or evidence_id not in ids:
        raise ValueError("Expected one uniquely identified own source")
    if any(
        any(
            not isinstance(value, str) or not value.strip()
            for value in (source.evidence_id, source.excerpt)
        )
        or any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in (source.source_event_id, source.source_ref)
        )
        for source in candidate.evidence
    ):
        raise ValueError("Incomplete source identity")
    # Bind all retained own sources conservatively. A changed sibling source can
    # invalidate an old assessment; it can never silently authorize that sibling.
    payload = {
        "schema": 1,
        "candidate": asdict(candidate),
        "evidence_id": evidence_id,
        "identity": asdict(identity),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


class QualificationCache:
    """Only assessed dispositions are reusable; missing entries stay unresolved.

    This cache cannot decide whether a source is true or a candidate is currently
    authorized. The caller's fence must include scope and forget eligibility.
    """

    def __init__(self) -> None:
        self._records: dict[str, Disposition] = {}

    def record(
        self,
        candidate: SupportCandidate,
        evidence_id: str,
        identity: QualificationIdentity,
        disposition: Disposition,
    ) -> None:
        if disposition not in ("supported", "unsupported", "unresolved"):
            raise ValueError("Unavailable or malformed assessments cannot be cached")
        key = qualification_key(candidate, evidence_id, identity)
        previous = self._records.get(key)
        if previous is not None and previous != disposition:
            raise ValueError("Conflicting assessment for a frozen qualification identity")
        self._records[key] = disposition

    def annotate(
        self,
        snapshot: SupportSnapshot,
        identity: QualificationIdentity,
        revalidate: SnapshotFence,
    ) -> tuple[QualifiedCandidate, ...]:
        revalidate()
        if len({c.fact_id for c in snapshot.candidates}) != len(snapshot.candidates):
            raise ValueError("Duplicate candidate identity")
        result = []
        for candidate in snapshot.candidates:
            assessments = []
            for source in candidate.evidence:
                key = qualification_key(candidate, source.evidence_id, identity)
                assessments.append(
                    SourceAssessment(
                        source.evidence_id,
                        self._records.get(key, "unresolved"),
                        key in self._records,
                    )
                )
            result.append(QualifiedCandidate(candidate, tuple(assessments)))
        revalidate()
        return tuple(result)
