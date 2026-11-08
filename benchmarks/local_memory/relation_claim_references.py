"""Explicit references for fallible inventory claims, bound to source revisions."""

from dataclasses import asdict, dataclass
import hashlib
import json

from .relation_inventory_cache_v2 import AnnotatedCandidate, SourceInventory, SourceRelation, validate_inventory
from .source_qualification import QualificationIdentity, qualification_key


@dataclass(frozen=True)
class ReferencedClaim:
    claim_ref: str
    fact_id: str
    evidence_id: str
    claim_index: int
    claim: SourceRelation


def reference_claims(cohort: tuple[AnnotatedCandidate, ...], identity: QualificationIdentity
                     ) -> tuple[ReferencedClaim, ...]:
    """Preserve cohort/source/claim order; never infer scope or reintroduce facts.

    Missing inventories remain the caller's unavailable work. References do not
    turn their absence into an empty, successfully assessed source inventory.
    """
    if not isinstance(identity, QualificationIdentity):
        raise ValueError('Source identity required')
    if len({row.candidate.fact_id for row in cohort}) != len(cohort):
        raise ValueError('Duplicate fact identity')
    result = []
    for row in cohort:
        if [v.evidence_id for v in row.inventories] != [e.evidence_id for e in row.candidate.evidence]:
            raise ValueError('Inventory coverage differs from original own sources')
        for inventory in row.inventories:
            source_key = qualification_key(row.candidate, inventory.evidence_id, identity)
            if inventory.status == 'missing':
                if inventory.claims:
                    raise ValueError('Missing inventory cannot carry claims')
                continue
            annotation = SourceInventory(inventory.status, inventory.claims)
            validate_inventory(row.candidate, inventory.evidence_id, annotation)
            inventory_digest = hashlib.sha256(json.dumps(asdict(annotation), sort_keys=True,
                ensure_ascii=False, separators=(',', ':'), default=str).encode('utf-8')).hexdigest()
            for index, claim in enumerate(inventory.claims):
                payload = dict(schema=1, source_key=source_key, inventory_sha256=inventory_digest, index=index)
                digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                    separators=(',', ':'), default=str).encode('utf-8')).hexdigest()
                result.append(ReferencedClaim('claim_' + digest, row.candidate.fact_id,
                                              inventory.evidence_id, index, claim))
    if len({r.claim_ref for r in result}) != len(result):
        raise ValueError('Duplicate claim reference')
    return tuple(result)


def resolve_complete_reply(expected: tuple[ReferencedClaim, ...], reply_refs: tuple[str, ...]
                           ) -> tuple[ReferencedClaim, ...]:
    """Resolve in response order, requiring exactly the complete supplied set."""
    by_ref = {r.claim_ref: r for r in expected}
    if (len(by_ref) != len(expected) or len(set(reply_refs)) != len(reply_refs)
            or set(reply_refs) != set(by_ref)):
        raise ValueError('Reply must cover every current claim reference exactly once')
    return tuple(by_ref[reference] for reference in reply_refs)
