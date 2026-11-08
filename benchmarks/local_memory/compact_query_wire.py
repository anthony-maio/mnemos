"""Compact model output with full evidence binding retained by the trusted owner.

This module neither calls a provider nor decides semantic support. The transport
owner must attach reply identity from the actual request, never from model text.
"""

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
from uuid import uuid4

from mnemos.support_intersection import SnapshotFence, SupportSnapshot
from .relation_claim_references import ReferencedClaim, reference_claims
from .relation_inventory_cache_v2 import AnnotatedCandidate
from .source_qualification import QualificationIdentity
from .temporal_support import RequestedTime


@dataclass(frozen=True)
class CompactRequest:
    request_id: str
    snapshot: SupportSnapshot
    source_identity: QualificationIdentity
    query_identity: QualificationIdentity
    cohort: tuple[AnnotatedCandidate, ...]
    aliases: tuple[tuple[str, ReferencedClaim], ...]
    model_input: str
    binding_sha256: str


@dataclass(frozen=True)
class TransportReply:
    request_id: str
    binding_sha256: str
    completed: bool
    text: str


@dataclass(frozen=True)
class CompactInterpretation:
    claim: ReferencedClaim
    relation_match: str
    condition_match: str


@dataclass(frozen=True)
class DecodedQuery:
    requested_time: RequestedTime
    interpretations: tuple[CompactInterpretation, ...]


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), default=str).encode('utf-8')).hexdigest()


def _binding(request: CompactRequest) -> str:
    payload = asdict(request)
    del payload['binding_sha256']
    return _digest(payload)


def prepare_compact(snapshot: SupportSnapshot, cohort: tuple[AnnotatedCandidate, ...],
                    source_identity: QualificationIdentity, query_identity: QualificationIdentity,
                    fence: SnapshotFence) -> CompactRequest:
    fence()
    if not isinstance(source_identity, QualificationIdentity) or not isinstance(query_identity, QualificationIdentity):
        raise ValueError('Exact source and query identities required')
    if (not snapshot.query.strip() or not snapshot.fingerprint or type(snapshot.reference_date) is not date
            or tuple(row.candidate for row in cohort) != snapshot.candidates):
        raise ValueError('Cohort must match the complete current snapshot')
    if len(cohort) > 20 or any(len(row.inventories) > 8 for row in cohort):
        raise ValueError('Cohort exceeds bounded contract')
    if any(not row.inventories or any(inv.status != 'complete' for inv in row.inventories) for row in cohort):
        raise ValueError('Missing or unresolved inventories remain unavailable work')
    references = reference_claims(cohort, source_identity)
    aliases = tuple((f'c{i}', reference) for i, reference in enumerate(references))
    facts = []
    claims = {}
    for i, row in enumerate(cohort):
        fact_label = f'f{i}'
        own = {e.evidence_id: f's{j}' for j, e in enumerate(row.candidate.evidence)}
        facts.append(dict(id=fact_label, fact=row.candidate.content,
                          sources={own[e.evidence_id]: e.excerpt for e in row.candidate.evidence}))
        for alias, reference in aliases:
            if reference.fact_id == row.candidate.fact_id:
                claims[alias] = dict(fact=fact_label, source=own[reference.evidence_id],
                                     annotation=asdict(reference.claim))
    # Evidence text and annotations stay complete. Only audit/identity metadata
    # moves outside the model's input and generated response.
    model_input = json.dumps(dict(query=snapshot.query, reference_date=snapshot.reference_date.isoformat(),
                                 facts=facts, claims=claims), ensure_ascii=False, separators=(',', ':'), default=str)
    if len(model_input.encode('utf-8')) > 64000:
        raise ValueError('Compact input exceeds byte limit; no truncation')
    draft = CompactRequest(uuid4().hex, snapshot, source_identity, query_identity, cohort, aliases, model_input, '')
    request = CompactRequest(draft.request_id, snapshot, source_identity, query_identity, cohort,
                             aliases, model_input, _binding(draft))
    fence()
    return request


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def decode_compact(request: CompactRequest, reply: TransportReply, fence: SnapshotFence) -> DecodedQuery:
    """Decode only; callers must apply full support/time/conflict decision rules.

    t: S=explicit source-state question, D=requested date, U=unresolved intent.
    d: ISO date for D, null otherwise. c maps every supplied alias to two letters:
    relation M=matches/X=excluded/U=unresolved; condition A=applies/X=excluded/U=unresolved.
    """
    fence()
    if (_binding(request) != request.binding_sha256 or reply.request_id != request.request_id
            or reply.binding_sha256 != request.binding_sha256 or reply.completed is not True):
        raise ValueError('Reply is not bound to this complete request')
    if not isinstance(reply.text, str) or len(reply.text.encode('utf-8')) > 64000:
        raise ValueError('Reply exceeds byte contract')
    payload = json.loads(reply.text, object_pairs_hook=_unique_object)
    if not isinstance(payload, dict) or set(payload) != {'t', 'd', 'c'}:
        raise ValueError('Unexpected compact response fields')
    if not isinstance(payload['t'], str):
        raise ValueError('Temporal intent must be a code string')
    intent = {'S': 'source_state', 'D': 'at_date', 'U': 'unresolved'}.get(payload['t'])
    if intent is None:
        raise ValueError('Unknown temporal intent')
    at = None
    if intent == 'at_date':
        if not isinstance(payload['d'], str):
            raise ValueError('Requested date must be an ISO date string')
        at = date.fromisoformat(payload['d'])
        if at.isoformat() != payload['d']:
            raise ValueError('Requested date must use YYYY-MM-DD')
    elif payload['d'] is not None:
        raise ValueError('Non-dated intent cannot carry a date')
    codes = payload['c']
    if not isinstance(codes, dict) or set(codes) != {alias for alias, _ in request.aliases}:
        raise ValueError('Reply must cover the complete alias set')
    interpretations = []
    for alias, reference in request.aliases:
        code = codes[alias]
        if not isinstance(code, str) or len(code) != 2 or code[0] not in 'MXU' or code[1] not in 'AXU':
            raise ValueError('Unknown relation/condition code')
        interpretations.append(CompactInterpretation(reference,
            {'M': 'matches', 'X': 'excluded', 'U': 'unresolved'}[code[0]],
            {'A': 'applies', 'X': 'excluded', 'U': 'unresolved'}[code[1]]))
    result = DecodedQuery(RequestedTime(intent, at), tuple(interpretations))
    fence()
    return result
