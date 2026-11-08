"""Bounded, development-only reuse of query-independent source annotations.

No provider, persistence, live authorization or query-time applicability decision.
Cached annotations are fallible assessments, never authoritative new facts.
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import date
import json
from typing import Literal

from mnemos.answer_support import SupportCandidate
from mnemos.support_intersection import SnapshotFence, SupportSnapshot
from .source_qualification import QualificationIdentity, qualification_key
from .source_quote_binding import bind_own_source_quotes, bind_unique_quote


@dataclass(frozen=True)
class SourceRelation:
    subject: str
    relation: str
    value: str
    subject_quote: str
    relation_quote: str
    value_quote: str
    whole_fact_quote: str | None
    temporal_quote: str | None
    condition_quotes: tuple[str, ...]
    valid_from: date | None
    valid_until: date | None
    applicability: Literal['bounded', 'unbounded', 'unresolved']
    whole_fact_support: Literal['supported', 'unsupported', 'unresolved']


@dataclass(frozen=True)
class SourceInventory:
    status: Literal['complete', 'unresolved']
    claims: tuple[SourceRelation, ...]


@dataclass(frozen=True)
class InventoryView:
    evidence_id: str
    status: Literal['missing', 'complete', 'unresolved']
    claims: tuple[SourceRelation, ...]
    cache_hit: bool


@dataclass(frozen=True)
class AnnotatedCandidate:
    candidate: SupportCandidate
    inventories: tuple[InventoryView, ...]


def validate_inventory(candidate: SupportCandidate, evidence_id: str, inventory: SourceInventory) -> None:
    if (not isinstance(inventory, SourceInventory) or inventory.status not in {'complete', 'unresolved'}
            or type(inventory.claims) is not tuple or len(inventory.claims) > 32):
        raise ValueError('Only bounded explicit source assessments can be recorded')
    if len(json.dumps(asdict(inventory), ensure_ascii=False, default=str).encode('utf-8')) > 64000:
        raise ValueError('Source inventory exceeds the record byte bound')
    own = {c.evidence_id: c.excerpt for c in candidate.evidence}
    for claim in inventory.claims:
        if not isinstance(claim, SourceRelation):
            raise ValueError('Expected a query-independent relation annotation')
        if any(not isinstance(s, str) or not s.strip() for s in (claim.subject, claim.relation, claim.value)):
            raise ValueError('Normalized relation keys must be nonblank')
        if claim.whole_fact_support not in {'supported', 'unsupported', 'unresolved'}:
            raise ValueError('Invalid whole-fact assessment')
        if claim.applicability not in {'bounded', 'unbounded', 'unresolved'}:
            raise ValueError('Invalid temporal assessment')
        if claim.whole_fact_support == 'supported' and claim.whole_fact_quote is None:
            raise ValueError('Supported whole facts require a source quotation')
        has_bound = claim.valid_from is not None or claim.valid_until is not None
        if ((claim.applicability == 'bounded' and not has_bound)
                or (claim.applicability == 'unbounded' and has_bound)
                or (has_bound and claim.temporal_quote is None)):
            raise ValueError('Temporal assessment lacks compatible source bounds')
        for boundary in (claim.valid_from, claim.valid_until):
            if boundary is not None and type(boundary) is not date:
                raise ValueError('Temporal bounds must be exact dates')
        if claim.valid_from is not None and claim.valid_until is not None and claim.valid_from >= claim.valid_until:
            raise ValueError('Empty or reversed interval')
        quotes = dict(subject_span=claim.subject_quote, relation_span=claim.relation_quote,
                      value_span=claim.value_quote, whole_fact_span=claim.whole_fact_quote,
                      temporal_span=claim.temporal_quote)
        bindings = bind_own_source_quotes(candidate.fact_id, evidence_id, quotes,
                                          sources={candidate.fact_id: own})
        if any(binding is not None and binding.status != 'bound' for binding in bindings.values()):
            raise ValueError('Source quotations must bind uniquely')
        if type(claim.condition_quotes) is not tuple or len(claim.condition_quotes) > 32:
            raise ValueError('Conditions must be a bounded immutable tuple')
        for quote in claim.condition_quotes:
            if bind_unique_quote(own[evidence_id], quote).status != 'bound':
                raise ValueError('Condition quotation must bind uniquely')


class RelationInventoryCache:
    """Owner-local FIFO cache; misses are unresolved work, not negative facts.

    Keys reuse the exact qualification boundary. No query, selected answer,
    condition-match decision or resolved query date is recorded here. Calls are
    serialized by the owner; the cache has no background worker or disk state.
    """

    def __init__(self, max_records: int = 256) -> None:
        if type(max_records) is not int or max_records < 1:
            raise ValueError('Cache capacity must be a positive integer')
        self.max_records = max_records
        self._records: OrderedDict[str, SourceInventory] = OrderedDict()

    def record(self, candidate: SupportCandidate, evidence_id: str, identity: QualificationIdentity,
               inventory: SourceInventory, revalidate: SnapshotFence) -> None:
        revalidate()
        key = qualification_key(candidate, evidence_id, identity)
        validate_inventory(candidate, evidence_id, inventory)
        previous = self._records.get(key)
        if previous is not None and previous != inventory:
            raise ValueError('Conflicting inventory for a frozen source identity')
        revalidate()
        self._records[key] = inventory
        while len(self._records) > self.max_records:
            self._records.popitem(last=False)

    def annotate(self, snapshot: SupportSnapshot, identity: QualificationIdentity,
                 revalidate: SnapshotFence) -> tuple[AnnotatedCandidate, ...]:
        revalidate()
        if len({c.fact_id for c in snapshot.candidates}) != len(snapshot.candidates):
            raise ValueError('Duplicate candidate identity')
        result = []
        for candidate in snapshot.candidates:
            inventories = []
            for source in candidate.evidence:
                key = qualification_key(candidate, source.evidence_id, identity)
                inventory = self._records.get(key)
                inventories.append(InventoryView(source.evidence_id,
                    inventory.status if inventory else 'missing',
                    inventory.claims if inventory else (), inventory is not None))
            result.append(AnnotatedCandidate(candidate, tuple(inventories)))
        revalidate()
        return tuple(result)
