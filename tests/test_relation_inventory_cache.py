from dataclasses import fields, replace
from datetime import date

import pytest

from mnemos.answer_support import SupportCandidate, SupportEvidence
from mnemos.support_intersection import SupportSnapshot
from benchmarks.local_memory.relation_inventory_cache import RelationInventoryCache, SourceInventory, SourceRelation
from benchmarks.local_memory.source_qualification import QualificationIdentity


IDENTITY = QualificationIdentity('a' * 64, 'b' * 64, 'c' * 64)
TEXT = 'Project owner is Mira, only during release.'


def fixture():
    evidence = SupportEvidence('source', TEXT, 'event', 'ref')
    candidate = SupportCandidate('fact', 1, TEXT, 'project', 'scope', (evidence,), 'fingerprint')
    claim = SourceRelation('project', 'owner', 'Mira', 'Project', 'owner', 'Mira', TEXT, None,
                           ('only during release',), None, None, 'unresolved', 'supported')
    return candidate, SourceInventory('complete', (claim,))


def view(cache, candidate, identity=IDENTITY, query='Who owns Project?'):
    snapshot = SupportSnapshot(query, date(2026, 11, 8), (candidate,), 'snapshot')
    return cache.annotate(snapshot, identity, lambda: None)[0].inventories[0]


def test_annotations_reuse_across_queries_without_condition_match_or_answer():
    candidate, inventory = fixture()
    cache = RelationInventoryCache()
    cache.record(candidate, 'source', IDENTITY, inventory, lambda: None)
    first = view(cache, candidate)
    second = view(cache, candidate, query='Who owns Project outside a release?')
    assert first == second and first.cache_hit
    assert first.claims[0].condition_quotes == ('only during release',)
    assert not {'condition_match', 'query', 'selected', 'at'} & {f.name for f in fields(SourceRelation)}


@pytest.mark.parametrize('change', ['version', 'content', 'scope', 'excerpt', 'reference', 'sibling', 'policy', 'model', 'runtime'])
def test_revision_or_identity_change_misses(change):
    candidate, inventory = fixture()
    cache = RelationInventoryCache()
    cache.record(candidate, 'source', IDENTITY, inventory, lambda: None)
    identity = IDENTITY
    if change == 'version': candidate = replace(candidate, version=2)
    if change == 'content': candidate = replace(candidate, content=TEXT + ' Changed.')
    if change == 'scope': candidate = replace(candidate, scope_id='other')
    if change == 'excerpt': candidate = replace(candidate, evidence=(replace(candidate.evidence[0], excerpt=TEXT + ' Changed.'),))
    if change == 'reference': candidate = replace(candidate, evidence=(replace(candidate.evidence[0], source_ref='other'),))
    if change == 'sibling': candidate = replace(candidate, evidence=(*candidate.evidence, replace(candidate.evidence[0], evidence_id='new')))
    if change in {'policy', 'model', 'runtime'}: identity = replace(identity, **{change + '_sha256': 'd' * 64})
    result = view(cache, candidate, identity)
    assert result.status == 'missing' and not result.cache_hit and not result.claims


def test_forgotten_candidate_not_reintroduced_and_fence_failure_propagates():
    candidate, inventory = fixture()
    cache = RelationInventoryCache()
    cache.record(candidate, 'source', IDENTITY, inventory, lambda: None)
    empty = SupportSnapshot('owner?', date(2026, 11, 8), (), 'new')
    assert cache.annotate(empty, IDENTITY, lambda: None) == ()
    def denied(): raise PermissionError('scope revoked')
    with pytest.raises(PermissionError): cache.annotate(empty, IDENTITY, denied)


def test_unavailable_or_ambiguous_inventory_cannot_be_recorded():
    candidate, inventory = fixture()
    cache = RelationInventoryCache()
    for bad in (replace(inventory, status='unavailable'),
                replace(inventory, claims=(replace(inventory.claims[0], value_quote='missing'),))):
        with pytest.raises(ValueError): cache.record(candidate, 'source', IDENTITY, bad, lambda: None)
    assert not view(cache, candidate).cache_hit


def test_failed_postvalidation_does_not_leave_record():
    candidate, inventory = fixture()
    cache = RelationInventoryCache()
    calls = 0
    def fence():
        nonlocal calls
        calls += 1
        if calls == 2: raise RuntimeError('source changed')
    with pytest.raises(RuntimeError): cache.record(candidate, 'source', IDENTITY, inventory, fence)
    assert not view(cache, candidate).cache_hit


def test_bounded_eviction_returns_missing_without_negative_assessment():
    candidate, inventory = fixture()
    cache = RelationInventoryCache(max_records=1)
    cache.record(candidate, 'source', IDENTITY, inventory, lambda: None)
    changed = replace(candidate, version=2)
    cache.record(changed, 'source', IDENTITY, inventory, lambda: None)
    assert view(cache, candidate).status == 'missing'
    assert view(cache, changed).cache_hit


def test_conflicting_inventory_cannot_replace_frozen_assessment():
    candidate, inventory = fixture()
    cache = RelationInventoryCache()
    cache.record(candidate, 'source', IDENTITY, inventory, lambda: None)
    with pytest.raises(ValueError, match='Conflicting'):
        cache.record(candidate, 'source', IDENTITY, replace(inventory, status='unresolved'), lambda: None)
    assert view(cache, candidate).status == 'complete'


def test_inventory_byte_bound_rejects_oversized_normalized_values():
    candidate, inventory = fixture()
    cache = RelationInventoryCache()
    oversized = replace(inventory, claims=(replace(inventory.claims[0], value='x' * 64001),))
    with pytest.raises(ValueError, match='byte bound'):
        cache.record(candidate, 'source', IDENTITY, oversized, lambda: None)
    assert not view(cache, candidate).cache_hit
