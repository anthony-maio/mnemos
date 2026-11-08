from dataclasses import replace
from datetime import date
import json

import pytest

from mnemos.support_intersection import SupportSnapshot
from benchmarks.local_memory.relation_inventory_cache_v2 import RelationInventoryCache
from benchmarks.local_memory.relation_claim_references import reference_claims, resolve_complete_reply
from benchmarks.local_memory.relation_query_v3 import interpret_query
from tests.test_relation_inventory_cache import IDENTITY, fixture
from tests.test_relation_inventory_v2 import populate, time_record


async def prepared():
    candidate, _ = fixture()
    other = replace(candidate, fact_id='second')
    cache = RelationInventoryCache()
    for fact in (candidate, other):
        assert (await populate(cache, fact, time_record(fact.evidence[0].excerpt))).status == 'assessed'
    snapshot = SupportSnapshot('What do the retained sources say?', date(2026, 11, 8),
                               (candidate, other), 'snap')
    return snapshot, cache


@pytest.mark.asyncio
async def test_two_source_local_zero_claims_have_distinct_explicit_refs():
    snapshot, cache = await prepared()
    cohort = cache.annotate(snapshot, IDENTITY, lambda: None)
    refs = reference_claims(cohort, IDENTITY)
    assert [r.claim_index for r in refs] == [0, 0]
    assert len({r.claim_ref for r in refs}) == 2
    assert resolve_complete_reply(refs, tuple(r.claim_ref for r in reversed(refs))) == tuple(reversed(refs))
    changed_query = replace(snapshot, query='A different question', reference_date=date(2027, 1, 1))
    assert reference_claims(cache.annotate(changed_query, IDENTITY, lambda: None), IDENTITY) == refs
    changed_fact = replace(cohort[0], candidate=replace(cohort[0].candidate, version=2))
    assert reference_claims((changed_fact, cohort[1]), IDENTITY)[0].claim_ref != refs[0].claim_ref
    changed_claim = replace(cohort[0].inventories[0].claims[0], whole_fact_support='unresolved')
    changed_inv = replace(cohort[0].inventories[0], claims=(changed_claim,))
    assert reference_claims((replace(cohort[0], inventories=(changed_inv,)), cohort[1]), IDENTITY)[0].claim_ref != refs[0].claim_ref


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation,expected', [('', 'selected'), ('reverse', 'selected'),
    ('missing', 'unavailable'), ('duplicate', 'unavailable'), ('foreign', 'unavailable'),
    ('legacy_indices', 'unavailable')])
async def test_query_copies_exact_refs_without_inferred_indices(mutation, expected):
    snapshot, cache = await prepared()
    async def provider(prompt):
        payload = json.loads(prompt.split('\nINPUT\n')[1])
        assert len(payload['referenced_claims']) == 2
        assert all('claim_index' not in item for item in payload['referenced_claims'])
        claims = [dict(claim_ref=item['claim_ref'], relation_match='matches', condition_match='applies')
                  for item in payload['referenced_claims']]
        if mutation == 'reverse': claims.reverse()
        if mutation == 'missing': claims.pop()
        if mutation == 'duplicate': claims[1] = claims[0]
        if mutation == 'foreign': claims[1]['claim_ref'] = 'claim_' + 'f' * 64
        if mutation == 'legacy_indices':
            claims = [dict(fact_id='fact', evidence_id='source', claim_index=0,
                           relation_match='matches', condition_match='applies')]
        return json.dumps(dict(request_key=payload['request_key'], status='resolved',
                               time_intent='source_state', at=None, claims=claims))
    result = await interpret_query(snapshot, IDENTITY, IDENTITY, cache, provider, lambda: None)
    assert result.decision.status == expected
    if expected == 'selected':
        assert set(result.decision.fact_ids) == {'fact', 'second'}
        assert {b.fact_id for b in result.selected_bases} == {'fact', 'second'}


@pytest.mark.asyncio
async def test_missing_source_is_not_hidden_by_flattened_refs():
    snapshot, cache = await prepared()
    missing = replace(snapshot.candidates[1], version=2)
    snapshot = replace(snapshot, candidates=(snapshot.candidates[0], missing))
    async def provider(prompt): pytest.fail('Missing inventory must remain unavailable')
    result = await interpret_query(snapshot, IDENTITY, IDENTITY, cache, provider, lambda: None)
    assert result.decision.status == 'unavailable' and result.error_type == 'IncompleteInventory'
