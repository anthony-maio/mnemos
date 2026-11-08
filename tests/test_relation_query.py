from dataclasses import replace
from datetime import date
import json

import pytest

from mnemos.support_intersection import SupportSnapshot
from benchmarks.local_memory.relation_inventory_cache import RelationInventoryCache
from benchmarks.local_memory.relation_query import interpret_query
from tests.test_relation_inventory_cache import IDENTITY, fixture


def setup():
    candidate, inventory = fixture()
    claim = replace(inventory.claims[0], applicability='unbounded', condition_quotes=())
    inventory = replace(inventory, claims=(claim,))
    cache = RelationInventoryCache()
    cache.record(candidate, 'source', IDENTITY, inventory, lambda: None)
    return SupportSnapshot('Who owns Project?', date(2026, 11, 8), (candidate,), 'snap'), cache, inventory


def reply(prompt):
    payload = json.loads(prompt.split('\nINPUT\n')[1])
    assert 'labels' not in payload
    return dict(request_key=payload['request_key'], status='resolved', at='2026-11-08',
                claims=[dict(fact_id=row['candidate']['fact_id'], evidence_id=inv['evidence_id'],
                             claim_index=i, relation_match='matches', condition_match='applies')
                        for row in payload['cohort'] for inv in row['inventories']
                        for i, _ in enumerate(inv['claims'])])


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation,expected', [('', 'selected'), ('missing', 'unavailable'),
    ('duplicate', 'unavailable'), ('stale', 'unavailable'), ('condition', 'unavailable'),
    ('excluded', 'absent'), ('unresolved', 'unavailable')])
async def test_complete_query_contract(mutation, expected):
    snapshot, cache, _ = setup()
    async def provider(prompt):
        data = reply(prompt)
        if mutation == 'missing': data['claims'] = []
        if mutation == 'duplicate': data['claims'] *= 2
        if mutation == 'stale': data['request_key'] = 'old'
        if mutation == 'condition': data['claims'][0]['condition_match'] = 'unresolved'
        if mutation == 'excluded': data['claims'][0]['relation_match'] = 'excluded'
        if mutation == 'unresolved': data['status'] = 'unresolved'
        return json.dumps(data)
    result = await interpret_query(snapshot, IDENTITY, IDENTITY, cache, provider, lambda: None)
    assert result.decision.status == expected
    if expected == 'selected': assert result.decision.fact_ids == ('fact',)


@pytest.mark.asyncio
async def test_missing_competitor_never_disappears():
    snapshot, cache, _ = setup()
    other = replace(snapshot.candidates[0], fact_id='competitor')
    snapshot = replace(snapshot, candidates=(*snapshot.candidates, other))
    async def provider(prompt): pytest.fail('Incomplete inventory must remain unavailable')
    result = await interpret_query(snapshot, IDENTITY, IDENTITY, cache, provider, lambda: None)
    assert result.decision.status == 'unavailable' and result.error_type == 'IncompleteInventory'


@pytest.mark.asyncio
async def test_conflicting_values_preserve_both_candidates():
    snapshot, cache, inventory = setup()
    other = replace(snapshot.candidates[0], fact_id='competitor', evidence=(
        replace(snapshot.candidates[0].evidence[0], excerpt='Project owner is Jon.'),))
    claim = replace(inventory.claims[0], value='Jon', value_quote='Jon', whole_fact_quote='Project owner is Jon.')
    cache.record(other, 'source', IDENTITY, replace(inventory, claims=(claim,)), lambda: None)
    snapshot = replace(snapshot, candidates=(*snapshot.candidates, other))
    async def provider(prompt): return json.dumps(reply(prompt))
    result = await interpret_query(snapshot, IDENTITY, IDENTITY, cache, provider, lambda: None)
    assert result.decision.status == 'conflict'


@pytest.mark.asyncio
@pytest.mark.parametrize('requested,expected', [('2025-01-01', 'selected'), ('2026-01-01', 'absent')])
async def test_historical_half_open_interval(requested, expected):
    snapshot, _, inventory = setup()
    candidate = snapshot.candidates[0]
    text = candidate.evidence[0].excerpt + ' Valid 2025-01-01 until 2026-01-01.'
    candidate = replace(candidate, evidence=(replace(candidate.evidence[0], excerpt=text),))
    claim = replace(inventory.claims[0], applicability='bounded', valid_from=date(2025, 1, 1),
                    valid_until=date(2026, 1, 1), temporal_quote='2025-01-01 until 2026-01-01')
    cache = RelationInventoryCache()
    cache.record(candidate, 'source', IDENTITY, replace(inventory, claims=(claim,)), lambda: None)
    snapshot = replace(snapshot, query='Who owns Project on ' + requested + '?', candidates=(candidate,))
    async def provider(prompt):
        data = reply(prompt)
        data['at'] = requested
        return json.dumps(data)
    result = await interpret_query(snapshot, IDENTITY, IDENTITY, cache, provider, lambda: None)
    assert result.decision.status == expected


@pytest.mark.asyncio
async def test_revocation_during_provider_prevents_selection():
    snapshot, cache, _ = setup()
    changed = False
    async def provider(prompt):
        nonlocal changed
        changed = True
        return json.dumps(reply(prompt))
    def fence():
        if changed: raise PermissionError('forgotten')
    result = await interpret_query(snapshot, IDENTITY, IDENTITY, cache, provider, fence)
    assert result.decision.status == 'unavailable'


@pytest.mark.asyncio
async def test_unsupported_claims_are_visible_but_not_selected():
    snapshot, _, inventory = setup()
    cache = RelationInventoryCache()
    claim = replace(inventory.claims[0], whole_fact_support='unsupported')
    cache.record(snapshot.candidates[0], 'source', IDENTITY,
                 replace(inventory, claims=(claim,)), lambda: None)
    async def provider(prompt):
        payload = json.loads(prompt.split('\nINPUT\n')[1])
        assert payload['cohort'][0]['inventories'][0]['claims'][0]['whole_fact_support'] == 'unsupported'
        return json.dumps(reply(prompt))
    result = await interpret_query(snapshot, IDENTITY, IDENTITY, cache, provider, lambda: None)
    assert result.decision.status == 'absent'


@pytest.mark.asyncio
async def test_invalid_query_identity_fails_before_provider():
    snapshot, cache, _ = setup()
    async def provider(prompt): pytest.fail('Unbound query policy must not run')
    result = await interpret_query(snapshot, IDENTITY, None, cache, provider, lambda: None)
    assert result.decision.status == 'unavailable' and result.error_type == 'ValueError'
