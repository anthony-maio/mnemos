from dataclasses import replace
from datetime import date
import json

import pytest

from benchmarks.local_memory.relation_inventory_cache_v2 import RelationInventoryCache
from benchmarks.local_memory.source_relation_producer_v2 import produce_source
from benchmarks.local_memory.relation_query_v2 import interpret_query
from mnemos.support_intersection import SupportSnapshot
from tests.test_relation_inventory_cache import IDENTITY, fixture
from tests.test_source_relation_producer import response
from tests.test_relation_query import reply


async def populate(cache, candidate, temporal, *, raw_mutation=None):
    async def provider(prompt):
        data = response(prompt)
        claim = data['claims'][0]
        for name in ('temporal_quote', 'valid_from', 'valid_until', 'applicability'):
            del claim[name]
        claim['subject_quote'] = 'Project'
        claim['whole_fact_quote'] = candidate.evidence[0].excerpt
        claim['condition_quotes'] = []
        claim['time'] = temporal
        if raw_mutation: raw_mutation(data)
        return json.dumps(data)
    return await produce_source(candidate, 'source', IDENTITY, cache, provider, lambda: None)


def time_record(text, meaning='present_assertion', assertion_date=None):
    return dict(meaning=meaning, quotation=text, valid_from=None, valid_until=None,
                assertion_date=assertion_date)


@pytest.mark.asyncio
@pytest.mark.parametrize('intent,at,expected,basis', [
    ('source_state', None, 'selected', 'source_assertion'),
    ('at_date', '2026-11-08', 'unavailable', None),
    ('at_date', '2020-01-01', 'unavailable', None),
    ('at_date', '2030-01-01', 'unavailable', None),
    ('source_state', '2026-11-08', 'unavailable', None),
])
async def test_source_to_query_preserves_temporal_basis(intent, at, expected, basis):
    candidate, _ = fixture()
    cache = RelationInventoryCache()
    produced = await populate(cache, candidate, time_record(candidate.evidence[0].excerpt))
    assert produced.status == 'assessed'
    snapshot = SupportSnapshot('What does the retained source say?', date(2026, 11, 8), (candidate,), 'snap')
    async def provider(prompt):
        data = reply(prompt)
        data.update(time_intent=intent, at=at)
        return json.dumps(data)
    result = await interpret_query(snapshot, IDENTITY, IDENTITY, cache, provider, lambda: None)
    assert result.decision.status == expected
    assert [b.basis for b in result.selected_bases] == ([basis] if basis else [])
    if basis:
        assert result.selected_bases[0].fact_id == candidate.fact_id


@pytest.mark.asyncio
@pytest.mark.parametrize('at,expected', [('2026-11-08', 'selected'), ('2026-11-09', 'unavailable')])
async def test_explicit_as_of_date_survives_production_and_selection(at, expected):
    candidate, _ = fixture()
    text = 'As of November 8, 2026, Project owner is Mira.'
    candidate = replace(candidate, content=text, evidence=(replace(candidate.evidence[0], excerpt=text),))
    cache = RelationInventoryCache()
    produced = await populate(cache, candidate, time_record(text, assertion_date='2026-11-08'))
    assert produced.status == 'assessed'
    snapshot = SupportSnapshot('Who owns Project on ' + at + '?', date(2026, 11, 8), (candidate,), 'snap')
    async def provider(prompt):
        data = reply(prompt)
        data.update(time_intent='at_date', at=at)
        return json.dumps(data)
    result = await interpret_query(snapshot, IDENTITY, IDENTITY, cache, provider, lambda: None)
    assert result.decision.status == expected
    assert [b.basis for b in result.selected_bases] == (['dated_support'] if expected == 'selected' else [])


@pytest.mark.asyncio
async def test_legacy_temporal_fields_cannot_enter_v2_cache():
    candidate, _ = fixture()
    cache = RelationInventoryCache()
    result = await populate(cache, candidate, time_record(candidate.evidence[0].excerpt),
        raw_mutation=lambda data: data['claims'][0].update(applicability='unbounded'))
    assert result.status == 'unavailable'
    assert not cache.lookup(candidate, 'source', IDENTITY, lambda: None).cache_hit


@pytest.mark.asyncio
async def test_foreign_temporal_quote_cannot_enter_v2_cache():
    candidate, _ = fixture()
    cache = RelationInventoryCache()
    result = await populate(cache, candidate, time_record('Another source says now.'))
    assert result.status == 'unavailable'
    assert not cache.lookup(candidate, 'source', IDENTITY, lambda: None).cache_hit
