import asyncio
from dataclasses import asdict, replace
import inspect
import json

import pytest

from benchmarks.local_memory.relation_inventory_cache import RelationInventoryCache
from benchmarks.local_memory.source_relation_producer import prepare_source, produce_source
from tests.test_relation_inventory_cache import IDENTITY, fixture, view


def response(prompt):
    inputs = json.loads(prompt.split('\nINPUT\n')[1])
    _, inventory = fixture()
    return dict(request_key=inputs['request_key'], fact_id=inputs['fact_id'],
                evidence_id=inputs['target_evidence_id'], status=inventory.status,
                claims=[asdict(c) for c in inventory.claims])


@pytest.mark.asyncio
async def test_source_only_request_and_repeat_hit_make_one_provider_call():
    candidate, _ = fixture()
    cache = RelationInventoryCache()
    calls = []
    async def provider(prompt):
        inputs = json.loads(prompt.split('\nINPUT\n')[1])
        assert not {'query', 'labels', 'reference_date', 'expected_answers'} & inputs.keys()
        assert inputs['candidate']['evidence'][0]['excerpt'] == candidate.evidence[0].excerpt
        calls.append(prompt)
        return json.dumps(response(prompt))
    first = await produce_source(candidate, 'source', IDENTITY, cache, provider, lambda: None)
    second = await produce_source(candidate, 'source', IDENTITY, cache, provider, lambda: None)
    assert first.status == second.status == 'assessed'
    assert not first.cache_hit and second.cache_hit and len(calls) == 1
    assert first.inventory == second.inventory
    assert 'query' not in inspect.signature(produce_source).parameters


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation', ['foreign', 'stale', 'condition_match', 'offsets', 'missing_quote', 'unavailable'])
async def test_invalid_reply_never_populates_cache(mutation):
    candidate, _ = fixture()
    cache = RelationInventoryCache()
    async def provider(prompt):
        data = response(prompt)
        if mutation == 'foreign': data['evidence_id'] = 'other'
        if mutation == 'stale': data['request_key'] = 'old'
        if mutation == 'condition_match': data['claims'][0]['condition_match'] = 'applies'
        if mutation == 'offsets': data['claims'][0]['value_start'] = 0
        if mutation == 'missing_quote': data['claims'][0]['value_quote'] = 'not in source'
        if mutation == 'unavailable': data['status'] = 'unavailable'
        return json.dumps(data)
    result = await produce_source(candidate, 'source', IDENTITY, cache, provider, lambda: None)
    assert result.status == 'unavailable' and result.inventory is None
    assert not view(cache, candidate).cache_hit


@pytest.mark.asyncio
async def test_source_change_during_generation_does_not_record():
    candidate, _ = fixture()
    cache = RelationInventoryCache()
    changed = False
    async def provider(prompt):
        nonlocal changed
        changed = True
        return json.dumps(response(prompt))
    def fence():
        if changed: raise RuntimeError('source changed')
    result = await produce_source(candidate, 'source', IDENTITY, cache, provider, fence)
    assert result.status == 'unavailable' and not view(cache, candidate).cache_hit


@pytest.mark.asyncio
async def test_provider_failure_and_cancellation_leave_no_record():
    candidate, _ = fixture()
    cache = RelationInventoryCache()
    async def failed(prompt): raise OSError('offline')
    result = await produce_source(candidate, 'source', IDENTITY, cache, failed, lambda: None)
    assert result.status == 'unavailable' and not view(cache, candidate).cache_hit
    async def cancelled(prompt): raise asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await produce_source(candidate, 'source', IDENTITY, cache, cancelled, lambda: None)
    assert not view(cache, candidate).cache_hit


def test_request_binding_and_context_include_changed_sibling_source():
    candidate, _ = fixture()
    first, key = prepare_source(candidate, 'source', IDENTITY)
    sibling = replace(candidate.evidence[0], evidence_id='sibling', excerpt='Different source context.')
    second, changed = prepare_source(replace(candidate, evidence=(*candidate.evidence, sibling)), 'source', IDENTITY)
    assert key != changed and first != second
    assert 'Different source context.' in second


@pytest.mark.asyncio
async def test_cached_inventory_still_requires_current_authorization():
    candidate, inventory = fixture()
    cache = RelationInventoryCache()
    cache.record(candidate, 'source', IDENTITY, inventory, lambda: None)
    async def provider(prompt):
        pytest.fail('Denied lookup must not call provider')
    def denied():
        raise PermissionError('scope revoked')
    result = await produce_source(candidate, 'source', IDENTITY, cache, provider, denied)
    assert result.status == 'unavailable' and result.inventory is None
    assert result.error_type == 'PermissionError'


@pytest.mark.asyncio
async def test_unresolved_assessment_is_reused_without_becoming_supported():
    candidate, _ = fixture()
    cache = RelationInventoryCache()
    calls = 0
    async def provider(prompt):
        nonlocal calls
        calls += 1
        data = response(prompt)
        data['status'] = 'unresolved'
        data['claims'][0]['whole_fact_support'] = 'unresolved'
        return json.dumps(data)
    for _ in range(2):
        result = await produce_source(candidate, 'source', IDENTITY, cache, provider, lambda: None)
        assert result.status == 'assessed'
        assert result.inventory.status == 'unresolved'
        assert result.inventory.claims[0].whole_fact_support == 'unresolved'
    assert calls == 1


@pytest.mark.asyncio
async def test_timeout_cancels_provider_without_recording(monkeypatch):
    candidate, _ = fixture()
    cache = RelationInventoryCache()
    real_wait_for = asyncio.wait_for
    cancelled = False
    async def short_wait(awaitable, timeout):
        assert timeout == 15
        return await real_wait_for(awaitable, timeout=0.01)
    monkeypatch.setattr(asyncio, 'wait_for', short_wait)
    async def provider(prompt):
        nonlocal cancelled
        try:
            await asyncio.Future()
        finally:
            cancelled = True
    result = await produce_source(candidate, 'source', IDENTITY, cache, provider, lambda: None)
    assert result.status == 'unavailable' and result.error_type == 'TimeoutError'
    assert cancelled and not view(cache, candidate).cache_hit


@pytest.mark.asyncio
async def test_oversized_request_never_calls_provider():
    candidate, _ = fixture()
    candidate = replace(candidate, content='x' * 64000)
    async def provider(prompt):
        pytest.fail('Oversized input must not be truncated and sent')
    result = await produce_source(candidate, 'source', IDENTITY, RelationInventoryCache(), provider, lambda: None)
    assert result.status == 'unavailable' and result.error_type == 'ValueError'
