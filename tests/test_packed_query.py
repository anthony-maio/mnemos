import asyncio
from dataclasses import replace
import json

import pytest

from benchmarks.local_memory.packed_query import bind_packed_transport, interpret_packed
from tests.test_relation_claim_references import prepared
from tests.test_relation_inventory_cache import IDENTITY


@pytest.mark.asyncio
@pytest.mark.parametrize('reverse', [False, True])
async def test_actual_supplied_order_controls_selected_original_fact(reverse):
    snapshot, cache = await prepared()
    if reverse:
        snapshot = replace(snapshot, candidates=tuple(reversed(snapshot.candidates)))
    calls = []
    async def provider(prompt, schema):
        inputs = json.loads(prompt.split('\nINPUT\n')[1])
        calls.append(inputs)
        assert inputs['claim_order'] == ['c0', 'c1']
        assert schema['type'] == 'string'
        return dict(done=True, response='"S|MAXA"')
    result = await interpret_packed(snapshot, IDENTITY, IDENTITY, cache, bind_packed_transport(provider), lambda: None)
    assert result.decision.status == 'selected'
    assert result.decision.fact_ids == (snapshot.candidates[0].fact_id,)
    assert result.raw_response == '"S|MAXA"' and len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['incomplete', 'malformed', 'revoked', 'cancelled'])
async def test_failed_transport_does_not_retry_or_abstain(failure):
    snapshot, cache = await prepared()
    calls = 0
    revoked = False
    async def provider(prompt, schema):
        nonlocal calls, revoked
        calls += 1
        if failure == 'cancelled': raise asyncio.CancelledError
        if failure == 'revoked': revoked = True
        return dict(done=failure != 'incomplete', response='"S|MA"' if failure == 'malformed' else '"S|MAXA"')
    def fence():
        if revoked: raise PermissionError('source revoked')
    if failure == 'cancelled':
        with pytest.raises(asyncio.CancelledError):
            await interpret_packed(snapshot, IDENTITY, IDENTITY, cache, bind_packed_transport(provider), fence)
    else:
        result = await interpret_packed(snapshot, IDENTITY, IDENTITY, cache, bind_packed_transport(provider), fence)
        assert result.decision.status == 'unavailable'
    assert calls == 1


@pytest.mark.asyncio
async def test_missing_inventory_blocks_generation():
    snapshot, cache = await prepared()
    snapshot = replace(snapshot, candidates=(replace(snapshot.candidates[0], version=2), snapshot.candidates[1]))
    async def provider(prompt, schema): pytest.fail('Missing source cannot disappear')
    result = await interpret_packed(snapshot, IDENTITY, IDENTITY, cache, bind_packed_transport(provider), lambda: None)
    assert result.decision.status == 'unavailable'
