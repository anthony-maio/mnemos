import asyncio
from dataclasses import replace
import json

import pytest

from benchmarks.local_memory.compact_query import bind_transport, interpret_compact, evaluate_reply
from benchmarks.local_memory.compact_query_wire import prepare_compact, TransportReply
from benchmarks.local_memory.relation_query_v2 import interpret_query
from benchmarks.local_memory.relation_inventory_cache_v2 import RelationInventoryCache, SourceInventory
from tests.test_relation_claim_references import prepared
from tests.test_relation_inventory_cache import IDENTITY


CODES = [r+c for r in 'MXU' for c in 'AXU']


@pytest.mark.asyncio
@pytest.mark.parametrize('first', CODES)
@pytest.mark.parametrize('second', CODES)
async def test_all_two_claim_judgments_preserve_v2_decisions(first, second):
    snapshot, cache = await prepared()
    async def compact(prompt, schema):
        assert schema['properties']['c']['required'] == ['c0', 'c1']
        return dict(done=True, response=json.dumps(dict(t='S', d=None, c={'c0':first,'c1':second})))
    async def original(prompt):
        data=json.loads(prompt.split('\nINPUT\n')[1])
        return json.dumps(dict(request_key=data['request_key'], status='resolved', time_intent='source_state',
            at=None, claims=[dict(fact_id=fact.fact_id, evidence_id='source', claim_index=0,
                relation_match={'M':'matches','X':'excluded','U':'unresolved'}[code[0]],
                condition_match={'A':'applies','X':'excluded','U':'unresolved'}[code[1]])
                for fact,code in zip(snapshot.candidates,[first,second])]))
    old=await interpret_query(snapshot,IDENTITY,IDENTITY,cache,original,lambda:None)
    new=await interpret_compact(snapshot,IDENTITY,IDENTITY,cache,bind_transport(compact),lambda:None)
    assert new.decision==old.decision
    assert new.selected_bases==old.selected_bases


@pytest.mark.asyncio
async def test_conflict_and_unsupported_fact_handling():
    snapshot, cache = await prepared()
    cohort=cache.annotate(snapshot,IDENTITY,lambda:None)
    other=replace(cohort[1].candidate, content='Project owner is Jon.',
        evidence=(replace(cohort[1].candidate.evidence[0], excerpt='Project owner is Jon.'),))
    source=cohort[1].inventories[0].claims[0]
    claim=replace(source,value='Jon',value_quote='Jon',whole_fact_quote='Project owner is Jon.',
                  condition_quotes=(),time=replace(source.time,quotation='Project owner is Jon.'))
    async def provider(prompt,schema):
        return dict(done=True,response=json.dumps(dict(t='S',d=None,c={'c0':'MA','c1':'MA'})))
    for support,expected in [('supported','conflict'),('unsupported','selected'),('unresolved','unavailable')]:
        current=RelationInventoryCache()
        current.record(cohort[0].candidate,'source',IDENTITY,
                       SourceInventory('complete',cohort[0].inventories[0].claims),lambda:None)
        current.record(other,'source',IDENTITY,SourceInventory('complete',
                       (replace(claim,whole_fact_support=support),)),lambda:None)
        snap=replace(snapshot,candidates=(cohort[0].candidate,other))
        result=await interpret_compact(snap,IDENTITY,IDENTITY,current,bind_transport(provider),lambda:None)
        assert result.decision.status==expected


@pytest.mark.asyncio
async def test_dated_request_does_not_turn_unanchored_assertion_into_support():
    snapshot,cache=await prepared()
    async def provider(prompt,schema):
        return dict(done=True,response=json.dumps(dict(t='D',d='2026-11-08',c={'c0':'MA','c1':'MA'})))
    result=await interpret_compact(snapshot,IDENTITY,IDENTITY,cache,bind_transport(provider),lambda:None)
    assert result.decision.status=='unavailable' and not result.selected_bases


@pytest.mark.asyncio
async def test_transport_associates_identity_locally():
    snapshot,cache=await prepared()
    cohort=cache.annotate(snapshot,IDENTITY,lambda:None)
    request=prepare_compact(snapshot,cohort,IDENTITY,IDENTITY,lambda:None)
    async def provider(prompt,schema):
        assert request.request_id not in prompt and request.binding_sha256 not in prompt
        return dict(done=True,response=json.dumps(dict(t='S',d=None,c={'c0':'MA','c1':'MA'})))
    reply=await bind_transport(provider)(request)
    assert reply.request_id==request.request_id and reply.binding_sha256==request.binding_sha256
    assert evaluate_reply(request,reply,lambda:None).decision.status=='selected'


@pytest.mark.asyncio
@pytest.mark.parametrize('failure',['incomplete','revoked','cancelled'])
async def test_provider_failures_cannot_become_absent_answers(failure):
    snapshot,cache=await prepared()
    revoked=False
    async def provider(prompt,schema):
        nonlocal revoked
        if failure=='cancelled': raise asyncio.CancelledError
        if failure=='revoked': revoked=True
        return dict(done=failure!='incomplete',response=json.dumps(dict(t='S',d=None,c={'c0':'MA','c1':'MA'})))
    def fence():
        if revoked: raise PermissionError('forgotten')
    if failure=='cancelled':
        with pytest.raises(asyncio.CancelledError):
            await interpret_compact(snapshot,IDENTITY,IDENTITY,cache,bind_transport(provider),fence)
    else:
        result=await interpret_compact(snapshot,IDENTITY,IDENTITY,cache,bind_transport(provider),fence)
        assert result.decision.status=='unavailable'
