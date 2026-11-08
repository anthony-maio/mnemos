from dataclasses import replace
import json

import pytest

from benchmarks.local_memory.compact_query import evaluate_reply
from benchmarks.local_memory.compact_query_wire import TransportReply
from benchmarks.local_memory.packed_query_wire import evaluate_packed, expand_packed, render_packed_request
from tests.test_compact_query_wire import request


CODES = [r+c for r in 'MXU' for c in 'AXU']


@pytest.mark.asyncio
@pytest.mark.parametrize('intent,day,header', [('S',None,'S'),('U',None,'U'),('D','2026-11-08','D20261108')])
async def test_all_two_claim_states_preserve_compact_decisions(intent, day, header):
    req = await request()
    for first in CODES:
        for second in CODES:
            original = TransportReply(req.request_id,req.binding_sha256,True,
                json.dumps(dict(t=intent,d=day,c={'c0':first,'c1':second})))
            packed = replace(original,text=json.dumps(header+'|'+first+second))
            baseline = evaluate_reply(req,original,lambda:None)
            result = evaluate_packed(req,packed,lambda:None)
            assert result.decision == baseline.decision
            assert result.selected_bases == baseline.selected_bases
            assert result.raw_response == packed.text


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['"S|MA"','"S|MAMAMA"','"S|MAMM"','"S|MA UU"',
    '"S20261108|MAUU"','"D20260230|MAUU"','"D2026-11-08|MAUU"','"S|MAUU|"',
    '"S|c0MAc1UU"','["S","MA","UU"]','"S|MAUU\\n"'])
async def test_malformed_or_incomplete_packed_reply_rejected(text):
    req=await request()
    with pytest.raises(ValueError):
        evaluate_packed(req,TransportReply(req.request_id,req.binding_sha256,True,text),lambda:None)


@pytest.mark.asyncio
async def test_explicit_order_is_in_model_input_and_missing_slots_never_shift():
    req=await request()
    prompt,schema=render_packed_request(req)
    payload=json.loads(prompt.split('\nINPUT\n')[1])
    assert payload['claim_order']==['c0','c1']
    assert payload['claims']==json.loads(req.model_input)['claims']
    assert '{2}' in schema['pattern']
    wire=TransportReply(req.request_id,req.binding_sha256,True,'"S|XUMA"')
    assert json.loads(expand_packed(req,wire).text)['c']=={'c0':'XU','c1':'MA'}


@pytest.mark.asyncio
async def test_identity_and_fences_still_apply_after_expansion():
    first,second=await request(),await request()
    wire=TransportReply(first.request_id,first.binding_sha256,True,'"S|MAMA"')
    with pytest.raises(ValueError): evaluate_packed(second,wire,lambda:None)
    with pytest.raises(ValueError): evaluate_packed(first,replace(wire,completed=False),lambda:None)
    def denied(): raise PermissionError('revoked')
    with pytest.raises(PermissionError): evaluate_packed(first,wire,denied)
