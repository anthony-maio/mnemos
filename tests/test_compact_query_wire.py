from dataclasses import replace
import json

import pytest

from benchmarks.local_memory.compact_query_wire import prepare_compact, decode_compact, TransportReply
from tests.test_relation_claim_references import prepared
from tests.test_relation_inventory_cache import IDENTITY


async def request():
    snapshot, cache = await prepared()
    return prepare_compact(snapshot, cache.annotate(snapshot, IDENTITY, lambda: None),
                           IDENTITY, IDENTITY, lambda: None)


def reply(req, data=None):
    data = data if data is not None else dict(t='S', d=None, c={'c0': 'MA', 'c1': 'UU'})
    return TransportReply(req.request_id, req.binding_sha256, True, json.dumps(data))


@pytest.mark.asyncio
async def test_full_evidence_stays_visible_without_generated_identity_metadata():
    req = await request()
    model = json.loads(req.model_input)
    assert len(model['facts']) == len(model['claims']) == 2
    for i, candidate in enumerate(req.snapshot.candidates):
        assert model['facts'][i]['fact'] == candidate.content
        assert model['facts'][i]['sources']['s0'] == candidate.evidence[0].excerpt
    assert req.binding_sha256 not in req.model_input
    assert all(ref.claim_ref not in req.model_input for _, ref in req.aliases)
    result = decode_compact(req, reply(req), lambda: None)
    assert result.requested_time.meaning == 'source_state'
    assert [r.claim.fact_id for r in result.interpretations] == ['fact', 'second']
    assert result.interpretations[1].relation_match == 'unresolved'
    assert result.interpretations[1].condition_match == 'unresolved'


@pytest.mark.asyncio
@pytest.mark.parametrize('code', ['MA', 'MX', 'MU', 'XA', 'XX', 'XU', 'UA', 'UX', 'UU'])
async def test_every_relation_condition_state_survives(code):
    req = await request()
    result = decode_compact(req, reply(req, dict(t='U', d=None, c={'c0': code, 'c1': code})), lambda: None)
    assert result.requested_time.meaning == 'unresolved'
    assert result.interpretations[0].relation_match == {'M':'matches','X':'excluded','U':'unresolved'}[code[0]]
    assert result.interpretations[0].condition_match == {'A':'applies','X':'excluded','U':'unresolved'}[code[1]]


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation', ['missing', 'foreign', 'bad_code', 'extra', 'wrong_date', 'bad_intent'])
async def test_malformed_reply_is_rejected(mutation):
    req = await request()
    data = dict(t='S', d=None, c={'c0':'MA','c1':'UU'})
    if mutation == 'missing': del data['c']['c1']
    if mutation == 'foreign': data['c']['c2'] = data['c'].pop('c1')
    if mutation == 'bad_code': data['c']['c1'] = 'MM'
    if mutation == 'extra': data['selected'] = ['fact']
    if mutation == 'wrong_date': data['d'] = '2026-11-08'
    if mutation == 'bad_intent': data['t'] = []
    with pytest.raises(ValueError): decode_compact(req, reply(req, data), lambda: None)


@pytest.mark.asyncio
async def test_duplicate_keys_rejected_instead_of_silently_overwritten():
    req = await request()
    wire = replace(reply(req), text='{"t":"S","d":null,"c":{"c0":"MA","c1":"UU","c1":"MA"}}')
    with pytest.raises(ValueError, match='Duplicate'): decode_compact(req, wire, lambda: None)


@pytest.mark.asyncio
async def test_cross_request_reply_and_local_envelope_tampering_fail():
    first, second = await request(), await request()
    assert first.aliases == second.aliases and first.request_id != second.request_id
    with pytest.raises(ValueError): decode_compact(second, reply(first), lambda: None)
    with pytest.raises(ValueError): decode_compact(replace(first, model_input='changed'), reply(first), lambda: None)
    with pytest.raises(ValueError): decode_compact(first, replace(reply(first), completed=False), lambda: None)


@pytest.mark.asyncio
async def test_revoked_snapshot_cannot_decode_previously_completed_reply():
    req = await request()
    def denied(): raise PermissionError('source forgotten')
    with pytest.raises(PermissionError): decode_compact(req, reply(req), denied)


@pytest.mark.asyncio
async def test_preparation_cannot_drop_competitor():
    snapshot, cache = await prepared()
    cohort = cache.annotate(snapshot, IDENTITY, lambda: None)
    with pytest.raises(ValueError): prepare_compact(snapshot, cohort[:1], IDENTITY, IDENTITY, lambda: None)


@pytest.mark.asyncio
async def test_dated_intent_preserves_explicit_date():
    req = await request()
    result = decode_compact(req, reply(req, dict(t='D', d='2025-01-02', c={'c0':'MA','c1':'MA'})), lambda: None)
    assert result.requested_time.at.isoformat() == '2025-01-02'
