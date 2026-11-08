import asyncio
from dataclasses import replace
from datetime import date
import json

import pytest

from mnemos.answer_support import SupportCandidate, SupportEvidence
from mnemos.support_intersection import SupportSnapshot
from benchmarks.local_memory.relational_support_producer import prepare, produce


def snapshot():
    text = 'Project owner is Taylor.'
    evidence = SupportEvidence('source', text, None, 'test:source')
    fact = SupportCandidate('fact', 1, text, 'project', 'scope', (evidence,), 'fact-binding')
    return SupportSnapshot('Who owns Project?', date(2026, 11, 8), (fact,), 'snapshot')


def reply(prompt):
    request = json.loads(prompt.split('\nINPUT\n')[1])
    text = request['candidates'][0]['evidence'][0]['excerpt']
    def span(value):
        start = text.index(value)
        return dict(start=start, end=start + len(value), text=value)
    claim = dict(evidence_id='source', subject='project', relation='owner', value='Taylor',
                 subject_span=span('Project'), relation_span=span('owner'), value_span=span('Taylor'),
                 valid_from=None, valid_until=None, temporal_span=None, applicability='unbounded',
                 whole_fact_support='supported', condition_match='applies', whole_fact_span=span(text))
    return dict(request_fingerprint=request['request_fingerprint'], query_status='resolved',
                frame=dict(subject='project', relation='owner', at='2026-11-08'),
                facts=[dict(fact_id='fact', assessment='complete', claims=[claim])])


@pytest.mark.asyncio
async def test_complete_bound_reply_selects_and_checks_snapshot_three_times():
    checks = []
    async def provider(prompt):
        return json.dumps(reply(prompt))
    result = await produce(snapshot(), 'policy', provider, lambda: checks.append(True))
    assert result.status == 'assessed'
    assert result.decision.fact_ids == ('fact',)
    assert len(checks) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'foreign', 'extra', 'partial', 'condition', 'quote', 'stale'])
async def test_bad_or_unresolved_reply_never_selects(mutation):
    async def provider(prompt):
        data = reply(prompt)
        if mutation == 'missing': data['facts'] = []
        if mutation == 'duplicate': data['facts'] *= 2
        if mutation == 'foreign': data['facts'][0]['claims'][0]['evidence_id'] = 'foreign'
        if mutation == 'extra': data['label'] = 'expected-answer'
        if mutation == 'partial': data['facts'][0]['claims'][0]['whole_fact_support'] = 'unresolved'
        if mutation == 'condition': data['facts'][0]['claims'][0]['condition_match'] = 'unresolved'
        if mutation == 'quote': data['facts'][0]['claims'][0]['value_span']['text'] = 'Morgan'
        if mutation == 'stale': data['request_fingerprint'] = 'old'
        return json.dumps(data)
    result = await produce(snapshot(), 'policy', provider, lambda: None)
    assert result.status == 'unavailable'
    assert result.decision.fact_ids == ()


@pytest.mark.asyncio
async def test_snapshot_change_during_provider_discards_answer():
    changed = False
    async def provider(prompt):
        nonlocal changed
        changed = True
        return json.dumps(reply(prompt))
    def fence():
        if changed: raise RuntimeError('forgotten')
    result = await produce(snapshot(), 'policy', provider, fence)
    assert result.status == 'unavailable' and result.error_type == 'RuntimeError'


@pytest.mark.asyncio
async def test_provider_failure_is_unavailable_and_cancellation_propagates():
    async def failed(prompt):
        raise OSError('offline')
    result = await produce(snapshot(), 'policy', failed, lambda: None)
    assert result.status == 'unavailable' and result.error_type == 'OSError'
    async def cancelled(prompt):
        raise asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await produce(snapshot(), 'policy', cancelled, lambda: None)


def test_fingerprint_binds_query_policy_and_complete_fact_content():
    original = snapshot()
    fingerprints = {prepare(original, 'policy')[1], prepare(original, 'changed-policy')[1],
                    prepare(replace(original, query='Who formerly owned Project?'), 'policy')[1],
                    prepare(replace(original, candidates=(replace(original.candidates[0], content='changed'),)), 'policy')[1]}
    assert len(fingerprints) == 4


@pytest.mark.asyncio
async def test_oversized_input_rejected_before_provider():
    called = False
    async def provider(prompt):
        nonlocal called
        called = True
        return '{}'
    result = await produce(replace(snapshot(), query='x' * 64000), 'policy', provider, lambda: None)
    assert result.status == 'unavailable' and not called
