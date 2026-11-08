from dataclasses import asdict
import json

import pytest

from benchmarks.local_memory.relational_support_producer import prepare
from benchmarks.local_memory.relational_support_run import run_screen
from benchmarks.local_memory.token_support_prompt import render_raw_prompt
from tests.test_relational_support_producer import snapshot, reply


def rows():
    item = snapshot()
    prompt, fingerprint = prepare(item, 'policy')
    base = dict(snapshot=json.loads(json.dumps(asdict(item), default=str)),
                policy_identity='policy', prompt=prompt, request_fingerprint=fingerprint,
                raw_prompt=render_raw_prompt(prompt), candidate_ids=['fact'], snapshot_fingerprint='snapshot')
    return [dict(key=key, **base) for key in ['a', 'b']]


@pytest.mark.asyncio
async def test_exact_frozen_requests_are_executed_once_and_raw_replies_saved():
    prepared = rows()
    calls, events = [], []
    async def transport(raw):
        calls.append(raw)
        return {'done': True, 'response': json.dumps(reply(prepared[0]['prompt']))}
    result = await run_screen(prepared, transport, lambda kind, value: events.append((kind, value)), lambda: None)
    assert result == {'status': 'completed', 'attempted': ['a', 'b'], 'unattempted': []}
    assert calls == [r['raw_prompt'] for r in prepared]
    assert sum(kind == 'provider_reply' for kind, _ in events) == 2
    observations = [value for kind, value in events if kind == 'observation']
    assert all(o['decision']['fact_ids'] == ('fact',) for o in observations)


@pytest.mark.asyncio
async def test_later_request_corruption_is_rejected_before_any_call():
    prepared = rows()
    prepared[1]['raw_prompt'] += 'changed'
    calls = []
    async def transport(raw):
        calls.append(raw)
        return {}
    with pytest.raises(ValueError, match='changed'):
        await run_screen(prepared, transport, lambda *args: None, lambda: None)
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['timeout', 'incomplete'])
async def test_uncertain_completion_stops_without_retry_or_next_request(failure):
    calls, events = [], []
    async def transport(raw):
        calls.append(raw)
        if failure == 'timeout': raise TimeoutError('uncertain')
        return {'done': False, 'response': 'partial'}
    result = await run_screen(rows(), transport, lambda kind, value: events.append((kind, value)), lambda: None)
    assert result == {'status': 'stopped_uncertain_completion', 'attempted': ['a'], 'unattempted': ['b']}
    assert len(calls) == 1
    assert events[-1][1]['status'] == 'unavailable'


@pytest.mark.asyncio
async def test_completed_malformed_reply_is_preserved_without_retry():
    calls, events = [], []
    async def transport(raw):
        calls.append(raw)
        return {'done': True, 'response': 'not json'}
    result = await run_screen(rows(), transport, lambda kind, value: events.append((kind, value)), lambda: None)
    assert result['status'] == 'completed' and len(calls) == 2
    observations = [value for kind, value in events if kind == 'observation']
    assert all(o['status'] == 'unavailable' and o['raw_response'] == 'not json' for o in observations)
