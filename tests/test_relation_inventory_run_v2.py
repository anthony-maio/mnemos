import asyncio
from dataclasses import replace
import json

import pytest

from benchmarks.local_memory.relation_inventory_run_v2 import InventoryCase, prepare_run, run_inventory_screen
from tests.test_relation_query import setup, reply
from tests.test_source_relation_producer import response
from tests.test_relation_inventory_cache import IDENTITY


def cases():
    snapshot, _, _ = setup()
    return (InventoryCase('a', snapshot), InventoryCase('b', replace(snapshot, query='Who is the owner?')))


async def transport(stage, prompt):
    data = response(prompt) if stage == 'source' else reply(prompt)
    if stage == 'source':
        claim = data['claims'][0]
        for name in ('temporal_quote', 'valid_from', 'valid_until', 'applicability'):
            del claim[name]
        claim['time'] = dict(meaning='present_assertion', quotation=claim['whole_fact_quote'],
                            valid_from=None, valid_until=None, assertion_date=None)
    else:
        data.update(time_intent='source_state', at=None)
    return dict(done=True, response=json.dumps(data))


@pytest.mark.asyncio
async def test_source_once_query_each_pass_and_full_journal():
    events = []
    result = await run_inventory_screen(cases(), IDENTITY, IDENTITY, transport,
                                       lambda kind, row: events.append((kind, row)), lambda: None)
    assert result['status'] == 'completed' and not result['unattempted_queries']
    starts = [r for k, r in events if k == 'request_start']
    assert [r['stage'] for r in starts] == ['source', 'query', 'query', 'query', 'query']
    assert all(r['prompt'] and r['request_sha256'] for r in starts)
    assert len([r for k, r in events if k == 'provider_reply']) == 5
    observations = [r for k, r in events if k == 'query_observation']
    assert len(observations) == 4
    assert all(r['case_elapsed_ms'] >= r['query_elapsed_ms'] >= 0 for r in observations)
    assert events[0][0] == 'manifest' and events[-1][0] == 'summary'


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['exception', 'incomplete'])
async def test_uncertain_source_completion_stops_without_retry(failure):
    calls = []
    async def failed(stage, prompt):
        calls.append(stage)
        if failure == 'exception': raise OSError('connection lost')
        return dict(done=False, response='partial')
    result = await run_inventory_screen(cases(), IDENTITY, IDENTITY, failed, lambda *args: None, lambda: None)
    assert calls == ['source'] and result['status'] == 'stopped_uncertain_completion'
    assert len(result['unattempted_queries']) == 4


@pytest.mark.asyncio
async def test_complete_malformed_source_is_not_retried_or_absent():
    calls, events = [], []
    async def invalid(stage, prompt):
        calls.append(stage)
        return dict(done=True, response='{}')
    result = await run_inventory_screen(cases(), IDENTITY, IDENTITY, invalid,
                                       lambda kind, row: events.append((kind, row)), lambda: None)
    assert result['status'] == 'completed' and calls == ['source']
    rows = [r for k, r in events if k == 'query_observation']
    assert len(rows) == 4 and all(r['decision']['status'] == 'unavailable' for r in rows)


@pytest.mark.asyncio
async def test_bad_later_case_blocks_all_inference():
    first, second = cases()
    candidate = second.snapshot.candidates[0]
    second = replace(second, snapshot=replace(second.snapshot,
                     candidates=(replace(candidate, content='x' * 64000),)))
    async def forbidden(stage, prompt): pytest.fail('Preflight must cover later inputs')
    with pytest.raises(ValueError):
        await run_inventory_screen((first, second), IDENTITY, IDENTITY, forbidden, lambda *args: None, lambda: None)


@pytest.mark.asyncio
async def test_journal_failure_is_not_swallowed_as_model_unavailability():
    calls = []
    def emit(kind, row):
        if kind == 'provider_reply': raise OSError('disk full')
    async def tracked(stage, prompt):
        calls.append(stage)
        return await transport(stage, prompt)
    with pytest.raises(OSError, match='disk full'):
        await run_inventory_screen(cases(), IDENTITY, IDENTITY, tracked, emit, lambda: None)
    assert calls == ['source']


@pytest.mark.asyncio
async def test_cancellation_propagates_without_next_request():
    calls = []
    async def cancelled(stage, prompt):
        calls.append(stage)
        raise asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await run_inventory_screen(cases(), IDENTITY, IDENTITY, cancelled, lambda *args: None, lambda: None)
    assert calls == ['source']


def test_manifest_binds_query_and_source_identities():
    original = prepare_run(cases(), IDENTITY, IDENTITY)['manifest_sha256']
    altered = replace(IDENTITY, policy_sha256='d' * 64)
    assert original != prepare_run(cases(), altered, IDENTITY)['manifest_sha256']
    assert original != prepare_run(cases(), IDENTITY, altered)['manifest_sha256']


@pytest.mark.asyncio
async def test_v2_support_basis_survives_durable_journal(tmp_path):
    from benchmarks.local_memory.evaluation_journal import EvaluationJournal, read_journal
    path = tmp_path / 'events.jsonl'
    with EvaluationJournal(path) as journal:
        await run_inventory_screen(cases(), IDENTITY, IDENTITY, transport, journal, lambda: None)
    observations = [r['data'] for r in read_journal(path) if r['kind'] == 'query_observation']
    assert len(observations) == 4
    for row in observations:
        assert row['decision']['status'] == 'selected'
        assert row['selected_bases'] == [dict(fact_id='fact', evidence_id='source', claim_index=0,
                                             basis='source_assertion')]
