import asyncio

import pytest

from benchmarks.local_memory.packed_query_probe import build_cases
from benchmarks.local_memory.packed_query_probe_run import run_probe
from benchmarks.local_memory.evaluation_journal import EvaluationJournal, read_journal
from tests.test_packed_query_probe import IDENTITY


@pytest.mark.asyncio
async def test_completed_malformed_reply_is_preserved_without_retry(tmp_path):
    cases = build_cases(IDENTITY, IDENTITY)[:2]
    calls = []
    async def provider(prompt, schema):
        calls.append(prompt)
        return dict(done=True, response='"S|MA"' if len(calls) == 1 else cases[1].expected_text)
    path = tmp_path / 'journal.jsonl'
    with EvaluationJournal(path) as journal:
        observations = await run_probe(tuple((c.key, c.request) for c in cases), provider, journal, lambda: None)
    assert len(calls) == 2
    assert [o['protocol_valid'] for o in observations] == [False, True]
    records = read_journal(path)
    assert records[-1]['kind'] == 'run_complete'
    assert [r['data']['wire']['response'] for r in records if r['kind'] == 'provider_reply'] == [
        '"S|MA"', cases[1].expected_text]
    assert all('expected_text' not in r['data'] for r in records)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['exception', 'incomplete', 'cancel', 'journal'])
async def test_uncertain_or_unrecordable_call_never_advances(failure):
    cases = build_cases(IDENTITY, IDENTITY)[:2]
    calls, events = [], []
    async def provider(prompt, schema):
        calls.append(prompt)
        if failure == 'exception': raise TimeoutError('uncertain')
        if failure == 'cancel': raise asyncio.CancelledError
        return dict(done=failure != 'incomplete', response=cases[0].expected_text)
    def emit(kind, data):
        if failure == 'journal' and kind == 'provider_reply': raise OSError('disk full')
        events.append(kind)
    args = (tuple((c.key, c.request) for c in cases), provider, emit, lambda: None)
    if failure in {'cancel', 'journal'}:
        with pytest.raises(asyncio.CancelledError if failure == 'cancel' else OSError):
            await run_probe(*args)
    else:
        assert await run_probe(*args) == []
        assert events[-1] == 'run_stopped'
    assert len(calls) == 1 and 'run_complete' not in events
