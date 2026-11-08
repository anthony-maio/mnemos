import json

import pytest

from benchmarks.local_memory.evaluation_journal import EvaluationJournal, read_journal
from benchmarks.local_memory.relation_inventory_run import run_inventory_screen
from tests.test_relation_inventory_run import cases, transport
from tests.test_relation_inventory_cache import IDENTITY


def test_exact_evidence_survives_close_and_prior_run_cannot_be_overwritten(tmp_path):
    path = tmp_path / 'events.jsonl'
    with EvaluationJournal(path) as journal:
        journal('request_start', {'prompt': 'first\nsecond', 'bytes': 12})
        journal('provider_reply', {'response': '{"quoted":"value"}'})
    before = path.read_bytes()
    rows = read_journal(path)
    assert [r['sequence'] for r in rows] == [0, 1]
    assert rows[0]['data']['prompt'] == 'first\nsecond'
    with pytest.raises(FileExistsError): EvaluationJournal(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize('damage', ['partial', 'modified', 'reordered'])
def test_detects_damage_without_rewriting(tmp_path, damage):
    path = tmp_path / 'events.jsonl'
    with EvaluationJournal(path) as journal:
        journal('one', {'value': 1})
        journal('two', {'value': 2})
    lines = path.read_bytes().splitlines(keepends=True)
    if damage == 'partial': lines[-1] = lines[-1][:-2]
    if damage == 'modified': lines[0] = lines[0].replace(b'"value":1', b'"value":9')
    if damage == 'reordered': lines.reverse()
    path.write_bytes(b''.join(lines))
    before = path.read_bytes()
    with pytest.raises(ValueError): read_journal(path)
    assert path.read_bytes() == before


def test_fsync_failure_poisoning_prevents_later_events(tmp_path, monkeypatch):
    import benchmarks.local_memory.evaluation_journal as module
    path = tmp_path / 'events.jsonl'
    def fail(fd): raise OSError('fsync failed')
    with EvaluationJournal(path) as journal:
        monkeypatch.setattr(module.os, 'fsync', fail)
        with pytest.raises(OSError): journal('request_start', {'prompt': 'one'})
        with pytest.raises(RuntimeError): journal('request_start', {'prompt': 'two'})
    assert b'two' not in path.read_bytes()


@pytest.mark.asyncio
async def test_harness_with_real_durable_journal(tmp_path):
    path = tmp_path / 'events.jsonl'
    with EvaluationJournal(path) as journal:
        summary = await run_inventory_screen(cases(), IDENTITY, IDENTITY, transport, journal, lambda: None)
    rows = read_journal(path)
    assert rows[-1]['kind'] == 'summary'
    assert rows[-1]['data'] == json.loads(json.dumps(summary))
    assert len([r for r in rows if r['kind'] == 'provider_reply']) == 5
