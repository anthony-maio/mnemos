from pathlib import Path
from collections import Counter
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parents[4]
folder = root / 'docs/validation/2026-09-19/retrieval-terminal-local-v1'
hashes = json.loads((folder / 'zip-members.json').read_bytes())
with zipfile.ZipFile(folder / 'retrieval-reports.zip') as archive:
    raw = archive.read('full.json')
    assert hashlib.sha256(raw).hexdigest() == hashes['full.json']['sha256']
    data = json.loads(raw)
dataset_raw = (Path(__file__).parent / 'dataset-exact.jsonl').read_bytes()
assert hashlib.sha256(dataset_raw).hexdigest() == data['dataset']['sha256']
sessions = {row['session_id']: row for row in (json.loads(line) for line in dataset_raw.splitlines()) if row.get('record_type') == 'session'}
counts = Counter()
misses = []
for observation in data['raw_outputs']:
    session = sessions[observation['session_id']]
    for comparison in observation['raw_output']['retrieval_comparisons']:
        if comparison['quality_kind'] != 'answerable':
            continue
        strategy = comparison['strategies']['full']
        counts['all_answerable'] += 1
        if observation['execution_status'] != 'complete':
            counts['partial_answerable'] += 1
            continue
        counts['eligible_answerable'] += 1
        if strategy['quality_hit'] is True:
            counts['original_hits'] += 1
            continue
        assert strategy['quality_hit'] is False
        counts['original_misses'] += 1
        query = next(row for row in session['queries'] if row['query_id'] == comparison['query_id'])
        golds = [row for row in session['facts'] if row['fact_id'] in query['relevant_fact_ids']]
        evidence_ids = {value for gold in golds for value in gold['evidence_event_ids']}
        linked = {row['runtime_fact_id'] for row in observation['raw_output']['extraction_judgements'] if evidence_ids.intersection(row['observed_source_event_ids'])}
        returned = strategy['result']['facts']
        linked_returned = [row for row in returned if row['id'] in linked]
        spans = [row for row in returned if any(gold['content'].strip().casefold() in row['content'].strip().casefold() for gold in golds)]
        exact_events = [event for event in session['events'] if event['event_id'] in evidence_ids and event['lifecycle_operation'] == 'remember']
        copied = [row for row in linked_returned if any(row['content'] == event['content'] for event in exact_events)]
        counts['miss_source_linked_returned'] += bool(linked_returned)
        counts['miss_full_gold_literal_span_returned'] += bool(spans)
        counts['miss_explicit_source_copied_verbatim'] += bool(copied)
        misses.append({'session_id': session['session_id'], 'repetition': observation['repetition'], 'query_id': query['query_id'], 'gold_contents': [gold['content'] for gold in golds], 'linked_runtime_ids': sorted(linked), 'linked_returned': [{'id': row['id'], 'content': row['content'], 'scope': row.get('scope'), 'scope_id': row.get('scope_id')} for row in linked_returned], 'full_gold_literal_span_returned': bool(spans), 'explicit_source_copied_verbatim': bool(copied)})
output = {'report_sha256': hashes['full.json']['sha256'], 'dataset_sha256': hashlib.sha256(dataset_raw).hexdigest(), 'counts': dict(counts), 'misses': misses, 'limits': ['Literal inclusion does not establish semantic correctness.', 'All original misses and partial rows retain their original verdicts.', 'Source association alone does not establish correct scope or assertion.']}
destination = Path(__file__).parent / 'full-strategy-boundaries.json'
destination.write_bytes((json.dumps(output, indent=2)+'\n').encode())
print(json.dumps({'counts': dict(counts), 'first_misses': misses[:2], 'output': str(destination)}, indent=2))
