from pathlib import Path
from collections import Counter
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DEST = ROOT / 'docs/validation/2026-09-20/retrieval-fixture-audit-v1'
SOURCE = DEST / 'dataset-exact.jsonl'
DATASET_SHA = '036fa51ee13271a85a14a62a2fe4ef65d13eaeb290331fe325aa5bf32ce893fc'
REPORT_SHA = '9553827a7c5f8b14079607c7fb5b38a4c40457d8987ad41efac1b825a028b8de'

DEST.mkdir(parents=True, exist_ok=True)
raw = SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == DATASET_SHA
(DEST / 'dataset-exact.jsonl').write_bytes(raw)
with zipfile.ZipFile(ROOT / 'docs/validation/2026-09-19/retrieval-terminal-local-v1/retrieval-reports.zip') as archive:
    report_raw = archive.read('full.json')
assert hashlib.sha256(report_raw).hexdigest() == REPORT_SHA
report = json.loads(report_raw)
assert report['dataset']['sha256'] == DATASET_SHA
sessions = [row for row in map(json.loads, raw.splitlines()) if row.get('record_type') == 'session' and row['split'] == 'heldout']
assert len(sessions) == 200
canaries = []
explicit = []
for session in sessions:
    facts = {fact['fact_id']: fact for fact in session['facts']}
    for event in session['events']:
        if event['lifecycle_operation'] != 'remember':
            continue
        for fact_id in event['evidence_fact_ids']:
            fact = facts[fact_id]
            assert event['event_id'] in fact['evidence_event_ids']
            assert event['content'].endswith(fact['content'])
            assert event['content'] != fact['content']
            explicit.append({'session_id': session['session_id'], 'event_id': event['event_id'], 'fact_id': fact_id, 'event_content': event['content'], 'gold_content': fact['content'], 'prefix': event['content'][:-len(fact['content'])], 'exact_match': False})
    if not {'cross_profile', 'scope_canary'}.issubset(session['scenario_types']):
        continue
    captures = [event for event in session['events'] if event['lifecycle_operation'] == 'capture']
    assert len(captures) == 1
    capture = captures[0]
    assert all(capture.get('target_' + key) is None for key in ('user_profile_id', 'project_id', 'workspace_id'))
    queries = [query for query in session['queries'] if query['kind'] == 'answerable']
    if not queries:
        continue
    assert len(queries) == 1
    query = queries[0]
    assert query['target_user_profile_id'] == 't18-canary-profile'
    assert query['target_project_id'] != session['project_id']
    assert query['relevant_fact_ids'] and query['expected_answer'] is not None
    assert all(facts[identifier]['scope'] == 'project' for identifier in query['relevant_fact_ids'])
    observations = [row for row in report['raw_outputs'] if row['session_id'] == session['session_id']]
    assert len(observations) == 3
    evidence = []
    for observation in observations:
        assert observation['execution_status'] == 'complete'
        comparison = next(row for row in observation['raw_output']['retrieval_comparisons'] if row['query_id'] == query['query_id'])
        assert comparison['quality_kind'] == 'answerable'
        strategies = comparison['strategies']
        assert len(strategies) == 4
        assert all(row['quality_hit'] is False and row['result']['facts'] == [] for row in strategies.values())
        evidence.append({'repetition': observation['repetition'], 'all_four_strategies_empty': True, 'original_quality_hit': False})
    canaries.append({'session_id': session['session_id'], 'source_project_id': session['project_id'], 'source_event': capture, 'query': query, 'observations': evidence})
assert [row['session_id'] for row in canaries] == [f'heldout-{number:03d}' for number in range(61, 81)]
assert len(explicit) == 60
prefix_counts = Counter(row['prefix'] for row in explicit)
assert dict(prefix_counts) == {'Please remember this project decision explicitly: ': 20, 'Remember for this project: ': 20, 'Remember temporarily: ': 20}
result = {'dataset_sha256': DATASET_SHA, 'report_sha256': REPORT_SHA, 'source_commit': 'fc72cc251aeafa161a1a390269b8dfa34d83cf2b', 'heldout_sessions': len(sessions), 'cross_scope_answerable_conflicts': len(canaries), 'complete_canary_observations': sum(len(row['observations']) for row in canaries), 'explicit_remember_gold_mismatches': len(explicit), 'prefix_counts': dict(prefix_counts), 'scope_canaries': canaries, 'explicit_remember': explicit, 'limits': ['Read-only audit of previously observed synthetic fixtures.', 'Original labels, thresholds, model outputs, and verdicts are unchanged.', 'Literal inclusion and source association do not establish semantic correctness.', 'No corrected accuracy or unseen heldout claim is made.']}
(DEST / 'fixture-audit.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
print(json.dumps({key: result[key] for key in ('heldout_sessions', 'cross_scope_answerable_conflicts', 'complete_canary_observations', 'explicit_remember_gold_mismatches', 'prefix_counts')}, indent=2))
