from pathlib import Path
import collections
import hashlib
import itertools
import json
import math
import sys
import zipfile

repo = Path(__file__).resolve().parents[1]
root = Path(sys.argv[1]).resolve()
archive = zipfile.ZipFile(root) if root.is_file() else None
read = lambda name: archive.read(name) if archive else (root / name).read_bytes()
report = json.loads(read('measurement.json'))
assert report['measurement_status'] == 'complete' and report['execution_complete']
assert report['measurement_inputs_match_head_before'] and report['measurement_inputs_match_head_after']
manifest = json.loads((repo / 'benchmarks/local_memory/datasets/source-association-development-v1.json').read_bytes())
vectors_path = repo / 'docs/validation/2026-09-20/source-association-role-vectors-v1/artifact.json'
assert hashlib.sha256(vectors_path.read_bytes()).hexdigest() == report['vector_artifact_sha256']
vectors = json.loads(vectors_path.read_bytes())['role_vectors']
cases = {case['id']: case for case in manifest['graph_cases']}
rows = report['rows']
arms = ('semantic', 'source_association', 'combined')
assert len(rows) == 90
assert {(r['embedding_contract'], r['case_id'], r['arm'], r['repetition']) for r in rows} == set(
    itertools.product(vectors, cases, arms, range(3)))
for name, expected in report['artifacts'].items():
    raw = read(name)
    assert len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256'], name

def cosine(a, b):
    return math.fsum(x*y for x, y in zip(a, b)) / math.sqrt(math.fsum(x*x for x in a)*math.fsum(y*y for y in b))

failures = collections.Counter()
target_rows = []
for row in rows:
    case = cases[row['case_id']]
    role = vectors[row['embedding_contract']]
    facts = {f['id']: f for f in case['facts']}
    allowed = set(facts)
    query = role['queries'][case['query']]
    pool = row['candidate_pool']
    assert set(pool['union_ids']) == set(pool['vector_ids']) | set(pool['lexical_ids'])
    assert set(pool['union_ids']) <= allowed
    assert row['target']['direct_candidate'] == (case['target_id'] in pool['union_ids'])
    assert row['target']['in_top_k'] == (case['target_id'] in row['returned_ids'])
    for score in row['candidate_scores']:
        assert score['id'] in allowed
        document = role['documents'][facts[score['id']]['content']]
        assert abs(score['semantic'] - cosine(query, document)) < 1e-6
        assert math.isclose(score['graph_contribution'], score['graph_energy']*0.1, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(score['total'], score['semantic']+score['graph_contribution'], rel_tol=1e-12, abs_tol=1e-12)
    ranked = sorted([s for s in row['candidate_scores'] if s['semantic'] >= 0.2],
                    key=lambda s: (-s['total'], facts[s['id']]['scope']['scope'] != 'project', s['id']))
    assert row['returned_ids'] == [s['id'] for s in ranked[:5]]
    for rank, score in enumerate(ranked, 1):
        assert score['rank'] == rank
    for edge in row['typed_provenance']:
        active = edge['components']
        assert active
        if row['arm'] != 'combined':
            assert active == [row['arm']]
        assert edge['effective_weight'] == max(edge[name] for name in active)
        left, right = edge['pair']
        assert row['edges'][left][right] == edge['effective_weight']
        if 'source_association' in active:
            assert edge['source_relations']
            for relation in edge['source_relations']:
                assert set(relation['fact_ids']) == set(edge['pair'])
                assert relation['source_event_id'] and len(relation['evidence_ids']) >= 2
    controls = row['control_observations']
    assert controls['edge_cut']['candidate_pool'] == pool
    for label, value in row['controls'].items():
        if value is False:
            failures[label] += 1
    if case.get('lifecycle_actions'):
        assert 'lifecycle' in controls
        lifecycle = controls['lifecycle']
        final = lifecycle['steps']['restart']
        correction, forget = case['lifecycle_actions'][:2]
        forgotten = next(f for f in final['facts'] if f['id'] == forget['fact_id'])
        assert forgotten['present'] is False
        assert forget['source_event_id'] not in {e['id'] for e in final['episodic_events']}
        assert forget['source_event_id'] in final['suppressions']['source_event_ids']
        corrected = next(f for f in final['facts'] if f['id'] == correction['fact_id'])
        assert corrected['version'] == 2 and corrected['content'] == correction['new_content']
    if row['repetition'] == 0:
        target_rows.append({'embedding_contract': row['embedding_contract'], 'case': case['id'],
                           'arm': row['arm'], 'seed': pool['actual_selected_seed_id'],
                           'target': row['target']})
assert report['controls_pass'] == (not failures)
assert len(report['failed_controls']) == sum(failures.values())
summary = {'status': 'verified', 'source_commit': report['source_commit'], 'rows': len(rows),
           'independent_cases': len(cases), 'coverage_pass': report['coverage_pass'],
           'control_failures': dict(failures), 'summary_by_embedding': report['summary_by_embedding'],
           'artifact_files': len(report['artifacts']),
           'measurement_sha256': hashlib.sha256(read('measurement.json')).hexdigest(),
           'unique_case_arm_targets': target_rows}
(repo / '.venv/source-association-role-verification.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({key: value for key, value in summary.items() if key != 'unique_case_arm_targets'}))
