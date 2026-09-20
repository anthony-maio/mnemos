"""Offline hash and semantic checks for one archived native outage attempt."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
index = json.loads((root / 'evidence-index.json').read_bytes())
for name, expected in index['files'].items():
    data = (root / name).read_bytes()
    assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'], name
sys.path.insert(0, str(root / 'sources'))
v = index['version']
m = importlib.import_module(f'native_hosts.claude_embedding_outage_v{v}')
r = json.loads((root / 'report.json').read_bytes())
native = [json.loads(line) for line in (root / 'native-projection.jsonl').read_bytes().splitlines()]
result = m.require_native_retrieval(native, submitted_prompt=m.PROMPT,
    target_content=m.TARGET_CONTENT, expected_model='claude-sonnet-4-6', expected_session_id=r['native_process']['session_id'])
d = json.loads((root / 'dispatch-records.json').read_bytes())
matched = [x for x in d if x['operation'] == 'retrieve' and x['payload'].get('query') == 'deployment marker']
assert len(matched) == 1
row = matched[0]
m.require_retrieval_response(row['result'], target_id=r['target_id'], target_content=m.TARGET_CONTENT,
    project_id=row['payload']['context']['project_id'], foreign_id=r['foreign_id'],
    foreign_content='Foreign deployment marker is not project memory.')
assert r['cleanup_established'] is True and r['embedding_responses_observed'] == 0
if v == 1:
    assert r['status'] == 'failed' and result['response'] != row['result']
    assert r['error'] == 'native result differs from recorded service response'
else:
    m.require_mcp_response_match(result['response'], row['result'])
    assert r['status'] == 'native_embedding_outage_lexical_retrieval_observed'
    assert r['final_seed_snapshot'] == r['seed_snapshot']
    assert r['final_fact_count'] == 2 and r['final_integrity'] == 'ok'
print(json.dumps({'verified_files': len(index['files']), 'status': r['status'], 'native_tool_result': 'degraded_lexical_retrieval'}))
