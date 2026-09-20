from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import time
from urllib.request import Request, urlopen

root = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
inputs = json.loads((root / 'inputs.json').read_bytes())
prep = json.loads((root / 'preparation.json').read_bytes())
for name, digest in prep['files'].items():
    assert sha(root / name) == digest
def verify_server():
    with urlopen(inputs['endpoint'] + '/api/version', timeout=10) as response:
        assert json.loads(response.read()) == inputs['server']
    with urlopen(inputs['endpoint'] + '/api/tags', timeout=10) as response:
        model = next(m for m in json.loads(response.read())['models'] if m['name'] == inputs['model']['name'])
    assert model['digest'] == inputs['model']['digest']
verify_server()
assert not (root / 'status.json').exists()
state = {'status': 'running', 'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
         'processed': 0, 'request_errors': 0, 'preparation_sha256': sha(root / 'preparation.json')}
def save():
    (root / 'status.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
save()
for i, request in enumerate(inputs['requests']):
    row = {'index': i, **{k: request[k] for k in ('case_id', 'route', 'think')}}
    start = time.monotonic()
    try:
        req = Request(inputs['endpoint'] + '/api/' + request['route'],
                      data=json.dumps(request['payload']).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=inputs['timeout']) as response:
            raw = response.read()
            row['http_status'] = response.status
        (root / f'response-{i:02d}.json').write_bytes(raw)
        data = json.loads(raw)
        final = data.get('response') if request['route'] == 'generate' else data.get('message', {}).get('content')
        thinking = data.get('thinking') if request['route'] == 'generate' else data.get('message', {}).get('thinking')
        row.update(status='response_received', final_nonempty=isinstance(final, str) and bool(final.strip()),
                   thinking_nonempty=isinstance(thinking, str) and bool(thinking.strip()),
                   done=data.get('done'), done_reason=data.get('done_reason'), eval_count=data.get('eval_count'),
                   response_sha256=hashlib.sha256(raw).hexdigest())
        if row['final_nonempty']:
            try:
                parsed = json.loads(final)
                row['final_json'] = parsed
            except ValueError:
                row['final_json_parse_error'] = True
    except Exception as exc:
        row.update(status='request_error', error_type=type(exc).__name__, error=str(exc))
        state['request_errors'] += 1
    row['elapsed_seconds'] = time.monotonic() - start
    with (root / 'observations.jsonl').open('a', encoding='utf-8') as output:
        output.write(json.dumps(row) + '\n')
    state['processed'] += 1
    save()
    print(json.dumps({k: v for k, v in row.items() if k != 'final_json'}), flush=True)
verify_server()
state.update(status='completed', finished_at=datetime.now(timezone.utc).isoformat(), observations_sha256=sha(root / 'observations.jsonl'))
save()
