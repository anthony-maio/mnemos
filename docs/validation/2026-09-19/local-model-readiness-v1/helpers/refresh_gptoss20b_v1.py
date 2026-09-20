"""Refresh the authorized local model after preserving its prior tag and digest."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import httpx

root = Path('C:/Users/Ant/AppData/Local/Temp/mnemos-model-readiness-v1')
state_path = root / 'pull-state.json'
assert not state_path.exists()
state = {'status': 'running', 'pid': os.getpid(), 'model': 'gpt-oss:20b',
         'started_at': datetime.now(timezone.utc).isoformat(),
         'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

def save():
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    temporary = root / 'pull-state.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    temporary.replace(state_path)

save()
try:
    with httpx.Client(trust_env=False, timeout=httpx.Timeout(120, connect=10)) as client:
        with client.stream('POST', 'http://127.0.0.1:11434/api/pull', json={'model':'gpt-oss:20b','stream':True}) as response:
            response.raise_for_status()
            with (root / 'pull-frames.jsonl').open('x', encoding='utf-8') as log:
                last_print = None
                final = None
                for line in response.iter_lines():
                    if not line:
                        continue
                    log.write(line + '\n')
                    log.flush()
                    row = json.loads(line)
                    state['last_frame'] = row
                    save()
                    if row.get('error'):
                        raise RuntimeError(row['error'])
                    progress_key = (row.get('status'), row.get('completed', 0) // 1_000_000_000)
                    if progress_key != last_print:
                        print(json.dumps(row), flush=True)
                        last_print = progress_key
                    final = row
                if not final or final.get('status') != 'success':
                    raise RuntimeError('Model pull ended without an explicit success frame')
        for operation, payload in (('tags', None), ('show', {'model':'gpt-oss:20b'})):
            url = 'http://127.0.0.1:11434/api/' + operation
            response = client.get(url) if payload is None else client.post(url, json=payload)
            (root / (operation + '-after-pull.json')).write_text(json.dumps({'status_code':response.status_code,'body':response.json()},indent=2)+'\n',encoding='utf-8')
            response.raise_for_status()
        state['status'] = 'passed'
except Exception as error:
    state['status'] = 'failed'
    state['error'] = repr(error)
finally:
    state['finished_at'] = datetime.now(timezone.utc).isoformat()
    save()
print(json.dumps(state), flush=True)
raise SystemExit(0 if state['status'] == 'passed' else 1)
