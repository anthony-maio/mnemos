"""One fixed readiness request, separate from tool-workflow development cases."""
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

root = Path('C:/Users/Ant/AppData/Local/Temp/mnemos-model-readiness-v1')
out = root / 'load-readiness'
out.mkdir(exist_ok=False)
expected = '17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7'
request = {'model': 'gpt-oss:20b', 'stream': False, 'think': 'low',
           'messages': [{'role': 'user', 'content': 'Reply with READY.'}],
           'options': {'temperature': 0, 'seed': 42, 'num_ctx': 32768, 'num_predict': 128}}
def save(name, value):
    (out / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
state = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
         'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'kind': 'load readiness only; no tools or memory cases'}
save('state.json', state)
try:
    with httpx.Client(base_url='http://127.0.0.1:11434', trust_env=False, timeout=180) as client:
        tags = client.get('/api/tags'); tags.raise_for_status(); save('tags.json', tags.json())
        model = next(m for m in tags.json()['models'] if m['name'] == request['model'])
        assert model['digest'] == expected
        save('request.json', request)
        start = time.monotonic()
        response = client.post('/api/chat', json=request)
        save('response.json', {'status_code': response.status_code, 'body': response.json(),
                              'elapsed_seconds': time.monotonic() - start})
        response.raise_for_status()
        body = response.json()
        assert body.get('done') is True
        assert body.get('message', {}).get('content', '').strip() == 'READY'
        ps = client.get('/api/ps'); ps.raise_for_status(); save('ps.json', ps.json())
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        save('nvidia-smi.json', {'returncode': gpu.returncode, 'stdout': gpu.stdout, 'stderr': gpu.stderr})
        state['status'] = 'passed'
except Exception as error:
    state['status'] = 'failed'
    state['error'] = repr(error)
finally:
    state['finished_at'] = datetime.now(timezone.utc).isoformat()
    save('state.json', state)
print(json.dumps(state), flush=True)
raise SystemExit(0 if state['status'] == 'passed' else 1)
