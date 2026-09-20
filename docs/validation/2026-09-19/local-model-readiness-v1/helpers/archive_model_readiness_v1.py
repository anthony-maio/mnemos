import hashlib
import json
import shutil
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
source = Path('C:/Users/Ant/AppData/Local/Temp/mnemos-model-readiness-v1')
out = repo / 'docs/validation/2026-09-19/local-model-readiness-v1'
out.mkdir(exist_ok=False)
shutil.copytree(source, out / 'raw')
(out / 'helpers').mkdir()
for name in ('refresh_gptoss20b_v1.py', 'model_load_readiness_v1.py', 'archive_model_readiness_v1.py'):
    shutil.copyfile(repo / '.venv' / name, out / 'helpers' / name)
profiles = {
    'schema_version': 1,
    'kind': 'prospective development profiles; not native-host acceptance',
    'endpoint': 'http://127.0.0.1:11434',
    'ollama_version': '0.34.1',
    'runtime_commit': 'e454a85f7b4f79ce2b483f738d68d88a15bd72d6',
    'wheel_sha256': '94011300cda669ddcabefe9e9077af0f2aa01b6ed5f23bc13c8e6b7f2a5fbf35',
    'profiles': [
        {'order': 1, 'model': 'qwen3.5:9b', 'digest': '6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7', 'think': False},
        {'order': 2, 'model': 'gpt-oss:20b', 'digest': '17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7', 'think': 'low'},
    ],
    'options': {'temperature': 0, 'seed': 42, 'presence_penalty': 0, 'top_k': 20, 'top_p': 0.95, 'num_ctx': 32768, 'num_predict': 2048},
    'case_count_per_model': 3,
    'repetitions': 1,
    'readiness_interpretation': 'Pull helper timed out during checksum verification; later tags/show independently establish refreshed model. Load request returned HTTP 200, done true, READY.; exact READY assertion failed. Subsequent ps reports full model size in VRAM. No development case has run at profile creation.'
}
(repo / 'docs/plans/2026-09-19-local-model-forget-profiles-v1.json').write_text(json.dumps(profiles, indent=2) + '\n', encoding='utf-8')
(out / 'README.md').write_text('''# Local model readiness, September 19, 2026

The old `gpt-oss:20b` tag could be listed but `/api/show` failed with HTTP 500
and a GGUF tensor-size overflow. Before refreshing it, the old manifest was
copied to `mnemos-gptoss20b-before-refresh:20260919-v1`; the copied digest
matches the old `aa4295ac...` digest in the preserved API response.

The refresh downloaded a different model blob. Its client stopped with a
read timeout while Ollama was verifying SHA-256. This failed client record
is unchanged. A later read-only tags/show check found the new manifest
`17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7`
and HTTP 200 metadata. The server completed after the client stopped; no
second pull was launched.

A single separate load request used low reasoning, temperature 0, seed 42,
32,768 context, and a 128-token output cap. It returned HTTP 200, `done: true`,
and `READY.` in 37.06 seconds, including reported load time of 19.16 seconds.
The helper's exact `READY` assertion rejected the period, so its terminal
status remains failed. This is a helper assertion failure, not a provider
load failure. No replacement request was run. A subsequent read-only
`/api/ps` response reports 12,765,573,938 model bytes, all in VRAM, and a
32,768-token context. `nvidia-smi` identified the RTX 4080 with 16,376 MiB
total and 15,526 MiB used at that observation. This is a load observation,
not a sustained utilization or tool-quality result.

The raw folder retains successful and failed calls, the streaming pull
frames, the fixed readiness request and response, and follow-up reads.
The refresh, load, and archive helpers are preserved. The three small
read-only API observation commands were run interactively; their raw
responses are present, but they have no standalone helper artifact.
Times in JSON are UTC (September 20); the folder date follows local time.

The [development protocol](../../../plans/2026-09-19-local-model-forget-development-v1.md)
and [profiles](../../../plans/2026-09-19-local-model-forget-profiles-v1.json)
are separate prospective inputs. No native-host, forgetting, or module
acceptance follows from this readiness check. No live host configuration
or memory database was changed.
''', encoding='utf-8')
items=[]
for path in sorted(out.rglob('*')):
    if path.is_file():
        items.append({'path': path.relative_to(out).as_posix(), 'bytes': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
(out / 'evidence-index.json').write_text(json.dumps({'files': items}, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'files': len(items), 'archive': str(out)}))
