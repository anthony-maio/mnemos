"""Execute the prospectively bound v4 matrix once in isolated directories."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

preparation_path = Path(sys.argv[1]).resolve()
preparation = json.loads(preparation_path.read_bytes())
snapshot = Path(preparation['snapshot']).resolve()
output_root = preparation_path.parent
if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != preparation['runner_sha256']:
    raise SystemExit('Runner bytes differ from preparation')
if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=snapshot, text=True).strip() != preparation['source_commit']:
    raise SystemExit('Source commit differs from preparation')
if subprocess.check_output(['git', 'status', '--porcelain'], cwd=snapshot).strip():
    raise SystemExit('Frozen source is not clean')
sys.path.insert(0, str(snapshot))
from benchmarks.local_memory.dev_smoke import _code_provenance
from benchmarks.local_memory.heldout_run import _live_model_metadata, run_heldout

code_sha, code_files = _code_provenance(snapshot)
if code_sha != preparation['code_sha256'] or code_files != preparation['code_files']:
    raise SystemExit('Code provenance differs from preparation')
dataset = snapshot / preparation['dataset']['path']
protocol = snapshot / preparation['protocol']['path']
for path, binding in ((dataset, preparation['dataset']), (protocol, preparation['protocol'])):
    if hashlib.sha256(path.read_bytes()).hexdigest() != binding['sha256']:
        raise SystemExit('Input bytes differ from preparation: ' + str(path))
models = _live_model_metadata(preparation['endpoint'], tuple(preparation['models']))
for name, expected in preparation['models'].items():
    if models[name]['digest'] != expected['digest']:
        raise SystemExit('Live model digest differs from preparation: ' + name)
output_dir = output_root / 'reports'
work_dir = output_root / 'work'
if output_dir.exists() or work_dir.exists() or (output_root / 'status.json').exists():
    raise SystemExit('Execution directories must be new; preserve existing attempts')
binding = {
    'protocol_version': 'contract-v4-evaluation-protocol-v1',
    'protocol_sha256': preparation['protocol']['sha256'],
    'dataset_sha256': preparation['dataset']['sha256'],
    'source_commit': preparation['source_commit'],
    'preparation_sha256': hashlib.sha256(preparation_path.read_bytes()).hexdigest(),
    'previously_observed_synthetic_repair': True,
}
status = {'status': 'running', 'pid': os.getpid(), 'source_commit': preparation['source_commit'], 'binding': binding, 'started_at': datetime.now(timezone.utc).isoformat()}

def save() -> None:
    pending = output_root / 'status.pending.json'
    pending.write_bytes((json.dumps(status, indent=2) + '\n').encode())
    pending.replace(output_root / 'status.json')

save()
print(json.dumps({'status': 'running', 'pid': os.getpid(), 'source_commit': preparation['source_commit']}), flush=True)
try:
    reports = run_heldout(
        dataset_path=dataset,
        output_dir=output_dir,
        work_dir=work_dir,
        endpoint=preparation['endpoint'],
        model='qwen3.5:9b',
        embedding_model='nomic-embed-text:latest',
        repetitions=3,
        root=snapshot,
        think=False,
        run_namespace='t18-contract-v4-matrix-v1',
        experiment_binding=binding,
    )
    status['profiles'] = {name: report['status'] for name, report in reports.items()}
    status['status'] = 'completed' if len(reports) == 7 and all(report['status'] == 'completed' for report in reports.values()) else 'incomplete'
except BaseException as error:
    status.update(status='failed', error_type=type(error).__name__, error=str(error))
    raise
finally:
    status['finished_at'] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(status), flush=True)
raise SystemExit(0 if status['status'] == 'completed' else 1)
