from pathlib import Path
import sys,json,hashlib,os,subprocess
from datetime import datetime,timezone
r=Path(sys.argv[1]);sys.path.insert(0,str(r/'source'))
from benchmarks.local_memory.heldout_run import run_heldout
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((r/'preparation.json').read_text())
for name,digest in p['source_files'].items():assert sha(r/'source'/name)==digest,name
assert sha(r/'corpus.jsonl')==p['corpus_sha256']
from benchmarks.local_memory.heldout_run import _live_model_metadata
models=_live_model_metadata(p['endpoint'],(p['model'],p['embedding_model']))
for name,item in p['models'].items():assert models[name]['digest']==item['digest']
state={'status':'running','pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),'source_commit':p['source_commit'],'preparation_sha256':sha(r/'preparation.json'),'runner_sha256':sha(Path(__file__))}
def save():(r/'status.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 hardware=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True)
 (r/'hardware.json').write_text(json.dumps({'exit_code':hardware.returncode,'stdout':hardware.stdout,'stderr':hardware.stderr},indent=2)+'\n')
 reports=run_heldout(dataset_path=r/'corpus.jsonl',output_dir=r/'reports',work_dir=r/'work',endpoint=p['endpoint'],model=p['model'],embedding_model=p['embedding_model'],repetitions=3,root=r/'source',think=False,temperature=0.0,max_tokens=8192,timeout_seconds=180.0,extraction_mode='source_bound',run_namespace='source-bound-fresh-v7',experiment_binding={'preparation_sha256':sha(r/'preparation.json'),'source_commit':p['source_commit'],'development_result_sha256':p['development_result_sha256']})
 state['reports']={name:report['status'] for name,report in reports.items()};state['status']='completed'
except BaseException as exc:
 state.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
finally:
 state['finished_at']=datetime.now(timezone.utc).isoformat();save()
