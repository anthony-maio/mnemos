from pathlib import Path
import sys,json,hashlib,os
from datetime import datetime,timezone
from urllib.request import urlopen
r=Path(sys.argv[1]);sys.path.insert(0,str(r/'source'))
from benchmarks.local_memory.contracts import load_dataset
from benchmarks.local_memory.harness import DEFAULT_PROFILES
from benchmarks.local_memory.ollama_adapter import make_ollama_service_factory,OllamaEvaluatorAdapter,ollama_options_sha256
from benchmarks.local_memory.dev_smoke import _code_provenance
from mnemos.scopes import ScopeRegistry,ProjectIdentity
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((r/'preparation.json').read_text())
for name,digest in p['source_files'].items():assert sha(r/'source'/name)==digest,name
assert sha(r/'corpus.jsonl')==p['corpus_sha256']
with urlopen(p['endpoint']+'/api/tags',timeout=10) as f:models={m['name']:m for m in json.load(f)['models']}
for name,item in p['models'].items():assert models[name]['digest']==item['digest']
dataset=load_dataset(r/'corpus.jsonl');session=next(x for x in dataset.sessions if x.session_id==p['session_id']);assert session.split=='development'
projects={session.project_id:'t18-synthetic-profile'}
for x in (*session.events,*session.queries,*session.state_pairs):
 project=getattr(x,'target_project_id',None) or session.project_id
 profile=getattr(x,'target_user_profile_id',None) or 't18-synthetic-profile'
 assert project not in projects or projects[project]==profile
 projects[project]=profile
registry=ScopeRegistry(profiles=sorted(set(projects.values())),projects={k:ProjectIdentity(id=k,root=str(r/k),user_profile_id=v,enabled=True) for k,v in projects.items()})
code_hash,code_files=_code_provenance(r/'source');options_hash=ollama_options_sha256(p['options'])
state={'status':'running','pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),'source_commit':p['source_commit'],'preparation_sha256':sha(r/'preparation.json'),'runner_sha256':sha(Path(__file__)),'code_sha256':code_hash,'profiles':[]}
def save():(r/'status.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 for profile in DEFAULT_PROFILES:
  state['active_profile']=profile.name;save()
  factory=make_ollama_service_factory(r/(profile.name+'.sqlite'),registry,base_url=p['endpoint'],model=p['model'],embedding_model=p['embedding_model'],timeout=p['options']['timeout_seconds'],temperature=0.0,max_tokens=8192,think=False,extraction_mode='source_bound')
  adapter=OllamaEvaluatorAdapter(factory,base_url=p['endpoint'],model=p['model'],embedding_model=p['embedding_model'],real_model=True,model_revision=p['models'][p['model']]['digest'],embedding_revision=p['models'][p['embedding_model']]['digest'],model_sha256=p['models'][p['model']]['digest'],embedding_sha256=p['models'][p['embedding_model']]['digest'],options_sha256=options_hash,code_sha256=code_hash,model_provenance=p['models'][p['model']],embedding_provenance=p['models'][p['embedding_model']])
  observation=adapter.evaluate_session(session,profile,1)
  result={'session_split':session.split,'session_id':session.session_id,'profile':profile.as_dict(),'observation':observation.as_dict(),'preparation_sha256':sha(r/'preparation.json')}
  (r/(profile.name+'.json')).write_text(json.dumps(result,indent=2,default=str)+'\n')
  state['profiles'].append({'profile':profile.name,'execution_status':observation.execution_status,'artifact_sha256':sha(r/(profile.name+'.json'))});save()
 state['status']='completed'
except BaseException as exc:
 state.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
finally:
 state['finished_at']=datetime.now(timezone.utc).isoformat();save()
