"""One frozen-source real-provider continuation of the preserved outage database."""
from pathlib import Path
from datetime import datetime,timezone
from urllib.request import urlopen
import asyncio,hashlib,inspect,json,os,sys,zipfile
sys.path.insert(0, str(Path(sys.argv[1]) / "source"))
from mnemos.client import MnemosClient
from mnemos.config import MnemosConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeContext,ScopeRegistry,ProjectIdentity
from mnemos.service import MnemosService
from mnemos.utils.embeddings import OllamaEmbeddingProvider
from mnemos.utils.llm import OllamaProvider
from mnemos.utils.storage import SQLiteStore

root=Path(sys.argv[1]);prep=json.loads((root/'preparation.json').read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in prep['files'].items(): assert sha(root/name)==digest,name
prior=json.loads(Path(prep['prior_result']).read_bytes())
assert prior['status']=='offline_restart_passed_recovery_not_run'
assert sha(root/'offline-result.json') == prep['prior_result_sha256']
installed=Path(inspect.getfile(MnemosClient)).parents[1]
assert installed.resolve() == (root/'source').resolve()
for name,digest in prep['source_files'].items(): assert sha(installed/name)==digest,name
with urlopen(prep['endpoint']+'/api/tags',timeout=10) as f: models={m['name']:m for m in json.loads(f.read())['models']}
for name,metadata in prep['models'].items(): assert models[name]['digest']==metadata['digest']
entries=json.loads((root/'source-outbox.json').read_bytes())
context=ScopeContext.model_validate(entries[0]['payload']['context'])
foreign=context.model_copy(update={'project_id':'outage-foreign-project','session_id':'outage-foreign-session'})
registry=ScopeRegistry(profiles=[context.user_profile_id],projects={c.project_id:ProjectIdentity(id=c.project_id,root=str(root/c.project_id),user_profile_id=c.user_profile_id,enabled=True) for c in [context,foreign]})
config=MnemosConfig();config.sleep.extraction_mode='source_bound';assert config.model_dump(mode='json')==prep['config']
descriptor=root/'runtime.json';outbox=root/'outbox.json';instances=[]
def append(name,value):
 with (root/name).open('a',encoding='utf-8') as f:f.write(json.dumps(value,default=str)+'\n')
def factory():
 memory=LocalMemoryService(MnemosEngine(config=config.model_copy(deep=True),llm=OllamaProvider(model=prep['model'],base_url=prep['endpoint'],timeout=180,temperature=0.0,max_tokens=8192,think=False,response_observer=lambda x:append('llm.jsonl',x)),embedder=OllamaEmbeddingProvider(model=prep['embedding_model'],base_url=prep['endpoint'],timeout=180,response_observer=lambda x:append('embedding.jsonl',x)),store=SQLiteStore(str(root/'memory.sqlite')),owns_providers=True),registry)
 instances.append(memory);return memory

def snapshot():
 store=instances[-1].engine.store
 return {'events':[e.model_dump(mode='json') for e in store.list_episodic_events()],'jobs':[j.model_dump(mode='json') for j in store.list_jobs()],'facts':[f.model_dump(mode='json') for f in store.get_all()]}

async def main():
 assert not (root/'result.json').exists()
 result={'status':'running','pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),'source_commit':prep['source_commit'],'preparation_sha256':sha(root/'preparation.json'),'steps':[],'semantic_review':'NOT SCORED: preserving the marker alone does not establish extraction fidelity. Review all retained content against original source.','limits':'One development replay of saved native events through frozen-source service; not a new native session, ablation, unattended-worker recovery or pilot.'}
 service=None;client=None
 def save():(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 def start():
  runtime=MnemosService(dispatcher_factory=factory,descriptor_path=descriptor,dispatch_timeout=900)
  runtime.start()
  return runtime,MnemosClient(descriptor_path=descriptor,outbox_path=outbox,auto_start=False,request_timeout=900)
 async def call(op,payload):
  append('requests.jsonl',{'operation':op,'payload':payload})
  value=await client.request(op,payload,queue_on_failure=False)
  append('responses.jsonl',{'operation':op,'result':value})
  return value
 save()
 try:
  service,client=start()
  assert snapshot()==prior['steps'][-1]['state'],'Copied database differs from frozen offline state'
  value=await call('maintenance',{})
  recovered=snapshot();result['steps'].append({'phase':'real_provider_recovery','maintenance':value,'state':recovered});save()
  assert all(j['status']=='completed' for j in recovered['jobs']),'Recovery left pending or exhausted jobs'
  assert len(recovered['facts'])==1,'Expected one retained project marker; preserve abstention or extra facts as failure'
  target=recovered['facts'][0];fact_id=target['id'];assert 'cedar-window-618' in target['content']
  assert target['metadata']['scope_id']==context.project_id
  outbox.write_bytes((root/'source-outbox.json').read_bytes())
  replay=await client.replay_outbox();assert replay['acknowledged']==2 and replay['pending']==0
  assert snapshot()==recovered,'Replay changed completed source or fact state'
  query={'query':'What is this project outage test marker?','allowed_scopes':['project'],'profile':'biomimetic','top_k':5,'max_context_tokens':2000}
  recall=await call('retrieve',{**query,'context':context.model_dump(mode='json')})
  assert any(f['id']==fact_id for f in recall['facts'])
  foreign_recall=await call('retrieve',{**query,'context':foreign.model_dump(mode='json')})
  assert not foreign_recall['facts']
  result['steps'].append({'phase':'scoped_recall','recall':recall,'foreign':foreign_recall});save()
  correction=await call('correct',{'context':context.model_dump(mode='json'),'event_id':'recovery-explicit-correction','fact_id':fact_id,'expected_version':target['version'],'content':'For this disposable project, the outage test marker is birch-window-618.'})
  await call('maintenance',{})
  updated=instances[-1].engine.store.get(fact_id)
  assert updated is not None and updated.version==target['version']+1 and 'birch-window-618' in updated.content
  assert instances[-1].engine.store.get_job(correction['job_id']).status=='completed'
  result['steps'].append({'phase':'correction','fact':updated.model_dump(mode='json')});save()
  await client.close();client=None;service.stop();service=None
  service,client=start()
  persisted=instances[-1].engine.store.get(fact_id)
  assert persisted is not None and persisted.model_dump(mode='json')==updated.model_dump(mode='json')
  recall=await call('retrieve',{**query,'context':context.model_dump(mode='json')})
  assert any(f['id']==fact_id and 'birch-window-618' in f['content'] for f in recall['facts'])
  preview=await call('forget_preview',{'context':context.model_dump(mode='json'),'fact_id':fact_id,'expected_version':persisted.version,'limit':100})
  assert preview['complete'] and not preview['candidates']
  await call('forget_related',{'context':context.model_dump(mode='json'),**preview['apply_preconditions'],'selected_fact_ids':[fact_id],'selected_source_event_ids':[s['source_event_id'] for s in preview['source_events']]})
  outbox.write_bytes((root/'source-outbox.json').read_bytes());replay=await client.replay_outbox()
  assert replay['acknowledged']==2 and replay['pending']==0
  await call('maintenance',{})
  assert instances[-1].engine.store.get_all()==[]
  await client.close();client=None;service.stop();service=None
  service,client=start();assert instances[-1].engine.store.get_all()==[]
  result['steps'].append({'phase':'forget_restart_replay','state':snapshot(),'replay':replay})
  result['status']='bounded_lifecycle_passed_semantic_review_pending'
 except Exception as exc:
  result.update(status='failed',error_type=type(exc).__name__,error=str(exc))
  if instances:
   try:result['failure_state']=snapshot()
   except Exception:pass
  raise
 finally:
  if client:await client.close()
  if service:service.stop()
  result['descriptor_removed']=not descriptor.exists();result['finished_at']=datetime.now(timezone.utc).isoformat();save()
asyncio.run(main())
