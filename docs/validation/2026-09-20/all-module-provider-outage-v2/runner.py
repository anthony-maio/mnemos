"""Installed all-module outage/restart probe; synthetic native capture replay."""
from pathlib import Path
from datetime import datetime,timezone
import asyncio,hashlib,inspect,json,socket,sys,zipfile
from mnemos.client import MnemosClient
from mnemos.config import MnemosConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeContext,ScopeRegistry,ProjectIdentity
from mnemos.service import MnemosService
from mnemos.utils.embeddings import OllamaEmbeddingProvider
from mnemos.utils.llm import OllamaProvider
from mnemos.utils.storage import SQLiteStore

root=Path(sys.argv[1]); phase=sys.argv[2]
assert phase=='offline', 'Recovery is a separate phase, not an implicit retry.'
prep=json.loads((root/'preparation.json').read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in prep['files'].items(): assert sha(root/name)==digest,name
installation=json.loads(Path(prep['installation']).read_bytes())
assert installation['status']=='passed'
assert installation['bindings_after']['commit']==prep['source_commit']
wheel=Path(prep['wheel']);assert sha(wheel)==prep['wheel_sha256']
installed=Path(inspect.getfile(MnemosClient)).parents[1]
verified=0
with zipfile.ZipFile(wheel) as z:
 for name in z.namelist():
  if name.startswith('mnemos/') and name.endswith('.py'):
   assert (installed/name).read_bytes()==z.read(name),name
   verified+=1
entries=json.loads((root/'source-outbox.json').read_bytes())
assert len(entries)==2
context=ScopeContext.model_validate(entries[0]['payload']['context'])
assert all(x['payload']['context']==context.model_dump(mode='json') for x in entries)
registry=ScopeRegistry(profiles=[context.user_profile_id],projects={context.project_id:ProjectIdentity(id=context.project_id,root=str(root/'project'),user_profile_id=context.user_profile_id,enabled=True)})
config=MnemosConfig();assert all(config.modules.model_dump().values())
descriptor=root/'runtime.json';outbox=root/'outbox.json'
assert not (root/'offline-result.json').exists() and not (root/'memory.sqlite').exists()
outbox.write_bytes((root/'source-outbox.json').read_bytes())
# Reserve a local port without listening. Only this probe's provider targets it.
blocked=socket.socket();blocked.bind(('127.0.0.1',0))
endpoint='http://127.0.0.1:'+str(blocked.getsockname()[1])
instances=[]
def factory():
 memory=LocalMemoryService(MnemosEngine(config=config.model_copy(deep=True),llm=OllamaProvider(model=prep['model'],base_url=endpoint,timeout=2,temperature=0.0,max_tokens=8192,think=True),embedder=OllamaEmbeddingProvider(model=prep['embedding_model'],base_url=endpoint,timeout=2),store=SQLiteStore(str(root/'memory.sqlite')),owns_providers=True),registry)
 instances.append(memory);return memory

def snapshot():
 store=instances[-1].engine.store
 return {'events':[e.model_dump(mode='json') for e in store.list_episodic_events()],'jobs':[j.model_dump(mode='json') for j in store.list_jobs()],'facts':[f.model_dump(mode='json') for f in store.get_all()]}

async def main():
 result={'status':'running','pid':__import__('os').getpid(),'started_at':datetime.now(timezone.utc).isoformat(),'source_commit':prep['source_commit'],'verified_modules':verified,'module_flags':config.modules.model_dump(),'config':config.model_dump(mode='json'),'offline_endpoint':endpoint,'steps':[],'limits':'Replay of previously captured native Codex events through actual authenticated loopback service and installed Ollama providers. Controlled connection refusal; no live host or real model success is claimed. All module flags enabled, but outage prevents downstream execution.'}
 service=None;client=None
 def save(): (root/'offline-result.json').write_text(json.dumps(result,indent=2)+'\n')
 save()
 try:
  service=MnemosService(dispatcher_factory=factory,descriptor_path=descriptor,dispatch_timeout=180)
  service.start();client=MnemosClient(descriptor_path=descriptor,outbox_path=outbox,auto_start=False,request_timeout=180)
  delivery=await client.replay_outbox();assert (delivery['acknowledged'],delivery['pending'])==(2,0),delivery
  captured=snapshot();assert len(captured['events'])==len(captured['jobs'])==2 and not captured['facts']
  expected={(x['payload']['role'],x['payload']['content']) for x in entries}
  assert {(e['role'],e['content']) for e in captured['events']}==expected
  response=await client.request('maintenance',{},queue_on_failure=False)
  failed=snapshot();result['state_after_outage']=failed;save();assert not failed['facts']
  assert all(j['status']=='pending' and 0 < j['attempt_count'] < j['max_attempts'] and j['last_error']=='ConnectionError' for j in failed['jobs']),failed['jobs']
  assert all(e['status']=='pending' for e in failed['events'])
  result['steps'].append({'phase':'provider_outage','delivery':delivery,'maintenance':response,'state':failed});save()
  await client.close();client=None;service.stop();service=None;assert not descriptor.exists()
  service=MnemosService(dispatcher_factory=factory,descriptor_path=descriptor,dispatch_timeout=180);service.start()
  reopened=snapshot();assert reopened==failed
  outbox.write_bytes((root/'source-outbox.json').read_bytes())
  client=MnemosClient(descriptor_path=descriptor,outbox_path=outbox,auto_start=False,request_timeout=180)
  replay=await client.replay_outbox();assert (replay['acknowledged'],replay['pending'])==(2,0)
  assert snapshot()==failed
  result['steps'].append({'phase':'restart_replay','replay':replay,'state':snapshot()})
  result['status']='offline_restart_passed_recovery_not_run'
 except Exception as exc:
  result.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
 finally:
  if client: await client.close()
  if service: service.stop()
  blocked.close()
  result['descriptor_removed']=not descriptor.exists();result['finished_at']=datetime.now(timezone.utc).isoformat();save()
asyncio.run(main())
