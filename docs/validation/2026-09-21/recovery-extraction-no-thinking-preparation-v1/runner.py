from pathlib import Path
from datetime import datetime,timezone
import asyncio,hashlib,json,os,sys
from urllib.request import urlopen
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prep=json.loads((root/'preparation.json').read_bytes()); inputs=json.loads((root/'inputs.json').read_bytes())
assert len(sys.argv)==2 and sys.argv[1] in ('prepare',sha(root/'preparation.json')),'Externally bound digest required'
for name,digest in prep['files'].items(): assert sha(root/name)==digest,name
sys.path.insert(0,str(root/'installed'))
from mnemos.modules.sleep import SleepDaemon
from mnemos.types import EpisodicEvent
from mnemos.utils.llm import OllamaProvider
class NoStore:
    def __getattr__(self,name): raise AssertionError('Provider-only diagnostic accessed store: '+name)
requests=[]
class Capture:
    supports_structured_output=True
    async def predict(self,prompt,**kwargs):
        requests.append({'prompt':prompt,'kwargs':kwargs})
        return '{"decision":"no_durable_facts","reason":"offline capture only","facts":[]}'
async def main():
    daemon=SleepDaemon(NoStore())
    events=[EpisodicEvent.model_validate(e) for e in inputs['events']]
    for event in events: await daemon.propose_facts([event],Capture(),existing_facts=[])
    if sys.argv[1]=='prepare':
        assert not (root/'requests.json').exists()
        (root/'requests.json').write_text(json.dumps(requests,indent=2)+'\n'); return
    assert requests==json.loads((root/'requests.json').read_bytes())
    with urlopen(inputs['endpoint']+'/api/tags',timeout=10) as r:
        model=next(m for m in json.loads(r.read())['models'] if m['name']==inputs['model'])
    assert model['digest']==prep['model']['digest']
    assert not (root/'result.json').exists()
    record={'status':'running','pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),'preparation_sha256':sha(root/'preparation.json'),'requests_sha256':sha(root/'requests.json'),'results':[]}
    def save(): (root/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    for event,request in zip(events,requests):
        metadata=[]
        provider=OllamaProvider(**{k:inputs[k] for k in ('model','timeout','temperature','max_tokens','think')},base_url=inputs['endpoint'],response_observer=metadata.append)
        row={'event_id':event.id,'role':event.role}
        class Bound:
            supports_structured_output=True
            async def predict(self,prompt,**kwargs):
                assert request=={'prompt':prompt,'kwargs':kwargs}
                raw=await provider.predict(prompt,**kwargs); row['final_response']=raw; return raw
        try:
            proposals=await daemon.propose_facts([event],Bound(),existing_facts=[])
            row.update(status='runtime_valid',proposals=[x.model_dump(mode='json') for x in proposals])
        except Exception as exc: row.update(status='error',error_type=type(exc).__name__,error=str(exc))
        with (root/'raw-outputs.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps({'event_id':event.id,'responses':metadata})+'\n')
        row['provider_counters']=[{k:m.get(k) for k in ('done_reason','eval_count','prompt_eval_count','total_duration')} for m in metadata]
        record['results'].append(row); save(); print(json.dumps({'role':event.role,'status':row['status']}),flush=True)
    for name,digest in prep['files'].items(): assert sha(root/name)==digest,name
    record.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat()); save()
asyncio.run(main())
