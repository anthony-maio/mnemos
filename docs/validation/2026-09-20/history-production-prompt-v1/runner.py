"""Development eligibility clarification comparison using actual proposal contract."""
from pathlib import Path
from datetime import datetime, timezone
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from urllib.request import urlopen

root = Path(__file__).resolve().parent
inputs = json.loads((root / 'inputs.json').read_bytes())
prep = json.loads((root / 'preparation.json').read_bytes())
snapshot = Path(inputs['snapshot'])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    for name, digest in prep['files'].items():
        assert sha(root / name) == digest
    for name, digest in prep['source_files'].items():
        assert sha(snapshot / name) == digest
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=snapshot, text=True).strip() == prep['source_commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=snapshot).strip()
    with urlopen(inputs['endpoint'] + '/api/tags', timeout=10) as response:
        model = next(m for m in json.loads(response.read())['models'] if m['name'] == inputs['model'])
    assert model['digest'] == prep['model']['digest']
verify()
sys.path.insert(0, str(snapshot))
from mnemos.modules.sleep import SleepDaemon
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.storage import InMemoryStore
from mnemos.utils.llm import LLMProvider, OllamaProvider

class Provider(LLMProvider):
    supports_structured_output = True
    def __init__(self, case, arm):
        self.case, self.arm = case, arm
        self.trace = {'response_metadata': []}
        from mnemos.runtime import build_llm_from_settings
        from mnemos.settings import AppSettings
        settings = AppSettings.model_validate({
            'llm': {'provider': 'ollama', 'model': inputs['model']},
            'providers': {'ollama': {'base_url': inputs['endpoint'], 'think': inputs['think'],
                'temperature': inputs['temperature'], 'max_tokens': inputs['max_tokens'], 'timeout': inputs['timeout']}}})
        self.provider = build_llm_from_settings(settings)
        assert isinstance(self.provider, OllamaProvider)
        self.provider.response_observer = self.trace['response_metadata'].append
        self.trace['provider_configuration'] = settings.providers.ollama.model_dump()
    async def predict(self, prompt, **kwargs):
        anchor = 'Do not invent facts or IDs. '
        assert prompt.count(anchor) == 1
        clause = inputs['candidate_clause'] + (inputs['eligibility_clause'] if self.arm == 'history_and_eligibility' else '')
        changed = prompt.replace(anchor, clause + anchor, 1)
        self.trace.update(prompt=changed, options=kwargs)
        if self.arm == 'history_only' and 'reference_request' in self.case:
            reference = self.case['reference_request']
            assert changed == reference['prompt'], 'not exact prior service prompt'
            assert kwargs == reference['options'], 'not exact prior service schema'
        result = await self.provider.predict(changed, **kwargs)
        self.trace['response'] = result
        return result
    async def classify(self, prompt, labels):
        raise AssertionError('No classifier in proposal diagnostic')

async def main():
    assert not (root / 'status.json').exists()
    state = {'status': 'running', 'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
             'preparation_sha256': sha(root / 'preparation.json'), 'processed': 0, 'errors': 0}
    def save():
        (root / 'status.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    save()
    for repetition in range(inputs['repetitions']):
        for index, case in enumerate(inputs['cases']):
            arms = inputs['arms'] if (index + repetition) % 2 == 0 else list(reversed(inputs['arms']))
            for arm in arms:
                provider = Provider(case, arm)
                row = {'case_id': case['id'], 'arm': arm, 'repetition': repetition, 'trace': provider.trace}
                try:
                    existing = [MemoryChunk.model_validate(case['existing_fact'])]
                    proposals = await SleepDaemon(InMemoryStore()).propose_facts(
                        [EpisodicEvent.model_validate(case['source'])], provider, existing_facts=existing)
                    row.update(status='runtime_valid', proposals=[p.model_dump(mode='json') for p in proposals])
                except Exception as exc:
                    row.update(status='error', error_type=type(exc).__name__, error=str(exc))
                    state['errors'] += 1
                with (root / 'observations.jsonl').open('a', encoding='utf-8') as output:
                    output.write(json.dumps(row) + '\n')
                state['processed'] += 1
                save()
                print(json.dumps({k: row[k] for k in ('case_id', 'arm', 'repetition', 'status')}), flush=True)
    verify()
    state.update(status='completed', finished_at=datetime.now(timezone.utc).isoformat(), observations_sha256=sha(root / 'observations.jsonl'))
    save()
asyncio.run(main())
