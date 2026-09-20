"""Frozen real SQLite/service extraction lifecycle probe; no semantic auto-score."""
from pathlib import Path
from datetime import datetime, timezone
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import traceback
from urllib.request import urlopen

root = Path(__file__).resolve().parent
inputs = json.loads((root / 'inputs.json').read_bytes())
prep = json.loads((root / 'preparation.json').read_bytes())
snapshot = Path(inputs['snapshot'])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def verify():
    for name, digest in prep['files'].items():
        assert sha(root / name) == digest, name
    for name, digest in prep['source_files'].items():
        assert sha(snapshot / name) == digest, name
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=snapshot, text=True).strip() == prep['source_commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=snapshot).strip()
    with urlopen(inputs['endpoint'] + '/api/tags', timeout=10) as response:
        models = {m['name']: m for m in json.loads(response.read())['models']}
    for key, metadata in prep['models'].items():
        assert models[inputs[key]]['digest'] == metadata['digest']

verify()
sys.path.insert(0, str(snapshot))
from benchmarks.local_memory.ollama_adapter import make_ollama_service_factory
from benchmarks.local_memory.harness import ExperimentProfile
from mnemos.scopes import ScopeRegistry, ProjectIdentity
from mnemos.utils.llm import LLMProvider
from mnemos.contracts import CaptureEvent, RetrievalRequest, ForgetRelatedRequest
import mnemos.memory_service
assert Path(mnemos.memory_service.__file__).resolve().is_relative_to(snapshot)

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')

def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, default=str) + '\n')

class CandidateProvider(LLMProvider):
    supports_structured_output = True
    def __init__(self, provider, path):
        self.provider, self.path = provider, path
        assert provider.supports_structured_output
    async def predict(self, prompt, **kwargs):
        anchor = 'Do not invent facts or IDs. '
        assert prompt.count(anchor) == 1
        changed = prompt.replace(anchor, inputs['candidate_clause'] + anchor, 1)
        call = {'prompt': changed, 'baseline_prompt': prompt, 'options': kwargs}
        append(self.path, {'stage': 'request', **call})
        try:
            response = await self.provider.predict(changed, **kwargs)
        except Exception as exc:
            append(self.path, {'stage': 'error', 'type': type(exc).__name__, 'error': str(exc)})
            raise
        append(self.path, {'stage': 'response', 'response': response})
        return response
    async def classify(self, prompt, labels):
        raise AssertionError('Only sleep extraction is enabled')

profile = ExperimentProfile('history-service-sleep-only', ('sleep_consolidation',))

def construct(case, folder):
    # Fixed run_key intentionally reopens the same physical database.
    profile_id, project_id, foreign_id = 'profile-' + case['id'], 'project-' + case['id'], 'foreign-' + case['id']
    registry = ScopeRegistry(profiles=[profile_id], projects={
        project_id: ProjectIdentity(id=project_id, root=str(folder / 'project'), user_profile_id=profile_id, enabled=True),
        foreign_id: ProjectIdentity(id=foreign_id, root=str(folder / 'foreign'), user_profile_id=profile_id, enabled=True)})
    factory = make_ollama_service_factory(folder / 'memory.sqlite', registry,
        base_url=inputs['endpoint'], model=inputs['model'], embedding_model=inputs['embedding_model'],
        temperature=inputs['temperature'], max_tokens=inputs['max_tokens'], timeout=inputs['timeout'], think=inputs['think'])
    context = {'user_profile_id': profile_id, 'project_id': project_id, 'session_id': case['id'], 'host': 'manual'}
    foreign = {**context, 'project_id': foreign_id, 'session_id': 'foreign-' + case['id']}
    return factory, context, foreign

async def one(case, index):
    folder = root / case['id']
    folder.mkdir()
    factory, context, foreign = construct(case, folder)
    run_key = case['id']
    service = None
    record = {'case': case, 'operations': [], 'checks': {}, 'semantic_review': 'NOT SCORED; inspect committed text against frozen criterion'}
    def check(name, condition):
        record['checks'][name] = bool(condition)
        if not condition:
            raise AssertionError(name)
    def open_service():
        memory = factory(profile, run_key)
        memory.engine.llm._provider = CandidateProvider(memory.engine.llm._provider, folder / 'llm.jsonl')
        assert memory.engine.config.modules.model_dump() == inputs['modules']
        record.setdefault('instances', []).append({'database': str(memory.engine.store.db_path), 'config': memory.engine.config.model_dump(mode='json')})
        return memory
    async def call(op, payload):
        request = {'operation': op, 'payload': payload}
        append(folder / 'dispatch.jsonl', {'stage': 'request', **request})
        result = await service.dispatch(op, payload)
        append(folder / 'dispatch.jsonl', {'stage': 'response', 'operation': op, 'result': result})
        record['operations'].append({'operation': op, 'result': result})
        dump(folder / 'trace.json', service._benchmark_provider_trace)
        dump(folder / 'result.json', record)
        return result
    async def drain(receipts):
        await call('maintenance', {})
        jobs = [service.engine.store.get_job(r['job_id']) for r in receipts if r.get('job_id')]
        record.setdefault('job_snapshots', []).append([j.model_dump(mode='json') for j in jobs if j])
        check('jobs_completed_' + str(len(record['job_snapshots'])), all(j and j.status == 'completed' for j in jobs))
    def facts_for(receipt):
        return [f for f in service.engine.store.get_all()
                if any(e.source_event_id == receipt['event_id'] for e in service.engine.store.list_fact_evidence(f.id))]
    async def retrieve(query):
        response = await call('retrieve', {'query': query, 'context': context, 'allowed_scopes': ['project'], 'top_k': 5, 'max_context_tokens': 2000})
        check('retrieval_scoped_' + str(len(record['operations'])), all(f['scope_id'] == context['project_id'] and f['scope'] == 'project' for f in response['facts']))
        return response
    payload = {'event_id': 'source-' + case['id'], 'context': context, 'scope': 'project',
        'occurred_at': datetime(2026, 9, 20, 12, index + 1, tzinfo=timezone.utc).isoformat(),
        'role': case['role'], 'content': case['content'], 'source_kind': 'development_transcript', 'source_ref': 'fixture://' + case['id']}
    CaptureEvent.model_validate(payload)
    try:
        service = open_service()
        record['health'] = await call('health', {})
        controls = [await call('remember', {'event_id': 'control', 'context': context, 'content': inputs['control']}),
                    await call('remember', {'event_id': 'foreign-control', 'context': foreign, 'content': inputs['foreign_control']})]
        await drain(controls)
        control_ids = {f.id for f in service.engine.store.get_all()}
        check('two_controls_committed', len(control_ids) == 2)
        captured = await call('capture', payload)
        check('capture_committed', captured['status'] == 'committed' and bool(captured['job_id']))
        await drain([captured])
        source = service.engine.store.get_episodic_event(captured['event_id'])
        record['source'] = source.model_dump(mode='json') if source else None
        check('source_exact', source is not None and source.content == case['content'] and source.role == case['role'] and source.scope_id == context['project_id'] and source.occurred_at.isoformat() == payload['occurred_at'])
        candidates = facts_for(captured)
        record['extracted'] = [await call('inspect', {'context': context, 'fact_id': f.id}) for f in candidates]
        record['extraction_outcome'] = 'committed_proposals' if candidates else 'abstained'
        check('controls_unchanged_after_extraction', all(service.engine.store.get(f) for f in control_ids))
        if 'correction' not in case:
            check('negative_control_no_facts', not candidates)
            record['status'] = 'completed_negative_control'
            return record
        if not candidates:
            record['status'] = 'downstream_blocked_by_abstention'
            return record
        # Do not guess which assertion to correct if a source yields several facts.
        if len(candidates) != 1:
            record['status'] = 'downstream_blocked_by_multiple_proposals'
            return record
        target = candidates[0]
        record['initial_retrieval'] = await retrieve(case['query'])
        record['initial_retrieval_contains_target'] = any(f['id'] == target.id for f in record['initial_retrieval']['facts'])
        correction_payload = {'event_id': 'correction-' + case['id'], 'context': context, 'content': case['correction'], 'fact_id': target.id, 'expected_version': target.version}
        corrected = await call('correct', correction_payload)
        check('correction_committed', corrected['status'] == 'committed')
        await drain([corrected])
        inspected = await call('inspect', {'context': context, 'fact_id': target.id})
        record['after_correction'] = inspected
        check('same_id_next_version_exact_correction', inspected['fact']['id'] == target.id and inspected['fact']['version'] == target.version + 1 and inspected['fact']['content'] == case['correction'])
        check('correction_evidence_linked', corrected['event_id'] in {e['source_event_id'] for e in inspected['evidence']})
        check('historical_revision_preserved', any(r['content'] == target.content for r in inspected['revisions']))
        await service.close()
        service = open_service()
        check('same_database_reopened', record['instances'][0]['database'] == record['instances'][-1]['database'])
        reopened = await call('inspect', {'context': context, 'fact_id': target.id})
        check('correction_survives_restart', reopened == inspected)
        response = await retrieve(case['query'])
        record['restart_retrieval_contains_target'] = any(f['id'] == target.id for f in response['facts'])
        preview = await call('forget_preview', {'context': context, 'fact_id': target.id, 'expected_version': target.version + 1, 'limit': 100})
        check('forget_preview_complete', preview['complete'] and bool(preview['apply_preconditions']))
        # Review graph closure: select only this source's fact, never controls.
        selected_sources = [s['source_event_id'] for s in preview['source_events']]
        check('forget_has_no_unrelated_candidates', not preview['candidates'])
        apply = {'context': context, **preview['apply_preconditions'], 'selected_fact_ids': [target.id], 'selected_source_event_ids': selected_sources}
        ForgetRelatedRequest.model_validate(apply)
        await call('forget_related', apply)
        check('forgotten_target_absent', service.engine.store.get(target.id) is None)
        check('both_source_ids_suppressed', all(service.engine.store.is_suppressed(source_event_id=r['event_id']) for r in (captured, corrected)))
        original_replay = await call('capture', payload)
        correction_event = service.engine.store.get_episodic_event(corrected['event_id'])
        check('correction_source_erased', correction_event is None)
        correction_replay_payload = {**payload, 'event_id': correction_payload['event_id'], 'content': case['correction'], 'source_kind': 'explicit_correction'}
        correction_replay = await call('capture', correction_replay_payload)
        check('original_and_correction_replay_suppressed', original_replay.get('processing_status') == correction_replay.get('processing_status') == 'suppressed')
        await call('maintenance', {})
        await service.close()
        service = open_service()
        check('final_restart_no_resurrection', service.engine.store.get(target.id) is None and {f.id for f in service.engine.store.get_all()} == control_ids)
        final = await retrieve(case['query'])
        check('final_retrieval_no_forgotten_id', all(f['id'] != target.id for f in final['facts']))
        record['status'] = 'completed_lifecycle'
    except Exception as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        if service:
            dump(folder / 'trace.json', service._benchmark_provider_trace)
            record['final_facts'] = [f.model_dump(mode='json', exclude={'embedding'}) for f in service.engine.store.get_all()]
            record['final_jobs'] = [j.model_dump(mode='json') for j in service.engine.store.list_jobs()]
            await service.close()
        dump(folder / 'result.json', record)
    return record

async def main():
    assert not (root / 'status.json').exists()
    status = {'status': 'running', 'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
              'preparation_sha256': sha(root / 'preparation.json'), 'cases': []}
    dump(root / 'status.json', status)
    for index, case in enumerate(inputs['cases']):
        row = await one(case, index)
        summary = {k: row.get(k) for k in ('status', 'extraction_outcome', 'error')}
        summary['case_id'] = case['id']
        status['cases'].append(summary)
        dump(root / 'status.json', status)
        print(json.dumps(summary), flush=True)
    verify()
    status.update(status='completed', finished_at=datetime.now(timezone.utc).isoformat(),
                  all_lifecycle_checks_passed=all(c['status'] in {'completed_lifecycle', 'completed_negative_control'} for c in status['cases']))
    dump(root / 'status.json', status)

asyncio.run(main())
