"""Preserve both scheduled development arms without selecting successful cases."""
from pathlib import Path
import hashlib
import json
import shutil

repo = Path(__file__).resolve().parents[1]
base = Path('C:/Users/Ant/AppData/Local/Temp/mnemos-model-forget-development-v1')
out = repo / 'docs/validation/2026-09-19/model-forget-development-v1'
out.mkdir(exist_ok=False)
(out / '.gitattributes').write_text('** -text -whitespace\n', encoding='utf-8')
reports = {}
for name in ('qwen', 'gptoss'):
    report = json.loads((base / name / 'report.json').read_text(encoding='utf-8'))
    assert report['status'] in ('passed', 'failed')
    assert len(report['cases']) == 3
    assert all(case['status'] in ('passed', 'failed') for case in report['cases'])
    reports[name] = report
    shutil.copytree(base / name, out / name)
for path in base.glob('*.json'):
    shutil.copyfile(path, out / path.name)
for relative in (
    'benchmarks/native_hosts/model_forget_development_v1.py',
    'benchmarks/native_hosts/hermes_trace_v2.py',
    'tests/test_model_forget_development.py',
    'tests/test_model_forget_runner.py',
    'docs/plans/2026-09-19-local-model-forget-development-v1.md',
    'docs/plans/2026-09-19-local-model-forget-profiles-v1.json',
):
    target = out / 'inputs' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo / relative, target)
(out / 'helpers').mkdir()
shutil.copyfile(Path(__file__), out / 'helpers' / Path(__file__).name)
for name in ('model-forget-runner-offline-v1.xml', 'model-forget-runner-offline-v2.xml',
             'model-forget-offline-final.xml', 'model-forget-offline-final.log',
             'model-forget-mypy-final.log', 'model-forget-black-final.log'):
    shutil.copyfile(repo / '.venv' / name, out / name)
rows = []
for arm, report in reports.items():
    for index, case in enumerate(report['cases'], 1):
        detail = json.loads((base / arm / f"case-{index}-{case['name']}" / 'report.json').read_text(encoding='utf-8'))
        failed = [name for name, value in (case.get('gates') or {}).items() if not value]
        reason = case.get('error') or detail.get('error_type') or ', '.join(failed) or ('all gates passed' if case['status'] == 'passed' else 'failed; see case report')
        rows.append(f"| {report['model']} | {case['name']} | {case['status']} | {case.get('elapsed_seconds', 0):.2f} | {reason} |")
text = '''# Reviewed forgetting: local model development comparison

This is a three-case-per-model development diagnostic using the installed
`e454a85` wheel, not a native Hermes lifecycle or a T18 replacement. Qwen ran
first, then GPT-OSS. Each arm ran every scheduled case once. The raw reports,
SQLite fixtures, exact prompts and requests, tool traces, state snapshots,
module origins, package hashes, input hashes, and cleanup outcomes are kept.
Transport IDs are normalized for this direct Ollama runner; they are not
native Hermes identities.

| Model | Case | Status | Seconds | Failure or outcome |
| --- | --- | --- | --- | --- |
''' + '\n'.join(rows) + '''

Qwen's backup-window case and GPT-OSS's backup-window and deployment-lane
cases returned no model response before the 60-second request limit. They
therefore do not establish tool-selection performance. Preserved server-log
observations record canceled model loading, followed by later successful
loads. The old preserved tag also produces metadata errors during background
listing; those messages alone do not establish the cause of load latency.

Qwen's two completed cases passed every gate before and after shutdown.
GPT-OSS's archive-bucket case retrieved, previewed, and applied target-only
deletion with empty additional-fact and source selections. It made no
inspection calls, retained all three known conversational copies, and then
claimed that all related copies were removed. Its storage result and final
language disagree. Qwen's passing final responses also repeat deleted values
and IDs; since capture is disabled here, these passes do not establish that
native capture would avoid recreating forgotten material.

Both arms use actual authenticated service dispatch and synthetic seeded
facts with SimpleEmbeddingProvider. The five cognition modules are disabled.
Conversation capture is disabled. This does not measure extraction quality,
Nomic retrieval, source replay, restarts, native callbacks, WSL transport,
cross-host continuity, or daily use. Fixture IDs and paths differ between
arms. Three cases do not estimate reliability or establish a causal effect
of model size, reasoning effort, or architecture.

Offline runner development initially failed three cleanup checks because
the helper read a connection after the service had closed it. The repaired
helper reopens the database read-only; all four runner checks then passed.
Both XML reports are retained. The final offline report exercises scoring
and transport evidence separately from all real-model outcomes above.

The [prospective protocol](../../../plans/2026-09-19-local-model-forget-development-v1.md)
and [frozen profiles](../../../plans/2026-09-19-local-model-forget-profiles-v1.json)
define inputs and limits. The [readiness archive](../local-model-readiness-v1/README.md)
preserves the old model failure, successful replacement, and failed readiness
assertion. The earlier [native v6 cohort](../hermes-native-cohort-v6/README.md)
remains zero of three complete lifecycles on its original runtime. No result
here changes those outcomes or closes native acceptance.
'''
(out / 'README.md').write_text(text, encoding='utf-8')
items = []
for path in sorted(out.rglob('*')):
    if path.is_file():
        items.append({'path': path.relative_to(out).as_posix(), 'bytes': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
(out / 'evidence-index.json').write_text(json.dumps({'files': items}, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'archive': str(out), 'indexed_files': len(items)}))
