from pathlib import Path
import json

repo = Path(__file__).resolve().parents[1]
archive = repo / 'docs/validation/2026-09-19/model-forget-development-v1'
counts = {}
for arm in ('qwen', 'gptoss'):
    report = json.loads((archive / arm / 'report.json').read_text(encoding='utf-8'))
    counts[arm] = sum(row['status'] == 'passed' for row in report['cases'])
paragraph = f'''The [local model tool diagnostic]({{link}}) on installed
`e454a85` finished with Qwen3.5:9b passing {counts['qwen']} of three cases and
GPT-OSS:20b passing {counts['gptoss']} of three. Each scheduled case ran once;
failures and startup timeouts remain in the archive. Passing cases require
observed inspection and preview, selected copies and their required sources,
persisted erasure and suppression, and unchanged own-project and foreign
controls. The runner also rechecks state through a read-only connection
after service shutdown. Its 68 focused offline tests, formatting, and
103-file type check pass. These counts are separate from the earlier full
package checkpoint.

This is a small development comparison with synthetic seeded facts,
SimpleEmbeddingProvider, and all five cognition modules disabled. It does
not automatically capture the conversation and is not a native Hermes,
restart, replay, extraction-quality, or three-host result. The native v6
failures and T18 negatives remain unchanged. The next native experiment
needs a separately frozen host/model profile and the existing erasure gates.
'''
for relative in ('docs/validation/2026-09-16/implementation-status.md',
                 'docs/validation/2026-09-16/progress.md'):
    p = repo / relative
    s = p.read_text(encoding='utf-8')
    title, rest = s.split('\n', 1)
    p.write_text(title + '\n\n' + paragraph.format(link='../2026-09-19/model-forget-development-v1/README.md') + '\n' + rest.lstrip('\n'), encoding='utf-8')
p = repo / 'docs/validation/2026-09-16/evaluation-running.md'
s = p.read_text(encoding='utf-8')
s = s.replace('## Latest native-host observation', '## Development tool comparison\n\n' + paragraph.format(link='../2026-09-19/model-forget-development-v1/README.md') + '\n## Latest native-host observation', 1)
p.write_text(s, encoding='utf-8')
p = repo / 'docs/handoffs/substack-local-memory.md'
s = p.read_text(encoding='utf-8')
s = s.replace('## Current evaluation update\n', '## Current evaluation update\n\n' + paragraph.format(link='../validation/2026-09-19/model-forget-development-v1/README.md'), 1)
p.write_text(s, encoding='utf-8')
p = repo / 'docs/plans/2026-09-16-functional-local-memory.md'
s = p.read_text(encoding='utf-8').replace('mnemos-substack-research-packet-2026-09-19-v10.zip', 'mnemos-substack-research-packet-2026-09-19-v11.zip')
p.write_text(s, encoding='utf-8')
print(json.dumps({'pass_counts': counts, 'updated_documents': 5}))
