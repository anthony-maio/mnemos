# Isolated extraction schema repair v2

The schema repair removed the invalid classification from v1 without weakening
runtime validation. It did not recover the missing correction. The model now
returns an unresolved review despite the correct target being present. Quality
remains 14/15 exact cases and 8/9 critical facts. Keep automatic extraction
experimental and treat the remaining miss as target-selection work.

The [v2 protocol](../../plans/2026-11-07-isolated-extraction-development-v2.md)
was committed with the repair at
`f67e16c19dd47fe092c4c3bbf86f01fc72790e2f`. The run reused the exact corpus,
labels, model, options, source events, target versions and classification
instructions from [v1](isolated-extraction-v1.md). Only the production decoder
schema changed. Both studies remain development evidence, with one repetition
and no retrieval, embeddings, accepted-fact commits or native-host execution.

Each user source now has separate decoder alternatives for continuing
requirements and other kinds. Only the continuing-requirement alternative
permits either revision flag. Other kinds require false, and non-user sources
retain their existing authority restrictions. Each alternative includes its
complete source ID, kind, field and object constraints. The runtime renderer
still rejects invalid responses from providers that ignore the schema.

| Measure | V1 | V2 |
| --- | ---: | ---: |
| Exact quality cases | 14/15 | 14/15 |
| Operational failures among quality cases | 1/15 | 0/15 |
| Exact critical facts retained as proposals | 8/9 | 8/9 |
| Exact supported accepted proposals | 9/9 | 9/9 |
| Unmatched accepted proposals | 0 | 0 |
| Explicit unresolved reviews | 1 | 2 |
| Correct no-retention decisions | 4/4 | 4/4 |
| Separate pre-provider scope guard | 1/1 | 1/1 |
| Median quality-case latency | 3,086 ms | 3,122 ms |
| p95 quality-case latency, nearest rank | 4,459 ms | 4,280 ms |

The correction's classification now has `kind: correction_or_rejection` and
`revises_existing_requirement: false`. Reconciliation received the original
Thursday backup-window fact at version 4, and its decoder offered that exact
target under `user_revision`. The model instead chose
`missing_or_ambiguous_referent`. The target was visible; the model's stated
ambiguity is not evidence that the input lacked a unique target. The resulting
review remains separate from accepted proposals and receives no retention credit.

V2 produced 26 responses for 26 logical calls, all with normal stop reasons.
The extra call is the correction reconciliation that v1 never reached.
Provider-reported counts were 28,673 prompt tokens, 24,153 cached prompt tokens
and 1,693 output tokens. Maximum prompt-plus-schema native count rose from
2,066 to 2,257, below the unchanged 5,632 allowance. Maximum reported prompt
input remained 1,483. This small repeated run does not establish latency or
reliability differences.

The independent audit verified all 132 bound source files, all 16 identical
input pairs and classification prompts, raw responses, source provenance,
target versions and all 15 successful scores. Ten new tests reproduced the
schema gap before repair. All 17 revision-constraint checks and 98 focused
checks passed after repair, as did formatting and strict types for the changed
module. The earlier full-suite result retains its own source binding.

The [receipt](isolated-extraction-v2-receipt.json) binds 160 verified private
archive members, including the original comparison inputs, under SHA-256
`f3e6e4de90ade97db050cd78d14c6736f680b05715792f8a7f144572d39ffa0b`.
Both studies and their original scores are preserved. The model unloaded and
both owned endpoints closed. The original process-tree stop reported one
unsupported termination; independent inspection found all listed processes
absent, with no additional kill or model retry.

The next extraction work must distinguish a uniquely matching target from an
unrelated target and multiple plausible targets without forcing a correction
merely because one target exists. Fresh evaluation, semantic review, persisted
extraction and the actual private pilot remain open.
