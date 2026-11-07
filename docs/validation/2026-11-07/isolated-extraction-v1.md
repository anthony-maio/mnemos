# Isolated extraction development v1

The source-bound pipeline matched 14 of 15 quality cases, retained eight of
nine annotated critical facts, and produced one explicit unresolved review.
The known-target correction failed classification before reconciliation. A
separate foreign-scope case rejected its target before any provider call.
Automatic extraction remains experimental.

The [protocol](../../plans/2026-11-07-isolated-extraction-development-v1.md),
scorer, runner and 16-case corpus were committed at
`28cf2ce72e6c60e7e471c94bfb85b5fda846e85b`. This was one repetition of authored
development cases using previously studied scenario families. Host-channel
labels were synthetic. The experiment called the production source-bound
classification and relationship reconciliation functions but performed no
retrieval, embeddings, accepted-fact commits or live configuration changes.

| Measure | Result |
| --- | ---: |
| Exact quality cases | 14/15 |
| Operational failures among quality cases | 1/15 |
| Exact critical facts retained as proposals | 8/9 |
| Exact supported accepted proposals | 9/9 |
| Unmatched accepted proposals | 0 |
| Explicit unresolved reviews | 1 |
| Correct no-retention decisions | 4/4 |
| Separate pre-provider scope guard | 1/1 |
| Median quality-case latency | 3,086 ms |
| p95 quality-case latency, nearest rank | 4,459 ms |

These are exact annotation proxies. They do not establish semantic factual
precision. The critical-retention denominator includes the failed correction;
the nine emitted proposals exclude that missing proposal. The 9/9 precision
proxy therefore does not imply complete extraction. Containment and exact
critical-span counts both equal eight in this run.

The rejected classification had `kind: correction_or_rejection` and
`revises_existing_requirement: true`. That flag is reserved for direct user
continuing requirements. The emitted structured decoder schema accepted the
combination, while the unchanged runtime renderer rejected it. The failure is
an underconstrained decoder contract, not a missing target or a storage error.
No observation was relabeled or rerun to repair the score.

Qwen3 14B produced 25 raw responses for 25 logical calls, all with normal stop
reasons. Provider-reported counts were 28,014 prompt tokens, 23,670 cached prompt
tokens and 1,629 output tokens. The largest native count for a prompt plus its
decoder schema was 2,066, below the frozen 5,632 allowance. The largest reported
prompt input was 1,483 tokens. Whole-case latency includes both extraction
stages and the separate native count checks.

The independent audit verified all 132 bound source files, original inputs,
request/response/completion accounting, raw classification and reconciliation
decisions, source provenance, target versions and every successful score. It
also confirmed that the rejected correction was valid under its emitted schema.
The model unloaded, both owned endpoints closed and all recorded owned
processes were absent.

Seventy-seven focused checks passed, including 20 new evaluator checks.
Formatting and strict types passed. The initial missing-module test failure,
an implementation syntax error and a test's incorrect source-boundary offset
are preserved with the final passing evidence; none caused model inference.

The [receipt](isolated-extraction-v1-receipt.json) binds 156 verified private
archive members under SHA-256
`a4afa55bfde056ef0c20723f5920fae6f78db7c33be2ca3ac657d09b6b1651fc`.
The next bounded step is a decoder-schema repair followed by a separately
recorded development rerun. Fresh evaluation, semantic review, persisted
extraction, native-host integration and pilot acceptance remain open.
