# Complete installed scoped ranking results

The scoped ranking candidate reduces complete recall latency while preserving
the returned facts, source evidence, scores and context. It does not meet the
100 ms complete-hook requirement. The candidate remains local, and the private
pilot remains inactive.

This comparison used candidate `5735c502` and baseline `feb6ac3b`, each installed
from its verified wheel. Four independent blocks ran in the prospective order
baseline-0, candidate-0, candidate-1, baseline-1. Only one service/worker pair ran
at a time. Each block started from a copy of the same 10,000 accepted facts,
original sources and real BGE document vectors. Both wheels used the same
Python, 57 dependency versions and 17 model assets.

Each block ran three recall warmups and 25 timed recalls, then one warmup and
25 timed commands for each hook event. The complete run retained 320
observations, including 300 timed samples. It took 593.80 seconds including
preparation, process supervision, restart checks and final verification.

| Complete path | Baseline median ms | Candidate median ms | Baseline p95 ms | Candidate p95 ms | Required p95 ms | Candidate result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Recall | 285.01 | 119.21 | 311.35 | 134.88 | 500 | PASS |
| Stop | 321.81 | 328.96 | 646.27 | 547.11 | 100 | FAIL |
| UserPromptSubmit | 666.41 | 510.87 | 895.70 | 860.06 | 100 | FAIL |

The candidate was faster in all 50 recall pairs and 44 of 50 prompt pairs.
Their median paired changes were -167.14 and -145.65 ms. Stop was faster in
24 of 50 pairs, with a median paired change of +1.44 ms. There is no consistent
Stop improvement here. Candidate maxima were 181.96 ms for recall, 895.62 ms
for Stop and 994.50 ms for prompt submission. No timed operation failed.

| Block | Recall median / p95 ms | Stop median / p95 ms | Prompt median / p95 ms |
| --- | ---: | ---: | ---: |
| baseline-0 | 276.85 / 299.53 | 319.05 / 688.74 | 642.50 / 820.77 |
| candidate-0 | 115.35 / 127.30 | 314.14 / 512.72 | 495.14 / 903.06 |
| candidate-1 | 121.56 / 146.10 | 333.23 / 534.62 | 511.89 / 725.44 |
| baseline-1 | 298.31 / 312.96 | 324.20 / 529.22 | 674.38 / 898.47 |

The recall improvement appears in both halves. Hook timing remains variable;
reversing the block order does not remove all time-dependent machine noise.
These figures measure the complete installed command and service paths with
real query embeddings. No startup time was subtracted, retrieval skipped,
context shortened or answer cache added.

All 160 baseline/candidate pairs, including warmups, returned identical decoded
responses and exact context strings. This was the prospective equality rule.
The earlier experiment's raw-byte failures remain unchanged. The candidate
preserves the same facts per context budget; this run makes no new semantic
quality or tokenizer-efficiency claim.

All 208 captured sources and jobs survived service restart. Every capture job
completed. Each block ended with its original 10,000 accepted facts plus 52
unknown captured chunks; no capture became an accepted fact. Accepted contents,
versions, metadata, sources, evidence and vectors remained equal to the seed.
SQLite integrity and foreign-key checks passed.

All 208 successful resident commands avoided fallback and ordinary runtime
imports. Twelve policy preflights passed across the four blocks. The original
private launcher checks, current settings/environment/working-directory checks,
scope registry resolution and durable outbox remained in the execution path.
All twelve owned service/worker lifecycles ended normally, with their processes
absent and descriptors removed. Effective settings and environment were recorded
only as hashes.

An independent audit reproduced all 320 observations, 160 paired comparisons,
latency summaries and 208 durable source/job checks. It verified 122 frozen
file bindings and 17 model assets per block, accepted-cohort equality and
database integrity, then confirmed all 235 recorded process IDs were absent.
The audit made no model calls. All twelve documentation checks pass.

The verified private archive contains 350 logical files in 188 content-addressed
members, including all four runtime copies and their original bindings.
SHA-256:
`a5e303ba8bd2c4cd3ee7d0281abd2fb0cf853c85441c7981ab01fee68ce0afb6`.

The remaining hook costs include the command interpreter/import boundary,
canonical Git project lookup and durable capture. A useful next experiment is
a faster command bootstrap with the original checked fallback. Any identity
optimization must preserve current scope resolution and fail-closed behavior.
The current result does not justify changing embedding thread settings or
weakening the latency requirement.

Current native-host acceptance, answer-support quality, canonical delivery and
the actual two-week pilot remain open. The older full-suite result is still
bound to its original source; this performance run does not replace it.

- [Prospective protocol](../../plans/2026-11-07-scoped-rank-performance-v1.md)
- [Implementation and component evidence](scoped-rank-v1.md)
- [Prior resident diagnostics](resident-hook-diagnostics-v1.md)
- [Verification receipt](scoped-rank-performance-v1-receipt.json)
