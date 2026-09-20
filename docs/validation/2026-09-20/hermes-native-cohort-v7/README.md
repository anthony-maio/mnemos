# Native Hermes v7: receipt package, three failed lifecycles

The fixed cohort completed all three scheduled attempts on verified runtime
`1a110583a18408cbd3de4a695a9962109c4d02e7`, wheel
`4e3d16d00f7b9ab54fe1e5350f7cf0fbe6209dc881bfc6d7d9f974f1c3f697a5`.
None completed the lifecycle. All three model-readiness checks passed, and
all attempts established cleanup. The cohort exited 1 with
`completed_with_failures`; this is a terminal failed result, not a running job.

The [prospective protocol](../../../plans/2026-09-20-hermes-native-lifecycle-protocol-v7.md)
and runner were committed before inference. Runner/input checkpoint:
`0751727f6398270ae28629ff9b248d321ed203d3`. The package's source/installed
verification remains separate from this native-model result. The eight tools,
prompts, capture, restart sequence, replay, evaluator, erasure scans, and
time budgets retain the v6 criteria. Inputs remained unchanged through the
cohort and were checked again while archiving.

| Attempt | Readiness chat | Last phase | Observed failure |
| --- | --- | --- | --- |
| 1 | 77.818 seconds | Correct | Retrieved the asserted target, then announced a correction without inspection or a correction call. Target stayed at version 1. |
| 2 | 9.271 seconds | Forget | Passed teaching, restarted recall, correction, and restarted corrected recall. At forget it retrieved, then announced an inspection without calling inspect, preview, or apply. Target and copies remained. |
| 3 | 8.955 seconds | Teach | Called remember and accurately reported its pending receipt, but omitted the explicitly required job-status call. The teach gate failed. |

Later phases stopped after each failure. No attempt reached source replay or
final recall. No attempt called related forget, so this cohort does not test
whether the new coverage receipt changes a model's deletion decision. Do not
describe it as a receipt-success or receipt-failure comparison.

## Readiness and scope of the result

All warmups used the pinned Qwen3.5:9b greedy-v1 alias, a 65,536-token context,
no tools, no memory content, and one request with a 180-second timeout.
Digests matched before and after. Each loaded-model response reported
8,044,454,869 bytes of GPU residency for the expected model. This establishes
positive GPU use, not a claim about every layer or steady-state performance.
Raw request bytes, response bodies, metadata, timings, and readiness records
are preserved for every attempt. The first load took longer than the 60-second
request budget used in the earlier direct development diagnostic; these are
different runs and workflows, so do not treat that observation as a causal
comparison of model quality.

The actual pinned Hermes host ran with the isolated cwd-only patch, a
disposable home/project/database, and automatic conversation capture enabled.
All five cognition modules were disabled for the native host baseline.
Nomic provided real query/document embeddings. Live host configuration and
live memory were unchanged. Import-only discovery and the 73 workflow plus
15 readiness offline checks are in the
[preflight archive](../hermes-native-discovery-preflight-v6/README.md).

## What the traces suggest, and what remains unproved

Attempt 1's correction retrieval returned the asserted target alongside
older raw captured instructions such as the prior recall request and its
instruction not to change memory. Those rows were labeled as unverified
conversation. The model then announced a correction without executing it.
This is an observed coexistence, not proof that the old instructions caused
the omission. Attempt 3 skipped job status in a fresh teach phase, so stale
conversation cannot account for every failure in this cohort.

A useful next diagnostic should separate recalled conversation from durable
facts and test next-action behavior under a prospectively fixed comparison.
Do not remove stored captures, relax erasure scans, or reinterpret a model's
announcement as a successful operation. Reliable native mutation, related-copy
forgetting, three-host continuity, complete capture, full-pipeline quality,
Desktop, and the pilot remain open. Earlier cohorts and T18 negatives stand.

The evidence index binds 190 raw files. It excludes this interpretive README.
The external disposable databases remain at the recorded paths; the archive
contains selected synthetic state snapshots and traces, not live credentials.
