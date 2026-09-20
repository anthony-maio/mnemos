# Native Hermes v14: copyable-preconditions package result

The fixed cohort completed one of three full lifecycles on verified product
`e6887deb89608dca34f7bbb463b5d0a3ba973011`, wheel SHA-256
`3d37b864d35916dbed0c22bcb0428a4e091d6b6c31a0177491130fa18da34fa1`.
Both failures remain part of the result. This is one bounded transport
baseline pass, not a reliability estimate or a full-module quality result.

| Attempt | Observed result | Independently inspected terminal state |
| --- | --- | --- |
| 1 | Passed teaching, restarted recall, correction, and corrected recall; stopped after a complete forget preview without calling apply | Target retained at version 2; zero suppression rows |
| 2 | Passed the complete lifecycle, including reviewed-copy deletion, source replay, restart, and final recall | Target absent; 22 suppression rows; scope controls unchanged |
| 3 | Stored the exact fact but omitted the required job-status call; stopped at the teaching gate | Target retained at version 1; zero suppression rows |

Attempt 1's final model message said it was proceeding with related forget,
but no apply call followed. Attempt 3 reported the pending job from the write
receipt without the requested status read. Neither is a successful workflow.
The second attempt made one related-forget call, copied all four returned
preconditions exactly, and passed the unchanged erasure and replay checks.
No target-only deletion call was used in any attempt.

## Runtime and verification

The runner and [protocol](../../../plans/2026-09-20-hermes-native-lifecycle-protocol-v14.md)
were committed at `871b329c224e9520f3149b847167d5ce6b0a27ee` before execution.
The [import-only preflight](../hermes-native-discovery-preflight-v14/README.md)
discovered all eight tools without inference or service writes.

All 35 recorded model responses had the pinned Qwen alias/digest,
65,536-token served context, and positive GPU residency. Exact outgoing
requests and preview messages were correlated with the native traces.
All three attempts established cleanup. Independent read-only SQLite
inspection confirmed unchanged local and foreign-scope facts, events, and
evidence, with database hashes unchanged by inspection.

All five cognition modules were disabled for this native transport baseline.
The new package changed preview fields and guidance; the runner preserved
the prior prompts, request schemas, capture behavior, deadlines, and stored-
state acceptance requirements. No arguments were repaired or calls supplied
for the model. Variable identifiers and conversations prevent interpreting
the result as a causal comparison with the zero-of-three v13 cohort.

## Remaining work

A fresh prospective three-host protocol may use this completed, cleaned
cohort as its prerequisite. The earlier v6 protocol remains NOT RUN on its
own unsatisfied v11 binding; do not relabel it as passed. Three-host continuity,
complete capture, Desktop, full-module quality, and the operational pilot
remain open. The two omitted-action failures still limit dependable native use.

No live host configuration, source, or memory changed. The evidence index
binds 328 raw files, including both failures, the passing attempt, exact
requests, scope controls, and independent terminal/selection analyses.
