# Native Hermes v13: review-rejection package result

The fixed cohort completed zero of three full lifecycles on verified product
`db217509477d92b9a82bd7e11d4a4bffb542fa29`, wheel
`8b1f560e98b93376fc2aa073be7c1500997d888d09ae87573d5d70ec8694ad00`.
Every attempt passed teaching, restarted recall, correction, and corrected
recall, then failed at forgetting. No attempt reached source replay or final
recall. All scheduled outcomes are retained.

| Attempt | Forgetting result | Independently inspected terminal state |
| --- | --- | --- |
| 1 | Three stale-preview rejections; apply used limit 20 after previews returned 100 | Target retained at version 2; zero suppressions |
| 2 | Three stale-preview rejections and one missing-version rejection; apply again used 20 after previews returned 100 | Target retained at version 2; zero suppressions |
| 3 | Rejected sources outside the selectable preview list, followed by target-only deletion | Target absent; four suppressions; reviewed-copy erasure unproven |

Attempt 3's final response correctly acknowledged that target-only deletion
did not cover the conversational copies. Its native process exited zero, but
the frozen evaluator rejected the unsuccessful reviewed-forget operation.
The full requested lifecycle therefore remains failed. Attempts 1 and 2
exited one after their repeated unsuccessful tool use.

## Runtime and evidence

The runner and protocol were committed at `dfb08d5` before the cohort.
The [import preflight](../hermes-native-discovery-preflight-v13/README.md)
discovered eight tools without inference. The native cohort kept all five
cognition modules off and preserved the preceding prompts, capture behavior,
deadlines, tool schemas, and erasure requirements.

All 55 recorded model responses had a verified pinned Qwen alias/digest,
65,536-token serving context, and positive GPU residency. Every attempt
established cleanup. Independent read-only database inspection confirmed
unchanged local-control and foreign-scope facts, events, and evidence in all
three attempts, with database hashes unchanged by inspection.

The archive preserves exact outgoing requests, traces, tool results, input
hashes, phase state, readiness observations, and selection/terminal analyses.
The new incomplete-preview and dependency-conflict rejections were not
exercised here. Package tests establish their bounded behavior; this cohort
does not establish model recovery or reliable deletion. No HTTP 500 related-
forget result was recorded, but that is not a complete lifecycle pass.

The model still failed to follow the reviewed selection contract. Changing
the stored-state checks or repairing its arguments would change the test.
Earlier failures and the single v10 pass retain their original package
bindings; these cohorts are not pooled into a reliability estimate.

No live host configuration or memory changed. Three-host continuity,
complete capture, Desktop, module quality, and the pilot remain open.
The index binds raw files; this README is outside the index.
