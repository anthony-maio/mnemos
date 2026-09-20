# Native Hermes fixed cohort v5: three failed lifecycles

All three scheduled attempts ran and failed. No attempt was replaced or
retried. Runner `b6e7c47afc23cf6e96552a889dfb14452b9b32c8` used verified runtime
`fd83fb27042c05897f29dba952b0924b15889aad` and wheel
`4958c5ef7a509fefecacc0666d87f578b46e94e1a33a896831468c0f971bf895`.
The pinned local Qwen3.5:9b sampling profile, isolated Hermes checkout,
prompts, phase budgets, and lifecycle criteria were unchanged from v4.
The run ended at 2026-09-20 01:43:38 UTC, September 19 local time.

| Attempt | Passed phases | First failed phase | Observed failure |
| --- | --- | --- | --- |
| 1 | Teach, restarted recall, correction, restarted corrected recall, forget, source replay | Recall after forgetting | Previously captured conversational copies survived the exact-ID deletions. The final model request still contained the corrected label. |
| 2 | Teach, restarted recall, correction, restarted corrected recall, forget, source replay | Recall after forgetting | Previously captured conversational copies survived. The final model request still contained a forgotten label, and the answer reported the old label from a conversational copy. |
| 3 | Teach, restarted recall, correction, restarted corrected recall | Forget | The model retrieved and inspected the target, then announced that it would forget it without making a forget call. The asserted fact remained present and unsuppressed. |

All three attempts passed correction and restarted corrected recall. No
provider-error gate failed. These observations do not establish reliable
full lifecycle behavior or isolate the effect of the preceding repairs.

Attempts 1 and 2 each passed the exact target deletion and original-source
replay gates, then failed `no_forgotten_label_in_request`. In attempt 1 the
model retrieved five facts, inspected them, deleted four, and preserved the
unrelated checksum control. Other older conversational copies remained
outside that retrieved set. The next prefetch included the original
correction instruction and its corrected label. This is distinct from v4's
new assistant narration escaping the tombstone guard. A bounded retrieval
result is not an exhaustive inventory of copies to erase.

The Hermes provider at this checkpoint exposes exact-ID forgetting but not
the service's existing reviewed-copy preview and deletion operations. That
is a concrete integration gap for the requested copy-erasure workflow.
Exposing those operations is prospective work, not a change to these
outcomes. It must retain scope checks, stale-preview rejection, and explicit
selection of additional facts and sources. Similarity alone is not authority
to delete another fact.

Attempt 3 contains no native `mnemos_forget` call. A final answer promising
to perform a mutation does not establish that the mutation happened. No
replay or final-recall phase ran in that attempt.

The cohort finished with zero of three complete passes. Cleanup was
established after every attempt. The three-host continuity prerequisite
remains unmet. Sampling evidence records the pinned server profile and
absence of conflicting overrides, not internal sampler execution.

`cohort.json` retains every outcome. `evidence-index.json` binds 191 files,
including executed inputs, request/response traces, selected synthetic
state, and logs. This interpretive README is outside that raw index.
Databases, runtime access descriptors, credentials, and live user content
are excluded. Later fixes cannot change these results.
