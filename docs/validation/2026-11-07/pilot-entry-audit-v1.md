# Private explicit-memory pilot entry audit

The prepared pilot remains inactive. Its settings use local BGE-small
embeddings, baseline retrieval, no judgment provider, no query-state inference
and all five experimental modules disabled. Four fresh configuration-only
probes passed across the two selected projects and both proposed hosts, even
with conflicting inherited model, module and storage settings. No service or
database was created.

The installed artifact is still product `6d0e5ba6`, wheel SHA-256
`ecd11eb7ecbb9cc4f14552d9a7253dead7a8348879c35063411bb850abe06036`.
All 86 package files and 57 dependency versions match. The launcher, settings
and project registry retain their recorded bytes. All seven unapplied patches
still match their expected original files or absence, and both reference
configuration files retain their recorded hashes. Thirteen watched paths
stayed unchanged through this audit. These checks establish current preparation
state, not entry acceptance or daily use.

The [receipt](pilot-entry-audit-v1-receipt.json) binds seven verified private
archive members, including the audit helper, results and installed execution
paths. Archive SHA-256:
`18451a0b2d8dab4bb72c5aa961ecd5a7978d0ebf47eed7178f22b0ea064ba09f`.

## Which requirements apply

| Requirement | Evidence and remaining work |
| --- | --- |
| Explicit remember, inspect, restart, correct and forget | [Codex v7](../2026-11-06/native-codex-v7-checkpoint.md) and [Claude v5](../2026-11-06/native-claude-v5-checkpoint.md) passed the six-phase lifecycle on this wheel. These are bounded synthetic sessions, not reliability estimates. |
| Automatic extraction and target selection | Disabled in this pilot. Explicit remember/correct jobs construct proposals from the supplied content and requested target. With sleep off, ordinary captured text becomes unasserted passthrough material; it is not an automatically accepted fact. The failed model comparisons remain open for enrichment and the broader improvement goal. They do not execute in this configuration. |
| Capture and participating host behavior | [Codex integration](native-pilot-codex.md) records two normal captures. [Claude integration](native-pilot-claude.md) records five arrivals, including a provisional prompt and pending work. Its missing occurrence identity remains a limitation. Normal interactive host trust is still unverified; the Codex experiment used a one-run override. |
| Recovery | [Startup outage](native-pilot-codex-outage.md) and [embedding error recovery](pilot-embedding-recovery.md) passed on this wheel. Each result retains its stated fault and coverage limits. The old blanket statement that provider recovery has no evidence is stale. |
| Configuration application and rollback | [Disposable rollback](pilot-config-rollback.md) passed exact restoration and drift refusal. The current patch preconditions still match. Fresh backups and explicit activation authorization remain required before any live application. |
| Retrieval and answer support | Still applicable. Disabling extraction does not establish that related retrieved text answers a question. Fresh recall and no-answer evaluation on accepted facts remains open; small native examples do not replace it. |
| Scale and latency | The original warm-recall and durable-capture thresholds remain. No current pilot-bound 10,000-fact result is supplied by this audit. |
| Daily use | Zero days. The two-week, two-project requirement and separate external-pilot requirement are unchanged. |

The execution-path assessment follows the installed `memory_service.py`
remember/correct branch and its sleep-disabled capture branch, together with
the installed `runtime.py` unavailable-provider fallback. Package hash checks
bind that assessment to the artifact. It is not a new inference experiment.

The [original acceptance targets](../../plans/2026-09-16-functional-local-memory.md)
remain unchanged: 95% supported-fact precision and critical-fact retention,
90% Recall@5 and correct no-answer behavior, the declared profile comparison,
and the recorded latency and scope/durability requirements. Automatic-extraction
metrics still require their own evaluation before enrichment can be promoted.
This audit neither declares those metrics passed nor replaces them with explicit
write fidelity. The [pilot protocol](../../plans/2026-09-20-local-memory-daily-use-pilot.md)
still requires the applicable quality gates before activation.

## Next evaluation boundary

The existing `RetrievalComparatorAdapter` fixes one accepted snapshot across
retrieval strategies, but constructs that snapshot through full-pipeline
ingestion. Running it unchanged would include extraction failures in this
pilot's input preparation. A pilot-specific retrieval evaluation must instead
bind an accepted-fact snapshot, its source evidence and lifecycle state before
comparing retrieval. Keep extraction, retrieval on accepted facts and persisted
full-pipeline results separate.

Prepare that evaluation before opening fresh cases. Keep the original minimum
sample counts, three repetitions, host/project coverage, confidence intervals
and paired comparisons. Preserve the unused fresh corpus and its existing
protocol; a new configuration must receive its own binding. No fresh cases or
model responses were read for this audit.

The current development source has additional changes beyond the installed
pilot. Its [new regression and package checks](checkpoint-749f1213.md) have a separate source binding;
the older native-host passes cannot be relabeled as validation of a new wheel.
No pilot configuration, host approval, live memory or published artifact changed.
