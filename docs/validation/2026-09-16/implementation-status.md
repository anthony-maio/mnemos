# Functional local memory implementation status

This record covers the unreleased `wt/functional-local-memory` worktree. It
separates implemented behavior from acceptance evidence. The original task
boundaries and thresholds remain in the [implementation plan](../../plans/2026-09-16-functional-local-memory.md).

| Plan item | Implemented behavior | Evidence and remaining acceptance |
| --- | --- | --- |
| T01 Runtime | Configured providers, SQLite default, explicit mock rejection, observed readiness probe | Runtime and health tests; actual Ollama smoke; installed-wheel lifecycle and assets passed |
| T02 Scope | Registered profile/project/workspace identities and explicit membership; read-only legacy review | Scope, canonical identity, foreign-project and foreign-profile contract tests; offline repair preserves ambiguous legacy ownership |
| T03 Durable writes | Atomic source/job creation, leased processing, evidence/revision commits | Retry, lease, CAS, rollback, and persistence tests |
| T04 Local service | One authenticated loopback owner, private descriptor, outbox, explicit shutdown | Real HTTP and subprocess tests; duplicate launcher regression passes |
| T05 Maintenance | Service-owned scheduler, bounded batches, visible pending/retry/failure state | Scheduler and durable-job tests; daily-use pilot not run |
| T06 Evidence and safety | Shared write firewall, source validation, role labels, governed explicit writes, local-original availability | Safety, retention, restart, and lifecycle tests; real extraction precision remains unproved |
| T07 Surprisal | Automatic gating, explicit-remember bypass, session-safe history and diagnostics | Deterministic mechanism tests; real filtering comparison pending |
| T08 Correction | Stable identity, expected-version CAS, revisions, canonical and graph refresh | Correction, duplicate, evidence, and graph regression tests |
| T09 Retrieval | Exact scoped cosine, lexical candidate union, one scoring contract, context budget | Native/fallback scope and top-1 tests pass. Frozen `b7a9d9d` recall p95 1,314.583 ms fails the 500 ms gate; capture p95 57.410 ms passes. Two recall stalls occurred inside durable metadata updates. Earlier 506.111 ms failure and 458.516 ms pass remain preserved; held-out quality pending |
| T10 Graph | Persisted mutual top-k links, independent of insertion order and traversal hydration; bounded recovery; versioned empty preferences and scope-safe traversal | v3 failure preserved. v4 focused, full-suite, installed-wheel, and browser recovery checks pass. All 15 real development graph observations pass recorded correctness controls; none supplies an eligible target path, so no recall benefit is established. Held-out evaluation queued |
| T11 Forget | Live deletion and suppression across replay, retry, inflight extraction, restore | Lifecycle and backup tests; historical backups/transcripts can retain content |
| T12 Claude Code | Native hooks, transcript deltas, capture-before-recall, durable checkpoints | Installed d17dd24 native repeat verified seeded recall, user and assistant capture, and restart persistence, with degraded transcript coverage. Source 20f271d repairs event/checkpoint scope collisions and preserves legacy forget suppression; 54 focused tests and an installed-wheel two-project/restart/forget/outbox smoke pass. Its first broad run passed 669 tests with two WSL skips and an environment-sync caveat; stable repeat pending. Legacy checkpoint binding, redaction, extraction, and full native lifecycle acceptance remain open |
| T13 Codex | Hook configuration, native payload/output adapter, explicit MCP fallback | Installed 20f271d native v8 used the actual generated TOML commands, answered a random canary absent from the prompt without tool calls, and captured both roles across restart. Constant fixture vectors and unavailable cognition isolate hook transport from semantic quality. No individual hook trace, Desktop, or full native lifecycle claim. Earlier failed and partial attempts remain preserved |
| T14 Hermes | Optional installed MemoryProvider integration and explicit Windows/WSL bridge | Installed API and actual bridge observed; native model turn did not complete |
| T15 UI | Project enable/pause, observed readiness, facts/history/correction/forget, job state, legacy and routing review | API and browser checks include profile-filtered diagnostic dismissal and source expiry; bounded-list and active-store labels disclosed |
| T16 Recovery | Consistent backup, new-path restore, suppression merge, staged reindex and activation backup | Backup, integrity, identity, ownership, and reindex tests |
| T17 Cross-host | One service contract for all three host contexts | Synthetic adapter continuity passes across restart/correct/forget; complete native matrix and two-week pilot remain open |
| T18 Evaluation | Frozen lifecycle datasets, real provider adapter, seven profiles, checkpoints, raw traces, intervals | Failed v2 lifecycle and original mechanism development runs preserved. V3 names actual lifecycle targets; schema repair passed four real correction/forget/replay/restart development cases. Repaired mechanism development completed with passing graph controls and no graph utility evidence; classifier fidelity passed 15/30 observations. Fresh `b7a9d9d` queue started; held-out quality pending |
| T19 Claims and packaging | Operational docs, host matrix, paper errata, packaged UI and helper agents | Source 20f271d passed 669 tests, two WSL skips, Black, strict mypy, JavaScript, build, Twine, and an installed-wheel scope lifecycle smoke. The first broad attempt carries an environment-sync caveat; the unchanged stable repeat is running. Earlier startup-deadline failure remains preserved. Native Codex generated config passed with the new wheel; native Claude current-wheel and complete cross-host acceptance remain open. Real-model quality is pending |

The current capture recommendation is a hybrid: automatically capture enabled
project conversations as retained evidence, extract durable facts separately,
and keep explicit remember/correct/forget available. Capture acknowledgments
must not be presented as accepted-fact acknowledgments. Automatic capture
improves continuity but needs retention, privacy controls, and extraction
evaluation; explicit-only storage is easier to inspect but misses information
an agent does not select. See the [operating guide](../../functional-local-memory.md).

The five modules are independently configurable mechanisms. Passing their unit
contracts does not establish a quality improvement. The frozen v2 lifecycle
dataset mostly contains single-fact sessions and cannot establish graph or
affective ranking utility by itself. The supplemental fixture must have
observed competing candidates; declared seed IDs cannot substitute for actual
embedding ranks or traversal results.

Negative measurements and failed host attempts are retained in the
[progress record](progress.md), [performance report](performance-10k.md), and
[native-host record](native-host-smoke.md). No release, website, paper, live
host configuration, or live memory database has been published or migrated by
this work.
