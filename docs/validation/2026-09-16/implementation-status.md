# Functional local memory implementation status

This record covers the unreleased `wt/functional-local-memory` worktree. It
separates implemented behavior from acceptance evidence. The original task
boundaries and thresholds remain in the [implementation plan](../../plans/2026-09-16-functional-local-memory.md).

The latest package checkpoint is source `6600dc2`. Its frozen verification
passed 865 tests with four platform-specific skips, formatting, typing,
JavaScript, build, and Twine checks. A fresh installed wheel passed 42
contract cases and seven replay assertions. These are package and installed
runtime checks; they do not close native-host, real-model, latency, pilot, or
publication gates. See the [SSD package checkpoint](ssd-package-checkpoint-v1/README.md).

| Plan item | Current implementation and evidence | Acceptance state |
| --- | --- | --- |
| T01 Runtime | Configured providers, SQLite default, explicit mock rejection, readiness checks, health reporting, and installed entry points are covered by the package and wheel checks. | Core runtime tested. Operational doctor, recurring health, and provider-failure artifacts remain separate readiness work. |
| T02 Scope | Profile, project, workspace, membership, enabled-project capture, foreign-project filtering, and legacy/routing review are implemented and covered by contract tests. | Local scope contract tested. Native identity coverage across every host remains open. |
| T03 Durable writes | Atomic source and job creation, leases, retries, CAS, revisions, evidence, rollback, and restart persistence are implemented and tested. | Local durability tested. Long-running operational recovery remains open. |
| T04 Local service | Authenticated loopback ownership, private descriptors, outbox handling, bounded shutdown, and HTTP/subprocess paths are implemented and tested. | Local service and installed replay checks pass. Native host lifecycle remains open. |
| T05 Maintenance | Service-owned scheduling, bounded batches, retry/failure state, retention, and visible maintenance status are implemented and tested. | Deterministic maintenance tested. A daily-use pilot has not run. |
| T06 Evidence and safety | Write firewall, source validation, role labels, governed explicit writes, private paths, retention controls, and local-original availability are implemented and tested. | Safety contracts pass. Real-model extraction precision and human acceptance remain unproved. |
| T07 Surprisal | Automatic gating, explicit-remember bypass, session-safe history, and diagnostics are implemented and covered by deterministic tests. | Mechanism contract tested. Real filtering benefit remains open. |
| T08 Correction | Stable identity, expected-version CAS, revisions, canonical updates, evidence links, and graph refresh are implemented and tested. | Local correction contract passes. Native cross-host correction remains open. |
| T09 Retrieval | Scoped cosine retrieval, lexical candidate union, one scoring contract, and context budgeting are implemented. | Scope and top-1 tests pass. The corrected v4 measurement has recall p95 1,088.094 ms against the 500 ms gate; capture p95 64.279 ms passes the 100 ms gate. Later variable timing comparisons do not establish a speedup. |
| T10 Graph | Persisted mutual top-k links, scope-safe traversal, bounded recovery, insertion-order controls, and capacity diagnostics are implemented and tested. | Structural controls pass. Graph utility and eligible-target coverage are not established by the available evidence. |
| T11 Forget | Exact fact deletion, suppression, replay protection, inflight handling, restore behavior, and selected related-removal preview/apply are implemented and tested. | Local and installed related-forget checks pass. Native lifecycle remains open. Logical forgetting does not erase external transcripts or historical backups. |
| T12 Claude Code | Hook capture, transcript deltas, checkpoints, capture-before-recall, identity diagnostics, and replay repair are implemented. | Native v12 passed health and retrieve calls against the intended project store, plus three hooks, identity, restart, and replay. The v11 pending-server failure is preserved. Native extraction, correction, and forget acceptance remain open. See the [native read archive](../2026-09-17/native-claude-mcp-v12/README.md). |
| T13 Codex | Generated hook configuration, native payload handling, explicit MCP fallback, and bounded transport checks are implemented. | Generated-config canary and restart/replay evidence pass. Full native MCP mutation and cross-host lifecycle acceptance remain open. |
| T14 Hermes | Installed provider integration, registry handling, Windows/WSL bridge behavior, and entry-point discovery are implemented and tested. | Installed bridge checks pass. A native model turn has not passed. |
| T15 UI | Project enable/pause, readiness, facts/history, correction/forget, captured-copy review, job state, and routing review are exposed and covered by API/browser checks. | Local UI and control-plane checks pass. Operational pilot and native-host coverage remain open. |
| T16 Recovery | Backup, integrity, new-path restore, suppression merge, staged reindex, and activation backup behavior are implemented and tested. | Local recovery contracts pass, including a backup during an active writer transaction. Live data migration remains separately authorization-bound. |
| T17 Cross-host | A shared service contract covers Claude, Codex, and Hermes contexts. Synthetic continuity covers teach, restart, recall, correction, forget, and foreign-project filtering. | Synthetic cross-host continuity passes. Complete native matrix and two-week pilot remain open. |
| T18 Evaluation | Frozen lifecycle datasets, request-bound extraction schemas, real-provider adapters, seven profiles, raw traces, checkpoints, Wilson intervals, and paired comparisons are implemented. | Power-interrupted checkpoints were preserved and the unchanged queue resumed after provenance checks. Complete real-model matrix acceptance remains open. Follow the [evaluation status and recovery record](evaluation-running.md). |
| T19 Claims and packaging | Claim ledger, compatibility matrix, release checklist, package artifacts, UI assets, and negative evidence archives are maintained. | Package verification is complete for the stated checkpoint. Publication, website, paper, and release actions remain separately authorization-bound. |

The current capture recommendation is a hybrid: automatically capture enabled
project conversations as retained evidence, extract durable facts separately,
and keep explicit remember/correct/forget available. Capture acknowledgments
must not be presented as accepted-fact acknowledgments. Automatic capture
improves continuity but needs retention, privacy controls, and extraction
evaluation; explicit-only storage is easier to inspect but misses information
an agent does not select. See the [operating guide](../../functional-local-memory.md).

Historical evidence remains available for review without changing the current
state above:

- [Claim and evidence ledger](claim-evidence-ledger.md)
- [Client compatibility matrix](../../client-compatibility-matrix.md)
- [Native plugin checkpoint](native-plugin-checkpoint-v1/README.md)
- [Atomic schema verification](atomic-schema-verification/README.md)
- [Related forget verification](related-forget-verification/README.md)
- [Corrected performance report](performance-corrected-v4/README.md)
- [Progress record, including negative measurements](progress.md)

The package and protocol checks do not promote unrun native, real-model,
latency, pilot, or human-acceptance gates. No live host configuration or live
memory database was published or migrated by this work. Publication remains a
separately authorization-bound action.
