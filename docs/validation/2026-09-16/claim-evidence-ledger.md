# Mnemos claim and evidence ledger

Assessment date: September 17, 2026. This covers the unreleased
functional-local-memory worktree, not the installed public package or website.
The capability IDs come from the original implementation plan.

The latest complete package verification is frozen source `6442044`:
951 tests passed with four skips, plus Black, strict mypy, JavaScript syntax,
build, and Twine. A fresh installed wheel passed 257 selected contract cases
with two matching platform skips and seven synthetic Stop replay assertions. Source, fixtures, and the installed
environment were on C:, outside the development checkout. The
[captured assertion package checkpoint](../2026-09-17/captured-status-package-checkpoint-v1/README.md)
binds the wheel, source, helpers, dependencies, exact test cases, and installed
package origins. It does not establish native plugin-session acceptance.

The [native captured-context observation](../2026-09-17/native-codex-claude-continuity-v2/README.md)
shows a corrected asserted target at version 2 alongside unknown captured
history, visibly qualified in Claude's actual retrieval context. The runner
failed a fixed-cardinality assertion and did not run its second service restart.
The [next attempt](../2026-09-17/native-codex-claude-continuity-v3/README.md)
failed at Codex shutdown before Claude started. These are preserved partial
observations, not passing end-to-end native acceptance.

The client now distinguishes an exited service from a live process reaching
its startup deadline, while preserving capture buffering and an existing
owner. Six startup cases pass in source and the installed wheel. A separate
[installed failure probe](../2026-09-17/startup-exit-diagnostic-v2/README.md)
returned the child's numeric exit code promptly. It does not explain the
earlier intermittent full-suite startup failure.

The lost-acknowledgment repair keeps capture on the durable outbox path when
the service commits an event but its HTTP reply is lost. Four injected HTTP
and SQLite cases cover ordinary and provisional captures, restart, and one
or two lost replies. All four fail before the repair and pass in source and
the installed wheel. The first full-suite attempt at source `6cde465` failed
an auto-start health check; an unchanged isolated diagnostic and a fresh full
run passed afterward. Its cause remains unproven, and the
[failed run](../2026-09-17/lost-ack-package-failure-v1/README.md) remains preserved.

The [concurrent revision regression](../2026-09-17/retrieval-concurrent-revision-v1/README.md)
reproduces stale candidate scoring and context-budget violations while a
correction commits during state classification. The repair reloads candidates
after that await. Six cases cover correction and forget, with graph expansion
on and off; all six fail on the old source and pass in the new source and wheel.
The active real-model run retains its original source and does not measure this
repair. Native Claude v17 remains bound to `8d865a5`; the separate v18 run uses
`dedfcfd` and includes this repair and the Claude startup default.

The earlier complete package checkpoint was frozen source `6600dc2`, with
865 tests passed and 42 installed contract cases. It remains historical
evidence and does not override the newer checkpoint. Source `56182d9` also remains
preserved with 898 passing source tests and 87 passing installed contract cases.

Historical source `312b8d6` passed 858 tests with four skips. Its
[fresh installed-wheel check](combined-package-checkpoint-v2/README.md) then
finished with 40 passing tests and two continuity failures. A subsequent
[installed diagnostic](installed-continuity-diagnostics-v1/README.md) reproduced
HTTP read timeouts and multi-second retrieval metadata updates. A later
[SQL timing profile](installed-continuity-sql-profile-v3/README.md) passed both
cases and located commit as the dominant storage cost in that run; the earlier
multi-second stalls did not recur and no repair is claimed. The separate installed replay
probe did not run. The new C: pass does not prove the earlier D: timeouts cannot
recur or close the 10,000-fact recall gate. The
[drive comparison](storage-drive-calibration-v1/README.md) supports storage
latency as a contributing factor, without establishing an exclusive cause.
The combined `9aba209` run's
benchmark compatibility failure and its repair remain preserved in the
[failed checkpoint](combined-package-failure-v1/README.md). Claude plugin runtime
repair `ec0f95e` passed 50 focused tests with two opt-in skips and is now included
in the `6600dc2` full source checks; native plugin-session acceptance remains open.
The [subsequent native plugin check](native-plugin-checkpoint-v1/README.md)
observed all three lifecycle hooks, seeded recall, MCP connection/tool inventory,
restart persistence, and reconstructed Stop replay. Its original exact-two raw
capture gate still fails because a separate provisional user event remains;
that event's exclusion from fact extraction passes. Native MCP tool execution
and the complete correction/forget conversation remain unproved.
The [v11 native read attempt](../2026-09-17/native-claude-mcp-v11/README.md)
subsequently left the MCP server pending and made no tool calls. Its successful
hook-context recall did not close native MCP retrieval acceptance.
The [v12 native read check](../2026-09-17/native-claude-mcp-v12/README.md)
observed successful health and retrieve calls, matching the intended
database/profile/project and returning the project-scoped seed. Its bounded
read, identity, restart, and replay criteria all passed. An offline diagnosis
of v13 recovered the nested correction payload and verified its target and
replacement hash, but the native attempt still contained degraded capture
diagnostics and did not run restart or forget. The [v15 attempt](../2026-09-17/native-claude-lifecycle-v15/README.md)
reported a pending plugin with zero tools and made no native calls. Native
correction and forgetting were unobserved at that point. Extraction and complete
cross-host continuity remain open.
The separate [hook-time identity observation](../2026-09-17/native-claude-hook-identity-v3/README.md)
found no prompt ID and no transcript file at UserPromptSubmit; identifiers
appeared by Stop. Both capture hooks still emitted diagnostics, so this
association limit remains an open native acceptance condition.
The [v16 startup-wait control](../2026-09-17/native-claude-lifecycle-v16/README.md)
then observed a connected catalog and passed all six native correction-phase
checks. Its direct post-maintenance retrieval failed before restart and forget,
so the complete lifecycle remains failed. Degraded hook capture is retained
separately from the successful native correction operation.
The later [v17 lifecycle check](../2026-09-17/native-claude-lifecycle-v17/README.md)
at installed `8d865a5` passed all 15 fact-lifecycle criteria: native correction,
consolidation, retrieval, forgetting, restart persistence, no resurrection after
reconstructed Stop replay, and preservation of an unrelated fact. Its lexical
vectors are a transport fixture, not a semantic-quality evaluation. Degraded
hook capture keeps the overall result partial; extraction and full native
capture acceptance remain open.
The [v18 lifecycle run](../2026-09-17/native-claude-lifecycle-v18/README.md)
repeated all 15 fact-lifecycle criteria successfully using installed `dedfcfd`,
the unchanged v17 runner, and the new per-server startup default with global
nonblocking startup enabled. Both tool inventories were ready. Capture remained
degraded, so the overall result is partial. A service response logged a
connection abort without an observed fact-criterion failure; its cause is
unproven. No semantic-quality claim follows from the lexical fixture.
The later [exact-query comparison](../2026-09-17/claude-v16-retrieval-diagnosis-v2/README.md)
verified the corrected target remained stored. All fixture embeddings were
identical, and ranking ties plus context budgeting excluded that target.
The broader diagnostic query does not count as a passed native lifecycle.
An implemented capability does not close its quality, native-host, or pilot
acceptance gate.

## Claims and current boundaries

| ID | Claim | Evidence and supported scope | Acceptance still open |
| --- | --- | --- | --- |
| C01 | Persistent local memory | SQLite/service regression checks cover durable events, jobs, restart, retry, and fact persistence. The [SSD source and installed checkpoint](ssd-package-checkpoint-v1/README.md) passes the selected lifecycle contracts. | Behavior under recurring slow-storage stalls, continuous daily operation, and the two-week pilot. |
| C02 | Project, workspace, and global isolation | Scope and lifecycle regressions are included in that package. Registry identities and membership replace folder-name identity. | Complete native-host isolation matrix. Passing fixtures do not prove every external client supplies correct identity. |
| C03 | Surprisal-gated encoding | Mechanism tests cover distance bounds, scoped prediction history, explicit-write bypass, and the later symmetric embedding role. | Real-model retention/noise benefit over baseline and simple filtering. The historical 40 percent claim remains unverified. |
| C04 | Evidence-based mutable memory | Expected-version corrections, revisions, graph refresh, and canonical handling have regression coverage. Source `46cc101` additionally rejects a conflict proposal that would relabel an existing canonical assertion. | General semantic support of extracted proposals. The collision rule is conservative and does not establish entailment. |
| C05 | Affective routing | Scoring, explicit-state handling, fallback, and top-1 execution are tested. Real-model mechanism evaluation is separate from injected-state controls. | Reliable classifier fidelity and a reproducible retrieval benefit within the declared accuracy and latency budgets. |
| C06 | Sleep consolidation and compression | Durable maintenance, evidence-linked processing, retention, and failure/retry machinery are implemented. [Development extraction evidence](preference-contract-development-v1/README.md) preserves omissions and unsupported proposals. | Held-out supported precision, critical retention, and information-preserving compression. A valid JSON response is not a supported fact. |
| C07 | Associative graph retrieval | Traversal, scope, persistence, and insertion-order controls are tested. The [capacity diagnostic](graph-capacity-diagnostic-v2/README.md) completed 15 development observations; successful recall was 0/5 at capacities 4, 8, and 16. | Associative retrieval benefit. Increasing capacity is not an accepted remedy; defaults and evaluation gates remain unchanged. |
| C08 | Independently usable, composable modules | Standalone and integrated module regression checks exercise write filtering, scope, graph state, and lifecycle behavior. | Completed real-model baseline, full-pipeline, and five module-disabled comparisons. Mechanical correctness does not establish usefulness. |
| C09 | Inspectable, correctable, forgettable memory | Facts expose status, version, evidence, and ranking traces. Correction and forget/replay suppression have persistent-service checks. | Complete native correction/forget continuity. Logical forgetting does not erase historical backups or external host transcripts. |
| C10 | Truthful local setup and readiness | Configured runtime, explicit providers, generated interpreter-bound host commands, and input-specific readiness checks are implemented. | Full native installation acceptance. Embedding revision is an operator-managed identity label; the runtime does not automatically verify a mutable Ollama tag's digest. |
| C11 | Shared memory across Codex, Claude Code, and Hermes | [Host records](../../client-compatibility-matrix.md) distinguish authenticated synthetic continuity, installed provider/bridge calls, bounded native capture, and native lifecycle attempts. Claude v12 observed scoped health and retrieve calls; later v13 and v15 attempts retain failed lifecycle evidence. | Complete native lifecycle and Codex Desktop acceptance. The latest [Codex retrieve v2 attempt](../2026-09-17/native-codex-retrieve-v2/README.md) was cancelled by the host before returning a result. A prior native Codex health call proves health transport only; it does not prove mutation or retrieval. |
| C12 | Proven retrieval, compression, or replacement advantage | The [corrected performance run](performance-corrected-v4/README.md) measured recall p95 1,088.094 ms against a 500 ms gate and durable capture p95 64.279 ms against 100 ms. The database after loading was 324,862,864 bytes. | Recall latency passes, completed real-model ablations, retained-information accuracy, and external pilot evidence. No general superiority or replacement claim is accepted. |

## Evidence bindings

The following indices bind raw files by byte count and SHA-256. Their READMEs
identify the source revision, configuration, and limitations. Historical results
remain attributed to the source that produced them.

- [Package verification index](access-metadata-package-verification/evidence-index.json): source `2494e31`, including schema 8. No later embedding-role or conflict-collision code is covered by this full checkpoint.
- [Combined checkpoint index](combined-package-checkpoint-v2/evidence-index.json): source `312b8d6`, passing source/package checks and the failed installed continuity check. The separate installed replay probe did not run.
- [Service/runtime package checkpoint index](../2026-09-17/service-runtime-package-checkpoint-v1/evidence-index.json): source `56182d9`, all six source/package checks, 87 installed contract cases, and seven synthetic Stop replay assertions. This is separate from native host and real-model evidence.
- [Host provenance package checkpoint index](../2026-09-17/host-provenance-package-checkpoint-v1/evidence-index.json): source `8d865a5`, 903 passing source tests with four skips, 110 passing installed contract cases with two matching skips, and seven synthetic Stop replay assertions.
- [Historical SSD checkpoint index](ssd-package-checkpoint-v1/evidence-index.json): source `6600dc2`, all six source/package checks and 42 installed contract cases. It remains separate historical evidence.
- [Native plugin index](native-plugin-checkpoint-v1/evidence-index.json): source `6600dc2`, failed budget-limited v9, v10 native plugin observations and remaining capture gates, plus the installed adapter timing preflight.
- [Installed diagnostic index](installed-continuity-diagnostics-v1/evidence-index.json): valid timeout reproduction plus the excluded first diagnostic whose fixture was incorrectly placed inside Git.
- [SQL timing index](installed-continuity-sql-profile-v3/evidence-index.json): two passing instrumented cases with separate SQLite execute/commit timings; not a repaired or isolated performance result.
- [Performance index](performance-corrected-v4/evidence-index.json): frozen d94 run with real query embeddings and synthetic load facts. It is not a semantic-quality test.
- [Metadata-write comparison index](access-metadata-comparison-v1/evidence-index.json): four fixed-order baseline/candidate arms. Baseline timing varied sharply, so the run does not establish a reliable speedup or close the full recall gate.
- [Graph diagnostic index](graph-capacity-diagnostic-v2/evidence-index.json): frozen d94 source and vectors, three capacities, development cases only, no new model calls.
- [Graph score sensitivity index](graph-weight-diagnostic-v2/evidence-index.json): twenty bound offline development observations. One eligible target enters top five at weight 0.4; four cases remain unreachable. This does not justify a default change or establish held-out utility.
- [Extraction development index](preference-contract-development-v1/evidence-index.json): development evidence, not held-out precision acceptance.
- [Native Codex index](native-codex-lifecycle-v4/evidence-index.json): installed-package preflight and the incomplete native attempt. It does not establish why the client cancelled the call.

The v4 model queue retains its frozen d94 code, model digests, dataset,
options, and output paths. See [evaluation status](evaluation-running.md) for
the power-interruption recovery. Completion and results must be checked before
promoting its running phases to accepted evidence. Later embedding-prefix
comparisons require separate outputs and bindings.

## Publication boundary

The [paper errata](../../paper-errata.md) is a local correction draft. No paper
publishing source has been located in this repository, and the hosted paper,
website, and package have not been changed by this work. Absolute competitor,
infinite-scale, production-readiness, and practical-superiority claims remain
unsupported here. Publication requires separate authorization.
