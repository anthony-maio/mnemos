# Mnemos claim and evidence ledger

Assessment date: September 18, 2026. This covers the unreleased
functional-local-memory worktree, not the installed public package or website.
The capability IDs come from the original implementation plan.

The newer [competing service recovery source check](../2026-09-18/competing-service-recovery-v1/README.md)
fixes a startup race: a second launcher could return a dead descriptor while
the new lock owner was still initializing. Competing launchers now verify
authenticated health before adopting an owner. Five added cases cover the
race, startup timeout, authentication rejection, and invalid health data.
The full source suite passes 1,054 tests with four skips; 47 focused checks
also pass. Simultaneous native host acceptance remains open; the package
checks below validate the installed service protocol.

The current [competing recovery package checkpoint](../2026-09-18/competing-recovery-package-checkpoint-v1/README.md)
at `a5a9101` passes 1,054 source tests with four skips, all six source/package
checks, 384 selected installed cases with two matching skips, and seven
synthetic replay assertions. The installed selection includes the five
competing-launcher regressions. Prior native observations retain their
original wheel bindings; this package does not establish native concurrency
or real-provider recovery.

The preceding [stale descriptor checkpoint](../2026-09-18/stale-descriptor-package-checkpoint-v1/README.md)
at `37e446f` passes 1,049 source tests with four skips, all six source/package
checks, 379 selected installed cases with two matching skips, and seven
synthetic replay assertions. Seven new cases cover delayed replacement,
live-owner reuse, failed or stalled startup, authentication rejection, and
an owner recovering while its competing launcher exits. The client probes
authenticated health before launching a replacement and waits past a known
stale descriptor without deleting it. Earlier native evidence remains bound
to its original wheel; configured real-provider recovery remains open.

The preceding [outbox selection checkpoint](../2026-09-18/outbox-selection-package-checkpoint-v1/README.md)
at `c25ed98` passes 1,042 source tests with four skips, all six source/package
checks, 372 selected installed cases with two matching skips, and seven
synthetic replay assertions. An explicitly supplied empty outbox now retains
its selected recovery file and stricter write policy. Three new cases cover
empty/nonempty selection, restart replay, and PII blocking. This checkpoint
adds package evidence; earlier native observations retain their original
source bindings.

A [concurrent native Codex recovery probe](../2026-09-18/native-codex-concurrent-recovery-v2/README.md)
on the installed `a5a9101` wheel used two separate Codex homes and one shared
outbox. Two outage turns queued four messages. After explicit fixture-service
restoration, two fresh native processes automatically replayed that queue.
A test barrier overlapped delivery of the same event, and Windows TCP ownership
plus process ancestry tied the requests to both native processes. All eight
messages across four sessions had one event, completed job, and evidence link
each. The outbox emptied, both controls survived, and restart preserved the
database. Generated hooks were unchanged and all four native processes exited
zero. The failed v1 home-path fixture is retained. Recovery-a displayed an MCP
startup interruption, and PowerShell emitted Import-Clixml diagnostics; those
PTY observations remain explicit limitations. This closes one bounded Codex
concurrency observation, while cross-host concurrency, real-provider recovery,
complete native acceptance, and the operational pilot remain open.

A [native startup-outage probe](../2026-09-18/native-codex-startup-outage-v1/README.md)
on the installed `c25ed98` wheel retained the real user prompt and final answer
in its outbox while malformed fixture configuration prevented service startup.
The native process exited zero; the original runner failed because the answer
omitted a requested period. That failure is preserved. A separate recovery
helper verified the actual messages, then explicitly started the fixture service
and replayed both events. An immediate second replay had no pending events.
After service restart, replaying the same events left all four core table counts
and complete-row hashes unchanged. Each message had exactly one stored event
and one evidence row. This establishes native outage capture followed by explicit
client recovery with fixture providers. That run did not exercise automatic
native-hook replay; the later two-session observation below does.

In a [fresh two-session probe](../2026-09-18/native-codex-native-recovery-v1/README.md),
the first native Codex session queued both messages during startup failure.
After explicit fixture-service restoration, the second session's unchanged
generated hooks drained that queue automatically, without a direct replay call.
Both processes exited zero. All four messages retained their correct session,
turn, role, project, and profile, with one evidence link each. The outbox emptied,
and complete-row hashes stayed unchanged across maintenance and service restart.
This verifies one native automatic replay observation. The exact triggering hook
was not traced; the separate process-kill observation follows below.

A [forced service-termination probe](../2026-09-18/native-codex-service-kill-v2/README.md)
then killed the actual disposable service after SQLite committed a native prompt
and before its HTTP response. The event and job survived under their original IDs.
After explicit fixture-service restoration replaced the dead owner's descriptor,
a fresh native Codex session automatically drained the queue. All four native
messages had one job and one evidence link each, with unchanged table hashes
across maintenance and restart. Both native processes exited zero. The first
session's Stop timeout and the second session's MCP startup warning remain
recorded as terminal observations; this is not an all-hooks acceptance result.
Arbitrary crash timing, concurrent hosts, real-provider recovery, and complete
host acceptance remain open. The failed v1 PID preflight is preserved separately.

Three fresh [native Codex sessions](../2026-09-18/native-codex-stop-repeat-v1/README.md)
on the `c25ed98` wheel used the generated hooks unchanged. The late native
hook start/completion pairs measured 1,015.009 ms, 1,031.318 ms, and
963.769 ms, each below the unchanged three-second deadline. Their placement
after the final answer and before turn completion identifies them as Stop by
temporal inference; the native log omits hook names and exit status. All three
sessions retrieved the scoped fact, excluded the foreign fact, and persisted
the user prompt and final answer across shutdown. A separate verifier checks
the evidence; the original harness failure reports remain preserved because
Codex added hook-trust metadata after approval. Three fixture observations
do not establish tail latency, the 100 ms durable-capture p95 gate, or complete
native lifecycle acceptance.

The preceding [Windows permission source validation](../2026-09-18/windows-private-acl-v1/README.md)
passes 1,039 tests with four skips and all static checks. Actual Windows ACL
tests exposed an incorrect inheritance-protection flag. The repair applies the
protected current-user policy and checks exact existing permissions before
skipping a write. Nine regressions ran, including later drift and failed-query
repair. Earlier package checks did not establish this protection property.
The [fresh installed package](../2026-09-18/windows-private-acl-package-checkpoint-v1/README.md)
at `5d8cacf` passes 369 selected cases with two skips and seven replay checks.
Its [native v3 probe](../2026-09-18/native-codex-private-acl-profile-v3/README.md)
completed and persisted both roles with protected fixture permissions. Shell
entry-to-exit was 876 ms with instrumentation, versus 2,052 ms in the preceding
old-wheel profile. These are individual observations, not a controlled speedup
estimate or repeated uninstrumented deadline acceptance.

The preceding [installed package validation](../2026-09-18/assistant-suppression-package-checkpoint-v1/README.md)
at `5af3d15` passes 1,030 tests with four skips, all six source/package checks,
360 selected installed cases with two matching skips, and seven synthetic replay
assertions. Exact normalized forget
suppressions now apply before source-role attribution and again at atomic
extraction commit using the persisted event role. Eight regression cases include
independent scope/profile isolation, unrelated proposals, explicit reteaching,
support preservation, and retained unknown paraphrases. Native v9 also includes
this repair; the frozen Ollama evaluation remains separately bound.

The preceding [similarity validation](../2026-09-18/related-forget-similarity-v1/README.md)
passes 1,022 tests with four skips and static checks. It adds bounded advisory
similarity discovery to the explicit related-forget review, with 13 regressions
covering selection, replay suppression, stale state, scope, and dependencies.
The vectors are controlled fixtures. Native acceptance and real embedding
quality are not established, and discovery is explicitly non-exhaustive.

The preceding [UUID source validation](../2026-09-17/uuid-write-safety-v1/README.md)
passes 1,009 tests with four skips, Black, and strict mypy after the shared UUID
write-safety repair. It includes the host-context authority changes. This is a
source checkpoint, now supplemented by the installed verification below.
Native-host evidence remains bound to its earlier source revisions.

The [native v9 run](../2026-09-18/native-codex-claude-continuity-v9/README.md)
on installed `5af3d15` passes 15 bounded lifecycle checks: cross-host correction
survives restart, reviewed deletion removes the target and five captured copies,
direct replay does not restore the target, and final Codex recall and answers
exclude both convention values while preserving the unrelated preference.
The run exercised similarity suggestions with fixture lexical vectors. All
Codex processes exited zero, but reverse and final Stop hooks timed out after
three seconds. Complete capture, native hook replay, Hermes, Desktop, and
real-model quality remain unverified.

The earlier [native v7 attempt](../2026-09-18/native-codex-claude-continuity-v7/README.md)
on installed `ac139dd` committed related deletion before Claude exceeded its
phase budget. SQLite reopen confirms the target and five selected copies absent,
with unrelated and foreign facts retained. Replay and final recall did not run.
[V8](../2026-09-18/native-codex-claude-continuity-v8/README.md) reached
the reverse Codex phase but timed out with tool approval pending. Its deadline
includes operator time; forgetting, replay, and final recall were not run.
Neither attempt establishes native post-forget acceptance.

The earlier [native v6 attempt](../2026-09-17/native-codex-claude-continuity-v6/README.md)
verified selected-fact deletion by Claude and direct service replay suppression
after Codex teaching and cross-host correction. Its final Codex answer still
repeated the deleted value from a surviving unknown captured statement, and the
host skipped the second required query. The overall result is FAILED. This is
not evidence of whole-topic erasure or accepted post-forget host behavior.
The [v5 wrong-query failure](../2026-09-17/native-codex-claude-continuity-v5/README.md)
remains preserved alongside it. The native fixtures do not use real cognition.

The preceding complete package verification was frozen at source `ac139dd`:
1,022 tests passed with four skips and its fresh installed wheel passed 352
selected contract cases. The current `5af3d15` package checkpoint supersedes it
for release binding. Source, fixtures, and installed environments were on C:,
outside the development checkout. Neither package checkpoint establishes native
plugin-session acceptance.
The interruption tests bind child imports to the parent package origin. The
new import guards likewise use fresh subprocesses and the installed package.
The earlier [capture interruption checkpoint](../2026-09-17/capture-durable-package-checkpoint-v1/README.md)
preserves its initial test-origin gap and the supplementary run that repaired it.

The [bidirectional native observation](../2026-09-17/native-codex-claude-continuity-v4/README.md)
on installed ebe1b0d verifies Codex store, Claude read/correct, and a fresh Codex
read of version 2 across two service restarts. Actual native completion records
and independent SQLite reopen support the claim. Foreign-project facts were
excluded. Both Codex processes still exited 0xc0000005, reverse Stop timed out,
and Claude capture remained degraded. No-integration controls exited zero;
the crash trigger remains unproven.
MCP-only and hooks-only controls likewise exited zero. The hooks-only control
retained both conversation roles across database close/reopen despite its
Stop timeout. These are bounded observations with fixture providers; see
[isolation controls](../2026-09-17/native-codex-isolation-controls-v1/README.md).

A later [combined control](../2026-09-17/native-codex-combined-control-v1/README.md)
retained hooks and MCP under explicit unelevated mode, retrieved the scoped
target, retained both capture roles across reopen, and exited zero. Stop still
timed out. One success does not explain or invalidate the earlier crashes.
The [Claude diagnostic repair](../2026-09-17/hook-diagnostic-reasons-v1/README.md)
adds fixed reason codes with 68 passing focused tests; it leaves identity
ambiguity degraded and is not new installed or native acceptance evidence.

The earlier [native captured-context observation](../2026-09-17/native-codex-claude-continuity-v2/README.md)
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
| C11 | Shared memory across Codex, Claude Code, and Hermes | [Native v6](../2026-09-17/native-codex-claude-continuity-v6/README.md) observes Codex store, Claude retrieval/correction/deletion, reverse Codex retrieval before deletion, and direct service replay suppression across restarts. [Host records](../../client-compatibility-matrix.md) preserve preceding failures and separate fixture transport from real-model quality. | V6 FAILED: final Codex reused a surviving unknown captured copy and skipped its second query. Broader related-forget, native answer behavior, complete capture, Hermes, Desktop, and pilot acceptance remain open. |
| C12 | Proven retrieval, compression, or replacement advantage | The [corrected performance run](performance-corrected-v4/README.md) measured recall p95 1,088.094 ms against a 500 ms gate and durable capture p95 64.279 ms against 100 ms. The database after loading was 324,862,864 bytes. | Recall latency passes, completed real-model ablations, retained-information accuracy, and external pilot evidence. No general superiority or replacement claim is accepted. |

## Evidence bindings

The following indices bind raw files by byte count and SHA-256. Their READMEs
identify the source revision, configuration, and limitations. Historical results
remain attributed to the source that produced them.

- [Package verification index](access-metadata-package-verification/evidence-index.json): source `2494e31`, including schema 8. No later embedding-role or conflict-collision code is covered by this full checkpoint.
- [Current assistant-suppression package index](../2026-09-18/assistant-suppression-package-checkpoint-v1/evidence-index.json): source `5af3d15`, 1,030 tests with four skips, six source/package checks, 360 selected installed cases with two matching skips, and seven synthetic replay assertions.
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
