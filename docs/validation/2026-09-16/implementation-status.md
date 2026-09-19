# Functional local memory implementation status

This record covers the unreleased `wt/functional-local-memory` worktree. It
separates implemented behavior from acceptance evidence. The original task
boundaries and thresholds remain in the [implementation plan](../../plans/2026-09-16-functional-local-memory.md).

The newer [Hermes safety checkpoint](../2026-09-19/hermes-safety-checkpoint-v1/README.md)
at `51904d3` passes 1,170 source tests with four skips, all six source/package
checks, 482 isolated installed cases with two matching skips, and seven replay
assertions. It supersedes the package counts below. The
[patched native CLI attempt](../2026-09-19/native-hermes-local-v3/README.md)
passed explicit scope and recall, but failed duplicate capture admission.
Native Hermes acceptance remains open; live host configuration is unchanged.

The latest [recovery checkpoint](../2026-09-19/loopback-recovery-checkpoint-v1/README.md)
at `de6cdcf` passes 1,160 source tests with four skips, all six source/package
checks, 474 isolated installed contract cases with two matching skips, and
seven replay assertions. It removes unused CA-loading overhead from local
HTTP readiness checks, preserves the existing startup polling deadline, and
retries atomic descriptor publication when a Windows reader briefly blocks
replacement. The two preceding full-suite failures remain archived. These
repairs establish package/recovery evidence, not native-host or quality
acceptance. T19 now also includes a working [Substack research handoff](../../handoffs/substack-local-memory.md).

Three fresh [SSD baseline performance runs](../2026-09-19/performance-recovery-v1/README.md)
on `de6cdcf` pass the unchanged gates: warm recall p95 245.214-252.222 ms and
durable capture p95 28.804-36.062 ms, all 150 captures committed. The fixture
uses real query embeddings, synthetic stored vectors and capture provider,
and no cognition calls. Cold timings and earlier failures remain preserved;
full-module latency, semantic quality and native-host gates remain open.

The local model matrix has stopped: five profiles were incomplete because 50
observations lacked an accepted correction target after extraction returned
no durable facts. Baseline and the no-sleep profile completed; completion does
not establish quality success. The [terminal archive](../2026-09-19/lifecycle-incomplete-local-v2/README.md)
preserves all seven reports and raw outputs. Supplemental comparisons did not
start. Earlier references below to running/queued processes are historical.

The [retrieval parity checkpoint](../2026-09-19/retrieval-parity-checkpoint-v1/README.md)
at `8fc79f0` aligns in-memory exact-tie ordering and empty/nonpositive retrieval
requests with SQLite. It passes 1,143 source tests with four skips, all six
source/package checks, and 440 isolated installed contract cases with two skips
and seven replay assertions. The repeated fixed-vector control now returns
identical baseline memory IDs from both stores on all 46 query/repetition
pairs. Both controlled and legacy replacement gates remain failed. This repairs
storage parity without establishing module utility or production superiority.

The [storage-location checkpoint](../2026-09-19/storage-location-checkpoint-v1/README.md)
at `7bcabde` passes 1,131 source tests with four skips and all six source/package
checks. It records benchmark storage paths and protects existing evidence.
Its wheel is byte-identical to `39b9c79`, so the previous 428 installed contract
passes and seven replay assertions are reused by exact hash, not rerun.
The [placement diagnostic](../2026-09-19/capture-placement-v1/README.md) passes
the unchanged 100 ms capture gate in three SSD repetitions (p95 31.795-35.024 ms)
and fails all three HDD repetitions (141.634-521.719 ms). All 300 captures
committed. These are synthetic capture-only results under shared-machine load;
recall latency and native capture acceptance remain open. Earlier failures stay
preserved. The [fair offline comparison](../2026-09-19/fair-comparison-diagnostic-v1/README.md)
removes the earlier cross-project regression with identical vectors and scope
rules, but does not establish a production quality advantage or pass the
replacement gate. The running real-model matrix remains frozen at `fc72cc2`.

The [embedding-client checkpoint](../2026-09-19/embedding-client-checkpoint-v1/README.md)
at `39b9c79` reuses Ollama embedding clients and closes owned providers after
pending work drains, including cancelled and concurrent cleanup. It passes
1,123 source tests with four skips, six source/package checks, 428 installed
contract cases with two matching skips, and replay checks. A fixed-vector HTTP
fixture reduced warm request median from 200.344 ms to 1.540 ms. This is client
setup evidence only; the real recall/capture gates remain failed. The separate
synthetic replacement gate failed both repetitions on both `fc72cc2` and
`39b9c79`, with MRR lift ranging from -0.4 to 0.0. No gate was relaxed, and this
checkpoint is not an all-CI-gates pass. Native observations retain their
original package bindings.

On September 19, the failed lab machine's evaluation was retired with verified
result/checkpoint backups. A [fresh local GPU queue](../2026-09-19/local-evaluation-launch-v2/README.md)
started on clean `fc72cc2` with the same model digests. Default-context warm
readiness passed on the RTX 4080 after a separately recorded 149.623-second
cold load. Evaluation, latency, and host/pilot acceptance remain open; see the
[current execution record](evaluation-running.md). Earlier measurements below
retain their original source and hardware bindings.

The subsequent [local performance phase](../2026-09-19/performance-local-v1/README.md)
failed both gates: recall p95 4,063.309 ms and capture p95 135.186 ms. All 50
captures committed. The timing trace identifies embedding latency as a major
component of the recall tail. A separate client-construction profile identifies
certificate-loading overhead, without establishing a fix or changing the run.
The supplemental comparisons are queued behind successful main completion.

The [local held-out mechanism phase](../2026-09-19/mechanism-local-v1/README.md)
has now completed on the unchanged `fc72cc2` snapshot. Graph structural
correctness passed, but eligibility was zero of 50 cases, failing the minimum
of 40. Synthetic gold-state affective scoring improved top-1 by 34 percentage
points; classifier fidelity passed on only 33 of 300 outputs. These results
do not close graph utility, end-to-end affective benefit, or T18 acceptance.
The main queue has advanced to lifecycle evaluation.

The [mixed-host replay and restart probe](../2026-09-18/native-crosshost-recovery-v4/README.md)
on installed `fc72cc2` passes the bounded native concurrency observation.
Three queued events had overlapping deliveries from Codex and Claude, each
resolving to one durable event and job. All four native turns exited zero;
eight normal messages and two non-extractable provisional records survived
owner restart unchanged. The outbox emptied and both scope controls survived.
The active-waiter shim retains the two-second deadline and records one expired
request. Claude's degraded identity diagnostics remain visible. This closes
that fixture replay/restart observation, while complete host acceptance,
configured-provider recovery, Desktop, Hermes, and pilot gates remain open.

The preceding [mixed Codex/Claude outage probe](../2026-09-18/native-crosshost-recovery-v3/README.md)
on the installed `fc72cc2` package remains a failed concurrency gate: both
recovery processes overlapped, but their selected replay requests missed the
two-second barrier. All four native turns exited zero. Independent read-only
verification found the queue empty, eight normal native messages with one
completed extraction job and fact-evidence link each, and two deferred Claude
provisional records with no extracted facts. Both scope controls survived.
The failed budget and exact-reply attempts are retained. This is partial
recovery evidence; the supervisor's restart gate was not reached, and T17,
real-provider recovery, complete native acceptance, and pilot gates remain open.

The [retrieval strategy checkpoint](../2026-09-18/retrieval-comparison-checkpoint-v1/README.md)
at `fc72cc2` separates vector, lexical, current hybrid, and full retrieval after
one full-pipeline ingestion. Each query and strategy gets an identical accepted
SQLite snapshot with isolated access writes. The 20 new cases cover selection,
scope, budgets, snapshot integrity, missing evidence, and report validation.
The frozen source passes 1,105 tests with four skips and all six repository
checks. Nine adapter cases also pass against the previously verified installed
wheel, whose package bytes and build hash are unchanged. The separate offline
preparation remains `NOT RUN`; this does not close real-model comparison,
module utility, native-host, latency, or pilot gates.

The [applied encoding comparison checkpoint](../2026-09-18/encoding-comparison-checkpoint-v1/README.md)
at `aefc338` adds the missing length/noise replacement arm alongside full and
surprisal-disabled controls. The filter runs inside durable extraction jobs,
preserves raw capture, bypasses explicit lifecycle actions, and records actual
decisions. The frozen source passes 1,085 tests with four skips and all six
source/package checks. The wheel passes 390 existing cases with two matching
skips, seven replay assertions, and ten additional comparator service cases.
All three prepared experiment artifacts remain `NOT RUN`. This supplies the
implementation and offline evidence; real-model comparison, paired cost and
latency interpretation, and T18 acceptance remain open.

The [storage-worker termination checkpoint](../2026-09-18/storage-process-kill-v1/README.md)
adds seven cases at `76940d2`: capture after event insertion, before commit,
and after commit; forget after suppression insertion, after source deletion,
before commit, and after commit. The actual SQLite worker is terminated.
Fresh reopen, 12-table hashes, integrity checks, repeated outbox replay,
derived-state cleanup, and retained local/foreign controls pass. The clean
source suite passes 1,067 tests with four skips, and all seven new cases
pass against the existing `983920a` wheel. Runtime/build Git objects and
installed wheel bytes are verified unchanged. These fixed-provider tests
cover selected storage boundaries; native host, provider, commit-instruction
interruption, and operational-pilot acceptance remain separate.

The capture/forget race repair checks suppression inside SQLite's capture
transaction. Previously, a capture or pending replay could pass the service
check, wait while a forget committed, then restore the erased raw source.
The final gate now checks source identity, verified Claude legacy aliases,
automatic assertion identity, and an explicit correction's forgotten target.
Suppressed captures acknowledge suppression without inserting an event or job.
A fresh explicit remember can still intentionally restore an assertion.
Six new cases cover these boundaries and database reopen. The focused suite
passes 62 tests. The [983920a package checkpoint](../2026-09-18/capture-forget-package-checkpoint-v1/README.md)
passes 1,060 source tests with four skips, all six source/package checks,
390 installed cases with two matching skips, and seven replay assertions.
The first full run had a degraded bridge result; three isolated diagnostics
and the unchanged full rerun passed. The original failure is retained and
its cause remains unresolved. Earlier native observations and the frozen
real-model evaluation keep their original source bindings.

The newer [competing service recovery source check](../2026-09-18/competing-service-recovery-v1/README.md)
fixes a startup race: a second launcher could return a dead descriptor while
the new lock owner was still initializing. Competing launchers now verify
authenticated health before adopting an owner. Five added cases cover the
race, startup timeout, authentication rejection, and invalid health data.
The full source suite passes 1,054 tests with four skips; 47 focused checks
also pass. Simultaneous native host acceptance remains open; the package
checks below validate the installed service protocol.

The preceding [competing recovery package checkpoint](../2026-09-18/competing-recovery-package-checkpoint-v1/README.md)
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
concurrency observation. The later mixed-host v4 observation above adds bounded
Codex/Claude evidence; real-provider recovery, complete native acceptance, and
the operational pilot remain open.

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

The preceding [Windows private-permission repair](../2026-09-18/windows-private-acl-v1/README.md)
passes 1,039 source tests with four skips, Black, strict mypy, and JavaScript
syntax. It corrects the Windows DACL protection flag and avoids repeated
permission writes only when a fresh descriptor check exactly matches the
protected current-user policy. Nine real Windows regressions cover inheritance,
legacy locks, later permission changes, and query failures. The fresh
[installed package](../2026-09-18/windows-private-acl-package-checkpoint-v1/README.md)
at `5d8cacf` passes 369 selected cases with two matching skips and seven replay
assertions; all six frozen-source/package checks pass.
The [native profile v3](../2026-09-18/native-codex-private-acl-profile-v3/README.md)
on that wheel completed with protected fixture permissions and both conversation
roles persisted. Its instrumented shell duration was 876 ms, adapter import
370 ms, and handler 373 ms. That single observation did not establish repeated
completion. The later three-session observation is recorded above; tail latency
and the broader host lifecycle remain open.

The preceding [installed package checkpoint](../2026-09-18/assistant-suppression-package-checkpoint-v1/README.md)
at `5af3d15` passes 1,030 tests with four skips, Black, strict mypy,
JavaScript syntax, build, and Twine. Its fresh installed wheel passes 360
selected contract cases with two matching skips and all seven synthetic replay
assertions.
Automatic extraction now checks forgotten assertions before adding source-role
attribution. Atomic commit repeats the check using the persisted event role;
candidate metadata cannot bypass it. Eight regressions cover scope and profile
isolation, mixed extraction batches, explicit reteaching, support preservation,
and the non-exact paraphrase boundary. Native v9 uses this installed package;
the frozen Ollama evaluation remains separately bound.

The preceding [similarity checkpoint](../2026-09-18/related-forget-similarity-v1/README.md)
passes 1,022 tests with four skips, Black, strict mypy, and JavaScript syntax.
Related-forget previews now supplement exact matches with bounded, advisory
similarity suggestions from stored captured-fact vectors. Explicit selections,
scope checks, source dependencies, and stale-preview rejection remain required.
The controlled-vector tests establish the deletion contract, not embedding
quality or native-host acceptance. Unindexed episodic paraphrases and
historical-only meanings may still be missed.

The preceding [similarity package checkpoint](../2026-09-18/related-similarity-package-checkpoint-v1/README.md)
validated source `ac139dd` with 1,022 tests and 352 selected installed cases.
It remains useful historical evidence for similarity review, UUID write safety,
and host-context authority, but the `a5a9101` checkpoint is the current package
binding. Native acceptance and the frozen real-model evaluation remain separate,
open gates.

The preceding [UUID write-safety checkpoint](../2026-09-17/uuid-write-safety-v1/README.md)
passes the full suite: 1,009 passed and four skipped, with Black and strict mypy
also passing. The shared privacy filter no longer treats numeric matches wholly
inside a canonical UUID as personal data. Other numbers, matches crossing the
UUID boundary, email, and credential assignments retain their existing checks.
The regression exercises outbox identity and durable remember/correct/replay.
This repairs the deterministically reproduced generated-ID rejection below;
the specific cause of the earlier full-suite failure remains unproven because
that run did not retain its generated ID. Installed-package validation now passes
as recorded above; the later native attempts remain incomplete.

The earlier host-context repair preserves recall authority at the host boundary.
Claude's adapter had discarded the service's assertion-status and history labels;
Codex and Hermes discarded them in fact-only compatibility responses. The shared
formatter now preserves service context, labels and quotes fallback records,
respects an empty context, and omits oversized context rather than emitting a
partial record. The legacy CLI formatter also retains the shared qualifications.
Nine adapter regressions failed before the repair; 97 focused adapter/import
checks passed afterward, followed by 15 authority tests including three actual
SQLite-backed service-to-adapter checks. Black and strict mypy pass. The
[full-suite run](../2026-09-17/hook-context-authority-v1/README.md) FAILED with
981 passed, four skipped, and one failure: the captured-history restart test
found no fact after its initial remember/maintenance step. The retained fixture
also had no events or jobs. The same test passed in isolation. A separate
[UUID metadata probe](../2026-09-17/uuid-metadata-rejection-v1/README.md)
reproduced rejection of a generated UUID whose numeric tail matched the
credit-card filter. This is a confirmed write defect, but the failed full run
did not retain its generated ID, so its specific cause remains unproven.
The latest installed checkpoint includes this repair. The incomplete later
native attempts do not close the native v6 failure below.

The [native v9 run](../2026-09-18/native-codex-claude-continuity-v9/README.md)
uses installed `5af3d15` and passes 15 bounded lifecycle criteria. Codex teaching,
Claude correction, and fresh Codex retrieval of version 2 survived service
restarts. Claude reviewed and deleted the target and five captured copies,
including similarity suggestions. Direct replay after restart did not restore
the target. Final Codex made both required queries, excluded the deleted values
and foreign fact, and retrieved the unrelated retained preference. All Codex
processes exited zero. Reverse and final Stop hooks still timed out after three
seconds; complete capture, native hook replay, Hermes, Desktop, real-model
quality, and the operational pilot remain open. The fixture uses lexical vectors
and disabled modules, so this is a bounded host-lifecycle observation.

Two preceding native attempts used installed `ac139dd`.
[V7](../2026-09-18/native-codex-claude-continuity-v7/README.md) committed
related deletion: SQLite reopen confirms the target and five selected copies
absent, with unrelated and foreign facts retained. Claude exceeded its phase
budget; replay and final Codex recall did not run.
[V8](../2026-09-18/native-codex-claude-continuity-v8/README.md) stopped
at the reverse Codex phase with tool approval pending. Its 180-second deadline
includes operator time, so this is not a product-latency result. Forgetting,
replay, and final recall did not run. Native acceptance remains open.

The [earlier native forget/replay attempt](../2026-09-17/native-codex-claude-continuity-v6/README.md)
is FAILED overall. Codex taught the fact, Claude corrected it, and fresh Codex
retrieved version 2 across restarts. Claude then deleted it and retrieved an
unrelated retained preference. Direct service replay after restart suppressed
the original teaching event and rejected correction of the deleted target;
maintenance and another restart preserved deletion. The final Codex session
retrieved a surviving unknown captured mention, repeated its value as current,
and skipped the second requested control query. Its native post-forget control
and answer behavior therefore failed acceptance. All three Codex processes exited
zero under explicit unelevated mode; reverse and final Stop timeouts remain.

The [v5 failure](../2026-09-17/native-codex-claude-continuity-v5/README.md)
is also preserved. Claude used the wrong query for the unrelated-fact control;
the fact remained in SQLite. V6 checked the exact query rather than accepting
the model's final prose. These observations require investigation of historical
capture presentation and instruction boundaries. Current labels alone did not
prevent the observed stale answer.

The [earlier native continuity check](../2026-09-17/native-codex-claude-continuity-v4/README.md)
observed both directions on installed `ebe1b0d`: Codex stored, Claude retrieved
and corrected after restart, then a fresh Codex session retrieved version 2
after another restart. All three native queries excluded the foreign canary.
Both Codex processes exited 0xc0000005; reverse Stop timed out and Claude capture
remained degraded. Two no-integration controls exited zero. The cause remains open.
Separate [MCP-only and hooks-only controls](../2026-09-17/native-codex-isolation-controls-v1/README.md)
also exited zero. The hooks-only control retained both capture roles after
database reopen despite a Stop timeout. Sandbox mode and tool-approval
differences prevent assigning the combined-run crash to one component yet.

A [combined control](../2026-09-17/native-codex-combined-control-v1/README.md)
with all generated hooks, MCP, and explicit unelevated sandbox mode completed
native scoped retrieval and exited zero. Two hook events survived SQLite reopen.
Stop still timed out, and operator/tool-approval time prevents using the elapsed
time as a performance result. The earlier crashes remain unresolved.

The current source adds allowlisted Claude diagnostic reasons for missing
identity and ambiguous transcript association. Its [focused regression](../2026-09-17/hook-diagnostic-reasons-v1/README.md)
passed 68 tests and strict typing. It preserves separate captures and degraded
status. This change is newer than the installed ebe1b0d native controls and has
not received a new installed-package or native-host acceptance run.

The [capture import boundary](../2026-09-17/hook-import-boundary-v1/README.md)
now keeps engine, embedding, and storage imports out of client startup. All 92
focused cases passed with two platform skips, and strict product typing passed.
Public exports resolve on first access. Diagnostic shell timings remain variable;
the native Stop deadline is not yet accepted.
The archived generated-command probe already used PowerShell with `-NoLogo`,
`-NoProfile`, and `-NonInteractive`. Its slow timings therefore do not support
adding those flags as a new fix. Two actual native diagnostic runs now record
those boundaries: [boundary v1](../2026-09-18/native-codex-stop-boundary-v1/README.md)
and [profile v2](../2026-09-18/native-codex-stop-profile-v2/README.md). Both exited
cleanly and persisted their user and assistant captures. Instrumented shell
durations were 2,123 and 2,052 ms; the profile attributed about 1.175 seconds to
permission updates, while service capture took 2.6 ms. The old wheel's lock
still inherited permissions, leading to the source repair above. These runs
do not measure host dispatch to shell entry or establish uninstrumented latency
acceptance. The newer installed v3 probe above also completed, with a lower
observed duration and protected permissions. Earlier Stop timeouts remain
preserved; one newer observation does not establish consistent deadline behavior.
The [complete package checkpoint](../2026-09-17/lazy-import-package-checkpoint-v1/README.md)
at `ebe1b0d` passed 962 tests with four platform skips and all static/build
checks. Its fresh installed wheel passed 268 selected cases with two matching
skips and seven replay assertions, including subprocess import provenance.

The [capture interruption regression](../2026-09-17/capture-interruption-v1/README.md)
forces a host client process to exit before sending or after service commit.
Capture now persists sanitized evidence before startup and HTTP waits, then
removes it only on a terminal reply. Both interrupted paths replay once after
SQLite/service restart. Seven new cases and 84 related cases pass, including
forget suppression before interrupted capture is acknowledged. Native Stop
timeout and host shutdown acceptance remain open.

The [captured-assertion repair](../2026-09-17/captured-assertion-regression-v1/README.md)
preserves automatic passthrough history while labeling newly indexed copies
as unknown captured evidence. Host context now marks it unverified and possibly
outdated. Canonical dedupe preserves explicit authority and source identities,
and related-forget review still includes captured copies. All 68 focused
regressions pass. This repairs evidence labeling; it does not suppress old
mentions, migrate existing records, or prove native answer selection.
The [native qualification observation](../2026-09-17/native-codex-claude-continuity-v2/README.md)
returned the corrected target asserted at version 2 and captured history marked
unknown with the qualifier visible in Claude's actual context. That runner
failed a fixed two-old-mentions assertion after receiving one, before its second
service restart. A [revised rerun](../2026-09-17/native-codex-claude-continuity-v3/README.md)
then failed during Codex shutdown before reaching Claude. Both reports remain
failed; the earlier observation is narrower than end-to-end acceptance.

The [native Codex-to-Claude continuity check](../2026-09-17/native-codex-claude-continuity-v1/README.md)
now proves an actual Codex store/consolidate call, Claude retrieval after service
restart, and Claude correction preserving the fact ID at version 2 through
another restart. Both native Claude queries excluded the foreign canary.
The corrected query also returned two independently captured records repeating
the old value. Stable-target correction passes; stale conversation context is
an open correctness gap. The fixture disabled all modules, so investigate the
verbatim automatic-capture fallback without treating this as an extraction
quality result. Codex exited nonzero after /exit, and Claude capture remained
degraded. Reverse recall, cross-host forget/replay, and Hermes remain open.

Source `1068a62` repairs non-ASCII capture in both host adapters by decoding
hook JSON as UTF-8 rather than the Windows stdin locale. Five regressions
failed before the repair, and 75 related cases pass afterward. The
[native Codex rerun](../2026-09-17/native-codex-utf8-v1/README.md) captured the
exact Unicode assistant response and user prompt through generated hooks;
both were present after service closure and database reopen. Actual scoped
MCP retrieval passed too. Shutdown still reached its watchdog, and native
mutation, cross-host, and complete conversation coverage remain open.

The current adapter also reports degraded capture when a Codex capture hook
has no usable text. Nine regression cases failed before this change; all ten
new cases and related adapter/configuration checks now pass (50 passed, two
platform skips). This repairs a separate reporting defect; the later Unicode
failure and native rerun are recorded above. See the
[coverage regression](../2026-09-17/codex-capture-coverage-v1/README.md).

Source `d58e082` repairs an argument mismatch observed in a native Codex
interactive call: MCP retrieval now accepts structured scope lists as well as
legacy comma-separated strings. Empty or invalid lists are rejected, and the
stdio regression verifies project isolation. The [native failure](../2026-09-17/native-codex-interactive-v1/README.md)
and [regression evidence](../2026-09-17/mcp-scope-arguments-v1/README.md) are preserved.
The complete package verification below includes this repair. The subsequent
[native interactive check](../2026-09-17/native-codex-interactive-v2/README.md)
returned the scoped seed through an approved MCP call against installed
`30eaf42`. Bounded native retrieval passes. Shutdown timed out and assistant
capture was absent, so complete conversation and lifecycle acceptance remain open.

An earlier capture-durability package checkpoint used source `b7b4d6b`. Its frozen verification
passed 958 tests with four platform-specific skips, formatting, typing,
JavaScript, build, and Twine checks. A fresh installed wheel passed 264
contract cases with two matching platform skips and seven replay assertions. These are package and installed
runtime checks; they do not close native-host, real-model, latency, pilot, or
publication gates. A seven-case supplement explicitly binds interruption-test
children to the installed package after correcting a source import in the test.
See the [capture interruption package checkpoint](../2026-09-17/capture-durable-package-checkpoint-v1/README.md).

This checkpoint repairs retrieval across a concurrent correction or forget:
candidates are refreshed after asynchronous state classification, so scoring,
graph expansion, and budgeting use current records. All six service-level
regressions fail on the prior source and pass with the repair. See the
[concurrency reproduction](../2026-09-17/retrieval-concurrent-revision-v1/README.md).

The Claude startup default at `dedfcfd` is included in this complete source and
installed-package checkpoint. A separate isolated native `alwaysLoad` control,
using an older wheel with a manifest overlay, passed nine startup/routing
criteria. See the
[startup control](../2026-09-17/native-claude-alwaysload-v2/README.md).

Source `6cde465` adds capture recovery when the service commits an event but
loses its HTTP response. The client now handles `httpx.RemoteProtocolError`
through the existing durable outbox path. Four injected HTTP/SQLite cases
failed before the repair and passed afterward; 35 related service cases pass.
All four recovery cases also pass in the complete source and installed-package
checkpoint above. See the [lost-acknowledgment regression](../2026-09-17/service-lost-ack-regression-v1/README.md).

The first full run of `6cde465` stopped with 917 passed, four skipped, and one
failed service auto-start check. Its initial health response was `degraded`;
the preserved fixture had no runtime directory or service log at inspection.
The unchanged isolated test subsequently passed, with health returning in
952.09 ms under the original ten-second startup limit. This does not establish
the cause of the full-run failure. Both the [failed checkpoint](../2026-09-17/lost-ack-package-failure-v1/README.md)
and [isolated diagnostic](../2026-09-17/lost-ack-startup-diagnostic-v1/README.md)
remain separate evidence.

Source `a607a5a` distinguishes an exited auto-start process from a live process
that reaches its startup deadline. The client reports a numeric exit code
promptly, accepts a competing owner descriptor, and preserves queued capture.
The [before-fix installed probe](../2026-09-17/startup-exit-diagnostic-v1/README.md)
observed a child exit at 652.265 ms followed by a generic client response at
2,048.842 ms. Six deterministic startup cases and 29 related service cases
pass with the repair. See the [regression record](../2026-09-17/startup-exit-regression-v1/README.md).
The [after-fix installed probe](../2026-09-17/startup-exit-diagnostic-v2/README.md)
observed a child exit at 655.468 ms and an explicit exit-code response at
686.713 ms under the same two-second deadline. This diagnostic repair is
included in the package checkpoint above. It does not establish the cause of
the earlier full-suite startup failure.

The later [access-metadata phase probe](../2026-09-17/access-metadata-phases-v1/README.md)
times the current storage implementation on two fresh 10,000-fact fixtures.
Across 100 five-fact updates on D:, total p95 was 313.7343 ms and commit p95
was 313.5566 ms; the SQL stages stayed below 0.5 ms per call. The same helper
on C: reported total p95 2.58 ms and commit p95 2.4211 ms. Both retained
WAL/FULL and all counters after reopen. The D-then-C order and concurrent
work prevent a controlled speedup claim. This localizes the observed cost to
the durability boundary and does not close the end-to-end recall gate.

A [33-call follow-up on current source](../2026-09-17/retrieval-commit-profile-v1/README.md)
again found commit dominated the write cost: total p95 86.372 ms and commit
p95 86.213 ms on D:, with WAL/FULL retained and all counters verified after
reopen. This smaller follow-up did not reproduce the original roughly 796 ms
metadata tail. It adds an observation of timing variability; it is not an
end-to-end recall pass or a measured product speedup.

An earlier full-suite attempt at `e1fd56e` finished with 897 passed, one failed,
and four skipped. A list response in the synthetic Codex continuity fixture
lacked `facts`; its cause is under investigation. The runner stopped before
building a new wheel. The [failed checkpoint](../2026-09-17/service-runtime-checkpoint-failure-v1/README.md)
remains separate from the earlier passing package checkpoint.

The [isolated rerun](../2026-09-17/crosshost-isolated-diagnostic-v1/README.md)
passed that exact parameter with its original timeout and durability settings.
All 40 traced requests completed, including a list response with one fact.
It did not reproduce or explain the full-suite failure.

The next full run at `33c6dd9` also finished with 897 passed, one failed,
and four skipped. The continuity case passed; the generated PowerShell hook
bootstrap exceeded its existing ten-second deadline. The
[second failed checkpoint](../2026-09-17/powershell-checkpoint-failure-v1/README.md)
is preserved, and no new wheel was built from it. Two dashboard unit tests
now inject client facades to avoid contacting the default service; their
16 focused control-plane/UI cases pass. This independent isolation repair
does not establish the cause of either earlier timeout or response failure.

The [Claude capture diagnosis](../2026-09-17/claude-capture-association-diagnostic-v1/README.md)
reproduced the provisional-user association warning when the prompt hook lacks
a native prompt ID. A completed-transcript control succeeds when supplied that
ID. The [native hook-time probe](../2026-09-17/native-claude-hook-identity-v3/README.md)
then observed no prompt ID and no transcript file before or after the actual
prompt hook. The transcript appeared by Stop with native identifiers. Reading
the transcript earlier cannot resolve that observed turn's association;
capture diagnostics and complete lifecycle acceptance remain open.

| Plan item | Current implementation and evidence | Acceptance state |
| --- | --- | --- |
| T01 Runtime | Configured providers, SQLite default, explicit mock rejection, readiness checks, health reporting, and installed entry points are covered by the `a5a9101` package and wheel checks. Windows private-path inheritance and permission drift are now explicitly tested. | Installed doctor cases cover configuration states but do not substitute for a live provider reachability probe. Managed service logging, omitted-provider startup, and generated host provenance are also covered. |
| T02 Scope | Profile, project, workspace, membership, enabled-project capture, foreign-project filtering, and legacy/routing review are implemented and covered by contract tests. | Local scope contract tested. Native identity coverage across every host remains open. |
| T03 Durable writes | Atomic source and job creation, leases, retries, CAS, revisions, evidence, rollback, and restart persistence are implemented and tested. | Local durability tested. Long-running operational recovery remains open. |
| T04 Local service | Authenticated loopback ownership, private descriptors, outbox handling, bounded shutdown, and HTTP/subprocess paths are implemented and tested. | Local service and installed replay checks pass, including supplied-outbox policy preservation, stale-descriptor replacement with authenticated live-owner checks, and competing launchers waiting for a ready owner. Native startup-outage capture survives explicit client replay without duplicates. A separate fresh Codex session automatically replayed the queued events after fixture-service restoration, preserving both sessions across restart. A later forced service kill after native capture commit preserved the original event/job IDs and recovered automatically through fresh native hooks after explicit service restoration. A later two-home Codex probe recovered a shared outbox under instrumented, process-attributed replay overlap without duplicated records. Mixed-host v4 subsequently verified three overlapping Codex/Claude replay pairs and unchanged records through restart on installed `fc72cc2`. Arbitrary crash timing, real-provider recovery, and complete host acceptance remain open. |
| T05 Maintenance | Service-owned scheduling, bounded batches, retry/failure state, retention, and visible maintenance status are implemented and tested. | Deterministic maintenance tested. A daily-use pilot has not run. |
| T06 Evidence and safety | Write firewall, source validation, role labels, governed explicit writes, private paths, retention controls, and local-original availability are implemented and tested. | Safety contracts pass. Real-model extraction precision and human acceptance remain unproved. |
| T07 Surprisal | Automatic gating, explicit-remember bypass, session-safe history, and diagnostics are implemented and covered by deterministic tests. | Mechanism contract tested. Real filtering benefit remains open. |
| T08 Correction | Stable identity, expected-version CAS, revisions, canonical updates, evidence links, and graph refresh are implemented and tested. | Native Claude corrected a Codex-stored fact after restart, preserving its ID at version 2; a fresh Codex session retrieved it after another restart. Captured old mentions remain unknown and visibly qualified as possibly outdated. General stale-context answer consistency remains open. |
| T09 Retrieval | Scoped cosine retrieval, lexical candidate union, one scoring contract, and context budgeting are implemented. | Scope and top-1 tests pass. The original local run fails recall p95 4,063.309 ms against 500 ms and capture p95 135.186 ms against 100 ms. Subsequent capture-only diagnostics pass all three SSD repetitions (31.795-35.024 ms) and fail all three HDD repetitions (141.634-521.719 ms), with 300 committed captures under shared-machine load. Recall and native capture remain open. The prior lab measurement (1,088.094 ms recall, 64.279 ms capture) remains preserved. |
| T10 Graph | Persisted mutual top-k links, scope-safe traversal, bounded recovery, insertion-order controls, and capacity diagnostics are implemented and tested. | Local held-out structural controls pass, but zero of 50 graph cases are eligible against the minimum of 40. Graph utility remains unmeasured; see the [negative result](../2026-09-19/mechanism-local-v1/README.md). |
| T11 Forget | Exact fact deletion, suppression, replay protection, inflight handling, restore behavior, and selected related-removal preview/apply are implemented and tested. | Native v9 passes reviewed deletion of the target plus five selected copies, direct replay suppression, and final Codex recall without the removed values. Unindexed paraphrases and external transcripts remain outside that bounded result. Earlier v6's surviving-copy failure is preserved. |
| T12 Claude Code | Hook capture, transcript deltas, checkpoints, capture-before-recall, identity diagnostics, and replay repair are implemented. | Native v18 on dedfcfd passed all 15 fact-lifecycle criteria with the new startup default, including correction, forgetting, restart, and reconstructed replay. Degraded hook capture keeps the overall result partial. Mixed-host v4 adds bounded native queue replay and restart on `fc72cc2`, with identity warnings preserved. Native extraction and complete capture acceptance remain open. See the [native lifecycle archive](../2026-09-17/native-claude-lifecycle-v18/README.md). |
| T13 Codex | Generated hooks, UTF-8 payload handling, explicit MCP fallback, and bounded native continuity are implemented. | Native v9 passes teaching, corrected recall, and final post-forget recall; all three processes exited zero, but two Stop hooks timed out. The Windows ACL repair passes nine installed regressions and the installed v3 native probe persisted both roles with protected permissions and an 876 ms instrumented shell duration. Three uninstrumented sessions on c25ed98 also persisted both roles, with late native hook pairs below three seconds; Stop identity is inferred from turn timing. A separate two-session probe on this wheel automatically replayed outage captures through generated hooks and retained all four messages across restart. Mixed-host v4 adds bounded Codex/Claude concurrent replay and restart on `fc72cc2`. Tail latency, complete capture, and Desktop remain open. |
| T14 Hermes | Installed provider integration, registry handling, Windows/WSL bridge behavior, and entry-point discovery are implemented and tested. | Installed bridge checks pass. A native model turn has not passed. |
| T15 UI | Project enable/pause, readiness, facts/history, correction/forget, captured-copy review, job state, and routing review are exposed and covered by API/browser checks. | Local UI and control-plane checks pass. Operational pilot and native-host coverage remain open. |
| T16 Recovery | Backup, integrity, new-path restore, suppression merge, staged reindex, and activation backup behavior are implemented and tested. | Local recovery contracts pass, including a backup during an active writer transaction. Live data migration remains separately authorization-bound. |
| T17 Cross-host | Native v9 on installed `5af3d15` passes 15 bounded lifecycle checks for Codex teaching, Claude correction and reviewed deletion, restart, direct service replay suppression, foreign-project exclusion, final recall, and retained unrelated facts. The newer mixed-host v4 probe on `fc72cc2` verifies three paired same-event native deliveries, durable queue recovery, and unchanged records through owner restart. | Lexical fixture vectors, disabled modules, explicit restoration, and a test rendezvous limit the claim. V4 retains one pairing timeout and degraded Claude identity diagnostics; the earlier v3 gate remains failed. Earlier Codex Stop timeouts remain recorded. Complete capture, Desktop, Hermes, real-provider recovery, real-model quality, and the two-week pilot remain open. |
| T18 Evaluation | Frozen lifecycle datasets, request-bound extraction schemas, real-provider adapters, seven profiles, raw traces, checkpoints, Wilson intervals, and paired comparisons are implemented. Separate applied length/noise and vector/lexical/hybrid/full comparisons are implemented and prepared offline. | The frozen real-model matrix and both additional comparison executions remain open. Prepared artifacts have no real-model observations yet. Graph/affective utility evidence also remains a separate requirement. Follow the [evaluation status and recovery record](evaluation-running.md), [applied encoding checkpoint](../2026-09-18/encoding-comparison-checkpoint-v1/README.md), and [retrieval checkpoint](../2026-09-18/retrieval-comparison-checkpoint-v1/README.md). |
| T19 Claims and packaging | Claim ledger, compatibility matrix, release checklist, package artifacts, UI assets, and negative evidence archives are maintained. | The latest source/package checkpoint is `8fc79f0`, with 1,143 source passes and 440 newly observed installed contract passes. Both synthetic replacement comparisons remain failed despite repaired baseline storage parity. Real-model and native acceptance remain separate. The preserved intermittent bridge failure remains unexplained. Publication, website, paper, and release actions remain separately authorization-bound. |

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
