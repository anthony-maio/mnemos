# Functional local memory implementation record

Worktree: `D:/Development/mnemos/mnemos/.worktrees/functional-local-memory`
Branch: `wt/functional-local-memory`
Base: `5fcc925d5a90a087e5715c2fe6de56fa23a7d3a6`

This is a chronological checkpoint log. Later entries supersede earlier test
counts and open implementation items. An implementation or focused check does
not close a native-host, quality, performance, or pilot gate.

## Observed results

- Initial isolated Python 3.13 suite: 323 passed, 4 failed. MCP is now constrained to >=1,<2; the website consistency test locates the outer website checkout.
- Latest broad implementation snapshot: 419 passed, 2 failed. Failures concern the updated Claude hook manifest and inspection requiring an observed retrieval trace. Work remains active; this is not the final suite result.
- Scope and exact retrieval changes: SQL predicates precede candidate selection, native sqlite-vec uses cosine consistently, FTS5 provides bounded additional candidates, and the extension-free fallback loads only scoped rows. Native/fallback canaries and schema-4-to-5 preservation tests passed.
- Mutable review requires source IDs, profile/scope validation, and expected-version compare-and-swap. Query-only recall does not rewrite facts or queue reviews. Targeted module and durable review checks passed (63 tests).
- Authenticated loopback service has an owning async loop, process singleton lock, Windows private DACL, private descriptor, sanitized durable outbox, and restart checks. Its 9 transport tests passed repeatedly. Normal CLI/MCP/UI entry point migration remains open.
- Domain capture, explicit remember/correct, maintenance leases, automatic typed extraction, profile/session partitioning, and source excerpts are implemented. Automatic correction/support integration and all-module switches were added after the broad snapshot; further checks remain open.
- Forget/restart/replay/pending-correction and processed-evidence retention checks passed. Explicit fresh remember can restore a forgotten assertion; replay cannot. This is live-store logical deletion, not physical deletion from backups or external transcripts.
- Graph startup reads are bounded; query-time neighborhood loading covers persisted nodes beyond the startup page. Existing graph/storage/engine checks passed (67 tests). Graph format migration and the new large-store path still need dedicated validation.
- Ollama `http://192.168.86.35:11434` responded with version 0.34.1. Discovery and complete model digests are in `ollama-discovery.json`.
- Actual `gemma3:4b` generation and `nomic-embed-text:latest` embeddings passed the provider smoke (`ollama-smoke.json`). Cold generation took 62.205 seconds; two 768-dimensional embeddings took 5.941 seconds.
- Actual domain-service remember -> durable processing -> SQLite restart -> retrieve passed (`service-provider-smoke.json`). Capture: 0.76 ms; processing: 8.673 seconds; baseline retrieval: 414.83 ms; biomimetic retrieval: 3784.26 ms. This is a single synthetic fact, not a latency or quality acceptance gate.
- Evaluation dataset/harness and adapter are under review. No full real-model ablations are claimed. Extraction precision, output matching, actual cross-profile canaries, runtime isolation, raw provider outputs, and provenance need verification before accepting results.

## Open gates

T01-T11 have partial implementations and focused checks; no task is accepted solely from this ledger. Claude and Codex native adapter implementation is active. Hermes, complete CLI/MCP/UI integration, backup/reindex identity, installed-host continuity, full real-model ablations, 10,000-fact latency, and final packaging/docs remain open. The two-week pilot is not run.

No release, live host configuration change, or live database migration has occurred. Use `.venv/Scripts/python.exe` in this worktree; the global editable installation points elsewhere.

## Subsequent integration checkpoint

- CLI subprocess lifecycle and real MCP stdio lifecycle now use one authenticated owning service. Remember, processing, evidence inspection, stable-ID correction, and forget pass in isolated stores.
- Host-independent Codex/Claude/Hermes continuity passes across service restart with a foreign-project canary and replay suppression. This is adapter/service evidence, not a native conversation.
- Actual WSL -> Windows bridge transport and the installed Hermes 0.21.3 MemoryProvider API passed isolated checks. See `hermes-bridge-smoke.md`. Full native Hermes conversation remains NOT RUN.
- Backup/restore and offline `reindex` now have consistent snapshots, current-tombstone merge, identity binding, resumable generations, exclusive runtime ownership, and pre-activation backup. Changing an embedding provider no longer retains an unrelated default model or silently reuses the old dimension.
- Schema migration takes a private pre-migration SQLite snapshot. SQLite uses FULL synchronization for acknowledged writes. Registry, configuration, descriptor, outbox, and backup writes use private permissions.
- Forget suppresses an extraction already in flight when the forget commits. A failed vector write rolls back its fact row and leaves the connection usable.
- Graph derivation metadata includes algorithm/settings/embedding identity. Incompatible derived edges rebuild in resumable batches; dedicated tests cover a persisted neighbor outside the startup page and semantic candidate pool. Float32 cosine roundoff is clamped to [-1, 1].
- UI operations now use the service, check CSRF and loopback Host headers, escape stored content, and support versioned correction/forget plus project enable/pause. Readiness requires extraction of a fresh input-specific fact; a configured provider class is not readiness evidence. Browser validation is in progress.
- Seven real-model development smoke artifacts preserve failures: baseline used real embeddings without cognition; full and four ablations encountered model responses missing the required proposal operation. A strict structured-output repair and development rerun are in progress. No held-out result is claimed.
- Last broad snapshot: 466 passed, 2 failed. Both failures were Windows HTTP connection resets during early rejection with an unread request body. The bounded response/drain fix passed all 12 focused service tests. A new full-suite run is in progress.
- Python 3.10.18 compatibility environment was installed separately under `.venv/compat-py310`; the main environment remains Python 3.13.7. Type, packaging, 10,000-fact latency, full ablations, and installed-host conversation gates remain open.

## Runtime and distribution checkpoint

- Broad Python 3.13 snapshot: 488 passed in 247.01 seconds. Later shutdown, CLI status, and documentation changes passed focused checks; the final frozen-source suite remains pending.
- Python 3.10 compatibility: 34 selected service, graph, backup, persistence, and MCP tests passed in 75.10 seconds. Default mypy passed all 59 source files after the Hermes and CLI typing fixes.
- Authenticated `mnemos service-stop` passed a separate-process shutdown and ownership-lock release test. CLI degraded or queued results now return failure exit status. Three managed CLI tests passed in 15.73 seconds.
- UI browser validation passed enable/pause, HTML escaping, correction/history, maintenance, and forget. Follow-up screenshots confirmed truthful job counts and long-path wrapping. UI previews disclose the 500-fact limit.
- A wheel and source distribution built successfully; Twine metadata checks passed. A clean wheel installed with newly resolved dependencies was exercised from outside the source checkout: packaged UI assets and Hermes entry point existed; CLI service capture, processing, restart, stable fact identity, and forget passed. Dependency snapshot: MCP 1.30.0, sqlite-vec 0.1.9, NumPy 2.5.3. This is an intermediate build, not a release.
- Native Claude Code 2.1.191 hooks executed in a disposable project. The first default-model attempt failed on a host version requirement. An explicit model retry completed and recalled a seeded fact. Extraction, the complete lifecycle matrix, and clean host-hook isolation are separate open gates; see native-host-smoke.md.
- The first 10,000-fact fixture contained a provider mismatch: stored vectors came from Ollama but engine queries still used SimpleEmbeddingProvider. Its recall result is invalid for the intended Ollama gate and is being preserved with an erratum. Corrected performance measurement and diagnosis are in progress.
- Full held-out baseline and five-module ablations are still NOT RUN. Development artifacts retain malformed provider responses and failed jobs; strict-schema and scoring review continue before freezing the full run.
- Operational docs now distinguish the managed service from the embedded API, registry identities from folder names, processed evidence retention from accepted facts, and tested adapters from observed native hosts. The old embedded recall-gate preset is no longer the recommended service setup.

## Scale and native-host follow-up

- The subsequent broad run reached 494 passed and one documentation failure.
  The Codex guide was missing its AGENTS.md reference; that was restored and
  the 12 focused documentation/MCP checks passed. Further changes below still
  require the final broad rerun.
- Thirteen service tests passed, including duplicate-launcher exit without
  stopping the existing owner. Wheel packaging now includes the Claude agent
  templates, with an installed-package fallback path; a fresh wheel check is
  pending.
- Native Codex CLI 0.146.0 completed two disposable model turns, but no Mnemos
  hook callback, checkpoint, or seeded recall was observed. Configuration
  parsing alone does not establish capture. The native evidence record keeps
  this result separate from passing adapter contracts.
- Native Hermes 0.21.3 loaded its installed provider API and used the explicit
  WSL -> Windows bridge. The actual agent turn remains unproven: the small
  model failed context-size validation, gemma3:4b lacked tool support, and
  qwen3.5:9b timed out after 240 seconds. No live Hermes configuration changed.
- The corrected 10,000-fact performance runs exposed expensive JSON parsing
  and vec0 point lookups. Schema 7 adds a synchronized regular float32 vector
  table for exact scoped cosine search. Portable JSON vectors remain the
  authority; fallback writes, corrections, deletion, rebuild, and reindex
  maintain the binary table. Thirty-two focused storage/lifecycle/backup
  checks passed before the subsequent batched-access change.
- One measured post-cache run reached recall p95 735.094 ms (FAIL against
  500 ms) and durable capture p95 77.316 ms (PASS against 100 ms), with WAL and
  synchronous FULL. The fixture uses real query embeddings and synthetic
  load vectors, so it establishes no semantic quality result. Batched access
  metadata and content-only FTS updates are undergoing a separate rerun.
- Evaluation review found that the frozen v2 lifecycle scenarios mostly have
  one fact. Those rows cannot establish associative graph utility or affective
  ranking improvements. A separate competing-candidate fixture and controlled
  diagnostics are being prepared; the v2 labels are unchanged. Full held-out
  results remain NOT RUN.

## Integration freeze checkpoint

- Broad Python 3.13 suite: 508 passed, 2 skipped in 251.56 seconds. The two
  installed Hermes/WSL checks are opt-in for portable CI; both passed when
  explicitly enabled in a separate 29-test bridge/settings/UI run.
- Subsequent artifact regression: 29 benchmark tests passed. Raw JSONL names
  now include the report stem, preventing one evaluation profile from
  overwriting another profile's preserved output and invalidating its digest.
- Black passed across 112 Python files. Python 3.10 strict mypy passed across
  60 source files. The 50-session supplemental fixture passes scope preflight.
- The final scale run measured recall p95 506.111 ms, still FAIL against
  500 ms; durable capture p95 68.228 ms passes 100 ms. Exact scoped SQL p95
  was 117.791 ms. The 10,000-fact database occupied 324,862,864 bytes.
- The UI now persists the selected profile without copying project provider
  overrides into global settings. Generated host configuration carries the
  profile and registry. Claude Code imports and exports project `.mcp.json`;
  Claude Desktop discovery retains its distinct label. Claude automatic
  capture still requires the plugin; MCP setup alone does not install hooks.
- Source and wheel builds and Twine metadata checks passed. The source
  archive excludes temporary browser logs, fixture registry, and screenshots.
  A clean installed wheel, executed outside the checkout, passed service/CLI
  remember, processing, evidence, restart, stable-ID correction, and forget.
  Packaged UI, helper agent templates, and explicit host profile export passed.
  This lifecycle fixture uses unavailable cognition and simple embeddings;
  it validates distribution behavior, not real-model quality.
- Full real-model execution is the next step. Estimated serial duration is
  about 42 hours for 4,200 lifecycle observations and another 35 to 70 hours
  for 600 competing-candidate observations. Checkpoints bind dataset, model,
  options, and source hashes. No release or live migration is involved.

## Evaluation launch provenance correction

The first frozen launch from commit `9836bf2` was stopped after five baseline
observations. The source hasher excluded absolute `.venv` ancestors, so a
checkout inside `.venv/eval-frozen` produced an empty file list and the hash
of empty input. Those observations are preserved under `real-model/` as
invalid provenance and will not be counted toward the accepted evaluation.
The source exclusion now applies only to paths inside the checkout and rejects
an empty runnable surface. The replacement run uses a new frozen commit and
fresh output and checkpoint directories; dataset labels and quality gates
are unchanged.

The replacement queue is running from frozen commit `5e852e7`. Source
preflight and the live profile manifest agree on 118 source/configuration
files and hash `dd0631c56f5270679a84eee0d3f33c2c6f9c86f6cd7dcd4e4dcc98fde196d949`.
Eight baseline observations were durably checkpointed at the first progress
check; no quality acceptance result is claimed. The supplement is queued
after the lifecycle matrix. The provenance regression suite passed 31 tests.
The rebuilt frozen wheel is byte-identical to the installed wheel already
tested outside the checkout; source and wheel metadata checks passed. See
[evaluation-running.md](evaluation-running.md) and [frozen-build.json](frozen-build.json).

## Recovery and diagnostics follow-up

While the frozen model run continued, a completion audit found that manual
retry existed only in storage. The owning service now exposes scoped job
listing and exhausted-job retry. CLI `jobs`, `job-status`, and `retry-job`
and the UI recovery control use that service. Job counts now use the same
profile, project, and workspace boundary as the listing and retry operations.

Recovery tests first failed on the missing service operation. Focused checks
then passed across exhausted-write recovery, duplicate retry rejection,
foreign profile/project/workspace exclusion before the result limit, CLI
subprocess routing, and CSRF-protected UI retry. A real browser requeued the
exhausted fixture and maintenance created exactly one additional fact.
The UI now includes FastEmbed and removes stale managed-service claims for
the embedded recall preset. See [ui-smoke.md](ui-smoke.md).

Claude and Codex hook entrypoints now report sanitized failure categories to
stderr while keeping nonblocking exit behavior and the stdout host protocol.
The legacy CLI hook wrapper uses the same diagnostic. Secret-bearing
exception tests verify that messages and conversation content are excluded.

These follow-ups are later source changes. The ongoing real-model run remains
bound to its original frozen `5e852e7` checkout; no experimental source,
labels, thresholds, or checkpoints were changed. The later changes concern
recovery entrypoints, scoped operational counts, diagnostics, and UI behavior.

## Recovery verification checkpoint

Commit `ab3026a` passed the full Python 3.13 suite: 521 passed, 2 optional
checks skipped in 273.10 seconds. Black passed 115 Python files, strict mypy
passed 61 source files under Python 3.10, and the UI JavaScript syntax check
passed. A detached checkout built the wheel and source archive; both passed
Twine, and the source archive excludes generated browser and fixture files.
The installed wheel passed its lifecycle and exhausted-job recovery smoke
from `D:/WindowsTemp`, outside the source checkout. Exact package hashes and
the fixture scope are recorded in [recovery-build.json](recovery-build.json).

The supplemental v1 queue was withdrawn before execution because its direct
candidate pool already contained the intended graph target and it lacked
independent affective target labels. The coordinator was stopped intentionally;
the lifecycle worker continues independently from the unchanged frozen source.
A replacement mechanism protocol is being implemented. Neither completed unit
tests nor an evaluation observation's execution status establish model quality.

## Source availability, scope discovery, and local review

Source inspection now distinguishes a retained local original, an unavailable
original, and a fact without source evidence. Retention may remove the
original episode while preserving its excerpt and identifier. Restart and
retention regressions passed, and the browser rendered this distinction.

Graph refresh previously failed to discover permitted project/workspace
neighbors even though the connectivity rule allowed them. Discovery now uses
the registered workspace membership while preserving profile isolation. The
persisted edge-builder version is 3 so derived edges rebuild. Focused graph,
scope, retrieval, and restart tests passed. This changes the current product's
graph construction; the ongoing lifecycle run remains bound to version 2 in
`5e852e7`. Any new graph mechanism result must identify the later source.

The service, CLI, and UI now expose a read-only review of unassigned legacy
facts. Offline duplicate repair preserves unassigned or invalid scope metadata
and uses a separate canonical identity instead of silently assigning global
scope. No live legacy data was assigned or migrated.

Unresolved Claude Code, Codex, and Hermes routing now leaves a bounded local
diagnostic containing fixed categories and hashed identity hints. Conversation
content and raw paths are excluded. Profile-filtered review and dismissal work
without a running memory service; unconfigured-profile diagnostics require an
explicit selection. This queue records routing failures, not replayable
conversation content. Adapter, privacy, scope, CLI, and UI checks passed.

The replacement graph/affective harness is under review. Its authored labels
are frozen before embedding, and no replacement mechanism model calls have
been made. Interruption, resume, and aggregate reporting checks remain part of
the launch gate. See the [protocol](../../plans/2026-09-16-module-evaluation-protocol.md)
and [browser evidence](ui-smoke.md).

The follow-up broad run recorded 537 passes, 2 optional skips, and one
restart failure in 813.55 seconds. Shutdown joined the owner thread using
the startup timeout and then released ownership even if SQLite was still
closing. An immediate reopen could therefore fail with `database is locked`.
Shutdown now has its own 30-second deadline, preserves the ownership lock
and thread handle on timeout, and permits a later stop to finish cleanup
without interrupting an asynchronous close. Service and cross-host checks
passed after the fix, including a deliberately delayed close and repeated
stop attempts. Broad verification of the final source remains pending.

## Evaluator repair and mechanism development

The lifecycle v2 run was subsequently stopped after 600 baseline and 12 full
observations. Every baseline correction and forget attempt failed evaluator
target resolution before the service action. These files remain diagnostic
evidence; the earlier entries describing that process as ongoing are historical.
The v3 amendment preserves all 800 source events and adds explicit target
metadata. It is a repaired rerun of observed labels, not a fresh unseen dataset.

The replacement mechanism harness is committed at `e9d8ca2`. An offline
development fixture completed three persistent SQLite graph repetitions and
six controlled affective observations using synthetic embeddings. Graph
correctness passed; no model calls occurred and held-out coverage was not met.
The real-embedding mechanism development run now uses a detached checkout of
that commit. Its current status and artifact paths are in
[evaluation-running.md](evaluation-running.md).

The product wheel built from `4f0ec79` passed an installed-package smoke from
`D:/WindowsTemp`, covering evidence, restart, correction, forget, exhausted-job
retry, legacy review, routing diagnostic dismissal, and packaged assets. The
smoke used unavailable cognition and synthetic embeddings. A fixture type
narrowing followed in `8fb2972`; final source verification remains separate
from model-quality acceptance.

The first completed real-embedding graph development repetition exposed
insertion-order dependence: reversing insertion changed persisted edges and
ranked results, although the independent activation-energy calculation matched.
The development run was stopped with its databases and raw observations
preserved. This is a production graph construction investigation, and it blocks
the held-out mechanism run. The focused evaluator and graph suite passed
38 tests in 92.18 seconds; those tests did not establish this real-embedding
insertion-order property.

## Structured extraction development repair

The real lifecycle target preflight at `634b460` passed baseline correction and
both forget/replay cases, but full correction had no accepted target because
automatic extraction generated incomplete objects until its 512-token budget
was exhausted. Its source event and pending job remained durable. The schema's
partial `oneOf` alternatives exposed only operation-specific fields to the
constrained decoder. Each alternative now contains the complete object schema,
including content, evidence, and operation-specific targets and versions.

Commit `b5a2834` passed 35 focused extraction and evaluator tests and strict
contract typing. A fresh detached real Ollama run then passed all four
development cases without changing prompts, scenario labels, model pins, or
token budget. It verified corrected identity and version after restart in both
profiles, and suppression of forgotten facts after replay and restart. The
failed and passing raw outputs remain in `lifecycle-target-development-v3/`
and `lifecycle-target-development-v4/`. This clears the lifecycle dispatch
preflight; it does not establish extraction precision or held-out quality.

## Deterministic graph repair and recovery

Graph builder v4 persists directed preferences separately from reciprocal
traversable links. Source markers include completed empty neighbor sets. The
builder uses all compatible authoritative records, so insertion order, another
project's refresh, and partial startup hydration do not change the neighbor
rule. Storage commits preferences, projected edges, and markers atomically.
Correction and deletion invalidate affected sources; incomplete state remains
visible and disables graph scoring until bounded maintenance completes it.

Review caught and repaired missing empty-source state, promotion of uncomputed
sources during bounded recovery, incomplete in-memory preferences after a
mid-rebuild restart, and the standalone APIs' remaining greedy construction.
The standalone builder now follows the same mutual rule, including retained
custom source thresholds and candidate boundaries. Tests include the preserved
seven-fact real-vector development counterexample and a separate degree-one
counterexample checked against an independent nearest-neighbor calculation.

The service graph suite passed 19 focused regressions, engine/evaluator tests
passed 66, and subsequent transaction/version-marker and standalone checks
passed. Final frozen broad checks remain pending. A fresh browser fixture
completed its pending scoped and legacy source links in one maintenance run
and displayed ready with no document overflow.

The [v2 mechanism protocol](../../plans/2026-09-17-module-evaluation-protocol-v2.md)
records the graph construction amendment before any held-out mechanism case.
It preserves the original dataset, labels, budgets, thresholds, and coverage
minimums. Structural paths are counted without conditioning on successful
activation, so low-energy targets remain negative outcomes. No held-out graph
or affective utility claim follows from these implementation checks.

## Frozen graph v4 package verification

The complete suite at detached `b7a9d9d` passed with 624 tests and 2 explicit
WSL integration skips in 508.75 seconds. Black, strict mypy, JavaScript syntax,
wheel/source build, and Twine all passed. The installed wheel then passed
explicit-write, evidence, review, retry, restart, correction, forgetting, and
graph-readiness checks from outside the checkout. It also contained the UI and
host helper assets. This smoke used simple embeddings and unavailable
cognition. The [verification record](graph-v4-verification.md) preserves logs,
skip reasons, exact commands, and package hashes. Real-model quality remains
under evaluation.

## Fresh real-model queue at b7a9d9d

The repaired development mechanism run completed 15 graph observations and
30 affective condition observations. Recorded graph correctness controls
passed, but no graph case had an eligible target path. Classifier fidelity
passed 15 of 30 observations. These limitations are preserved in the
[development report](mechanism-development-v2.md).

The fresh queue started on September 17 at 05:03:42 UTC. All four integrated
real Ollama lifecycle preflight cases passed, including correction identity,
version, restart, and forget/replay suppression in baseline and full profiles.
Attempt `9fd1dc9612bd4f2bbbf24f9c6d97de17` is preserved under
`real-model-v3/lifecycle-preflight/`.

The same frozen source's 10,000-fact performance run passed capture at
57.410 ms p95 and failed recall at 1,314.583 ms p95. Two actual access-metadata
updates account for most of the slow queries' elapsed time; the underlying
stall cause is not established. The held-out mechanism run began at 05:05:27
UTC, followed in the queue by the complete seven-profile lifecycle matrix.
No passing quality claim or biomimetic-default promotion follows from these
preflights.

## Native capture repair and verification

The isolated Claude turn exposed a missing-transcript failure before the
assistant-message fallback. Commit `d17dd24` repairs that path, reports
unacknowledged capture and unavailable transcript coverage, and validates
checkpoint corruption separately. All 27 focused adapter tests pass. The
installed package then captured both roles of a native Claude conversation,
supplied the seeded SQLite memory, and retained both events after restart.

The first full suite on this checkpoint recorded 638 passes, 1 failure, and
2 skips. The service-start fixture did not become ready within 15 seconds;
the unchanged isolated retry passed in 4.68 seconds. The failure remains
preserved. A second full verification recorded 639 passes and 2 explicit
skips in 959.85 seconds; Black, strict mypy across 69 source files, JavaScript,
build, and Twine checks passed. The [verification record](native-host-repair-verification.md)
does not claim the intermittent startup issue is fixed.

Native Codex dispatched a Windows marker through `hooks.json`, and a separate
installed-wheel check verified distinct prompt captures and seeded recall
through direct Python and the tagged Windows shell command, including restart.
Native Mnemos conversation capture in Codex is still unproven. A real-service
Claude probe also reproduced rejected cross-project capture when session and
event IDs collide. Its repair design preserves old forgotten-source tombstones
and does not authorize migration of live data.

Indexed host and graph verification artifacts now preserve their original
line endings in Git. This fixes a Windows checkout normalization mismatch
between captured byte hashes and stored blobs; it does not change test results.

Native Codex capture v5 partially dispatched through the installed d17 wheel:
UserPromptSubmit and Stop completed, SessionStart timed out, both conversation
roles survived restart, and the prompt hook returned the SQLite seed. Codex
answered PostgreSQL, so seeded recall failed and the overall native acceptance
status remains `not_observed`. The report, exact runner, reviewed host output,
and cleanup checks are preserved under
`native-codex-capture-v5/`; native Codex support remains pending.

## Native Codex capture v6

The second repaired native run completed in 24.858 seconds with the installed
d17 wheel and a disposable service. `SessionStart`, `UserPromptSubmit`, and
`Stop` each invoked the wrapper and returned 0. The prompt hook returned the
seeded SQLite context. User and assistant capture rows survived restart, and
the host answered `The project uses SQLite.` The v6 report records those
capture and seeded-recall criteria as observed.

The run used the manual PowerShell wrapper from the probe rather than the later
production renderer package, so it does not validate that renderer in a native
Codex turn.

The fixture used simple embeddings and unavailable cognition. The host did not
report a model name, and the answer does not establish causal memory use,
model quality, or performance. The warm direct adapter preflight was excluded
from native counts. The exact runner, reviewed synthetic stdout observations,
and cleanup evidence are preserved under
`native-codex-capture-v6/`; the earlier v5 failure remains open alongside this
new partial success.


Source `20f271d` binds Claude event IDs and checkpoints to their resolved scope.
The service derives and checks old source suppressions before accepting a new
identity, retains exact legacy deduplication, and rejects ambiguous redaction.
A version field prevents older services from silently accepting v2 identities.
The 34 adapter/helper and 20 service identity tests passed. Full verification
started against a frozen checkout at 06:35 UTC on September 17. Native d17
results remain separate evidence. See [scope repair](claude-scope-repair.md) for
legacy checkpoint recovery and incomplete transcript coverage.

## Scoped capture verification

Frozen source `20f271d` passed 669 tests with two explicit WSL skips, followed
by Black, strict mypy, JavaScript syntax, build, and Twine. The installed wheel
also passed two-project Claude identity separation, independent checkpoints,
restart, legacy forget/replay suppression, and offline v2 capture queuing.
These are service and adapter checks using simple embeddings and unavailable
cognition. Native host and real-model evidence remain separate.

An unintended environment synchronization occurred during the first full
attempt. The [incident record](environment-sync-incident/README.md) preserves
the audit and its limits. A second full run began at 07:02:43 UTC against the
same frozen source with no further shared-environment synchronization. See
the [verification record](claude-scope-verification.md) for package hashes and
the first attempt's evidence. The real-model queue continues with the incident
disclosed; its recorded source and runtime binding still match.

## Generated native Codex configuration

The installed `20f271d` wheel's generated TOML hooks passed a native CLI turn:
the host returned a random memory canary absent from the prompt, made no tool
calls, and retained both captured roles after service restart. This fixture
used constant unit vectors and unavailable cognition. It verifies native
delivery and persistence without claiming semantic quality or full lifecycle
coverage. The [v8 record](native-codex-generated-v8/README.md) preserves exact
provenance and the preceding SimpleEmbeddingProvider preflight failure.

The stable full repeat on `20f271d` completed at 07:19:50 UTC. All 669 tests
passed, with two optional WSL skips, in 972.25 seconds. Black, strict mypy,
JavaScript syntax, build, and Twine also passed. No further shared-environment
synchronization occurred during the repeat. Its wheel and source archive are
byte-identical to the first build; installed-wheel evidence uses those same
wheel bytes. Both attempts and the original environment caveat remain in the
[verification record](claude-scope-verification.md).

The current wheel also passed a native Claude 2.1.191 canary turn at 07:21 UTC.
Both roles used scoped v2 source identities and survived restart. The run used
constant fixture vectors and unavailable cognition, and cost $0.009372.
UserPromptSubmit and Stop reported degraded adapter status; the test used no
session persistence and does not establish complete transcript coverage or
native correction/forget. See the [native record](native-claude-scope-v4/README.md).

The persistent-transcript v5 attempt timed out before native Stop. It preserved
a user capture but no checkpoint or assistant capture. Its partial assistant
stream contained the canary; final cost was unavailable. The negative result
is retained in [the v5 record](native-claude-persistent-v5/README.md).

A separate local SQLite probe then confirmed [Stop replay duplication](claude-stop-replay/README.md):
after reading the full transcript and restarting, replaying the same Stop
created a second assistant event under the fallback source reference. The
earlier broad test pass did not cover this sequence. A scoped adapter repair
and new regressions are in progress; native persistent capture remains open.

## Checkpoint timeout diagnostic and benchmark instrumentation review

The installed source20f direct adapter diagnostic completed UserPromptSubmit
in 1,291.0 ms and Stop in 1,050.3 ms, preserving the 10-second deadline.
Both checkpoint writes completed; fsync took 160.655 and 180.207 ms.
The original native persistent timeout did not reproduce and remains
unexplained. No native host or model was invoked. The report, exact runner,
and operation traces are preserved in
[the checkpoint diagnostic](claude-checkpoint-diagnostic/README.md).

Source review also found extra metadata writes and diagnostic reads between
timed performance queries. The recorded 1,314.583 ms recall failure remains
preserved, with this workload confound now explicit in
[the performance record](performance-10k.md). A benchmark-only correction
is being verified; no new latency result is claimed.

## Claude replay repair and v2 benchmark committed

The Claude replay repair is committed as `8d85894`; benchmark instrumentation
v2 is committed separately as `554a251`. Fifty-four focused adapter/identity
tests and four benchmark tests pass. Source type checking and formatting pass.
The earlier broad benchmark-file test attempt was interrupted after nine
reported tests stalled; it is not a broad pass. Only that owned pytest
process was interrupted, and the frozen evaluation continued.

The pinned real SQLite probe passes all seven positive assertions: two
events remain two after restart and replay, with unchanged IDs and one
native assistant record. Missing prompt identity remains degraded.
Unknown fallback/native associations preserve native records with explicit
ambiguity; full native deduplication is still open.

Full verification is running against an immutable `554a251` checkout.
The installed package and native host checks for this new source remain
pending; prior source20f checks cannot certify the repair.

## Replay repair full suite and installed wheel passed

Frozen `554a251` completed verification at 08:39:22 UTC on September 17:
691 tests passed, two optional WSL tests skipped, and formatting, typing,
JavaScript, build, and package checks passed. A fresh disposable installation
of that wheel passed all seven SQLite replay assertions. The exact logs,
commands, package hashes, and runners are preserved in
[the verification record](claude-replay-verification/README.md).

Native host acceptance remains open. The Claude
[identity contract](claude-host-identity-contract.md) explains why older hooks
cannot always associate fallback text with a later transcript UUID. A separate
Hermes discovery probe found a registry-path availability defect; its repair
passed focused tests and real disposable WSL plugin loading. Full verification
of that newer source is pending. The held-out model queue continues on its
original frozen source, and the corrected latency benchmark has not run.
## Hermes registration repaired; dispatcher startup blocks broad verification

Commit `2360891` fixes registration's handling of the configured scope registry.
Seventeen focused tests passed. The same isolated WSL loader probe rejects the
old frozen source and accepts the repaired frozen source with exact import
and adapter-hash checks. The [discovery record](hermes-registry-discovery/README.md)
preserves both outcomes and the earlier helper failure.

The full frozen-source run completed with 699 passes, two optional WSL skips,
and one failure. The service child exited when dispatcher initialization
exceeded its existing five-second deadline. This is not the earlier test-level
15-second timeout. The [failed run](hermes-registry-verification/README.md)
and its complete-schema fixture are preserved. No package was built by this
attempt; installed-wheel and native lifecycle follow-ups remain pending while
startup is investigated. The original real-model queue remains active.

## Atomic schema startup repair

Commit `9e25339` wraps schema creation and migration in one explicit SQLite
transaction and rolls back initialization failures, including interruption.
The startup deadline, WAL mode, and FULL synchronization are unchanged.

The [frozen-source comparison](service-startup-diagnostic/README.md) used the
same runner bytes for one fresh start per revision. Source236 failed with the
dispatcher timeout; source9e25339 started in 2,453.355 ms and read back WAL/FULL.
Schema initialization measured 4,774.106 ms before the repair and 776.631 ms
after it. These are local diagnostic observations, not a general latency result.

All three new regressions fail against frozen source236: interrupted creation
leaves partial schema objects, and failed migration changes an existing trigger.
They pass on the repaired source, as does the 33-test focused storage, migration,
and backup suite. The focused results were terminal output; the old-source
failure report and runner are preserved. Full frozen-source verification is
running before installation and native host follow-ups.

## Source9e package checks passed; captured-copy removal remains open

The atomic schema repair passed all repository checks: 703 tests, two optional
WSL skips, Black, strict mypy, JavaScript syntax, build, and Twine. The exact
wheel passed installed Claude replay, actual Hermes entry-point discovery, and
an isolated offline Codex API preflight. The [verification record](atomic-schema-verification/README.md)
preserves commands, hashes, logs, and the initial preflight digest typo. These
checks do not establish a new native model conversation.

The corrected [overlap probe](capture-forget-overlap/v2/README.md) confirmed that
record-only forget leaves an independently captured copy of the assertion
retrievable after restart when extraction is disabled. The first probe used
unrelated inputs and does not support that conclusion. Both are preserved.
The [related-forget plan](../../plans/2026-09-17-related-forget.md) adds a scoped
preview and atomic application of explicitly selected captured copies. Native
lifecycle acceptance remains open while this path is implemented and tested.

## Held-out mechanism phase completed on sourceb7

All 50 graph cases and 50 affective pairs completed three repetitions. The
[preserved results](real-model-v3/mechanism-heldout/REVIEW.md) pass graph
correctness controls but fail the graph eligibility gate: zero of 50 cases
had a valid measured path, against the required 40. Graph recall utility is
therefore unmeasured. Affective ranking with synthetic gold states records
top-1 selection of 0.33 versus 0.00 for semantic ranking; the template-clustered
paired interval is [0.18, 0.47]. Classifier fidelity passes 154/300 observations
and is a separate measurement. None of these establishes human acceptance.

The completed phase's eight original artifact hashes match the queue record.
Its source is frozen `b7a9d9d4`, not the later startup or related-forget changes.
The lifecycle phase is running separately; its mutable outputs are not included
in this checkpoint.

## Reviewed captured-copy removal implemented

Commit `c805cc2` adds scoped related-forget preview and atomic application
through the service, MCP, and memory UI. Additional captured records require
selection. The operation validates retained and expired source dependencies
through both evidence and job references, rejects stale or incomplete plans,
redacts foreign details, scrubs selected job payloads, and invalidates leases.

The [preserved development checks](related-forget-verification/README.md)
include 83 affected tests and 15 expanded safety tests. Browser v4 passed the
actual HTTP/control-plane/service/SQLite path, then reopened the database and
replayed the capture without resurrecting either selected fact. An initial
expired-source regression failure and a browser-driver timing failure remain
preserved. Every product file in those browser and focused reports matches the
committed implementation after line-ending normalization.

Full tests and package checks are running against a clean frozen `c805cc2`
checkout. Installed-wheel and native lifecycle acceptance remain pending. The
native Codex runner has passed offline parser and protocol checks, which do not
constitute a native conversation. The real-model lifecycle queue remains on
its original frozen `b7a9d9d4` source.


## Sourcec805 package and installed transport checks passed

The clean frozen source passed 738 tests with two optional skips, Black,
strict mypy, JavaScript, build, and Twine. Its exact wheel passed installed
Claude replay, actual Hermes entry-point discovery, and generated Codex MCP
handshake and scoped retrieval. An instrumented Claude adapter run captured
both roles, persisted its checkpoint, and replayed after reopen without
duplicate events, within each fixed ten-second deadline. SessionStart and
native Claude latency are outside that diagnostic gate.

The first native Codex attempt exposed an incomplete test installation:
the base wheel had no optional MCP SDK, and the API-only preflight missed
that condition. A fresh MCP-extra installation and actual transport preflight
passed. The subsequent native turn discovered all required tools, but its
store, consolidate, and retrieve calls were cancelled by the client before
successful results. Both failures remain preserved; no completed native
lifecycle is claimed. See the [checkpoint evidence](related-forget-verification/README.md).

The real-model lifecycle queue remains on frozen b7a9d9d4 and is still running.
No live host configuration, live memory data, or release was changed.

## Native diagnostics and completed baseline

The sourcec805 persistent Claude test now completes a native turn and preserves
the actual transcript, checkpoint, both captured roles, and restart/replay.
It still fails acceptance because the prompt is captured twice: the native
hook omits its turn identifier, while the later transcript has a UUID and
`promptId`. The adapter refuses to associate them by text alone and reports
degraded coverage. That failure remains explicit in the
[host evidence](related-forget-verification/README.md).

A matched Codex recorder test succeeds with accurate read-only annotations
and is cancelled before server dispatch without them. This explains a concrete
metadata gap to repair in the inspection tools. Mutating tool approval remains
a separate host policy; the recorder does not prove the Mnemos lifecycle.

The frozen baseline completed all 600 observations. Exact answerable recall
and critical retention are 40%; no-answer empty retrieval is 0%. The 8.3%
supported-text/evidence score is an exact-match proxy and does not establish
semantic factual precision. Its frozen note incorrectly says paraphrases are
unjudged, while the implementation counts unmatched text as false. The
[interim record](baseline-interim/README.md) preserves that distinction and the
failed gates. The full and ablation profiles are still running on unchanged
sourceb7; there is no full-pipeline or module-benefit conclusion yet.

## MCP operation metadata repaired

The MCP service tools now declare read-only, destructive, idempotent, and
external-provider effects. Six inspection handlers are read-only; retrieval
retains its write classification because it updates access metadata. Provider
operations and destructive maintenance keep conservative approval hints.

The affected MCP and related-forget transport checks passed 20 tests. The
expanded two-test annotation suite also verifies successful responses against
a populated SQLite store with an accepted fact, evidence, a completed job,
and a versioned correction. All six inspection calls preserve every logical
table, including after reopen. Black and compatible strict mypy passed.
These focused checks do not supersede the sourcec805 full-suite/package
checkpoint or establish native lifecycle acceptance for the new source.

The future benchmark report note now accurately describes extraction
precision as an exact-text, scope, and evidence proxy. Unmatched text,
including paraphrases, counts as false. This changes the explanation only;
the scorer, frozen sourceb7 run, thresholds, and prior artifacts are unchanged.

## Lifecycle run stopped on validation-trace serialization

The sourceb7 coordinator exited with code 1 at 12:21:39 UTC on September 17.
The baseline completed 600 observations; the last intact full-pipeline
checkpoint contains 36. During the next observation, a Pydantic validation
error left a `ValueError` object in its diagnostic context. JSON checkpoint
serialization then failed. The remaining profiles did not run.

The working benchmark now serializes Pydantic validation details through
their JSON representation, retaining the error type, location, message,
input, and context message. The original validation failure is still raised
and recorded. A regression reproduced the checkpoint crash before the fix;
all 36 affected benchmark tests pass afterward. No scorer or acceptance
threshold changed. The frozen source and failed run remain unchanged;
a resumed or replacement experiment must record the repaired code identity.

## Provisional Claude capture implemented

Missing-occurrence-ID Claude fallbacks now persist as deferred raw events
with capture-only jobs. Their completed result is `identity_pending`, with
no fact IDs. Both processing and SQLite commit guards reject attempts to
extract or commit facts from those events. Native transcript identities
remain independent, and legacy normal jobs retain replay compatibility.
The two affected transactions now roll back on interruption as well as
ordinary exceptions. Retention removes only completed deferred events;
their completed jobs prevent raw-event resurrection on replay.

The affected adapter, identity, service, and outbox suite passed 77 tests.
The expanded 13-test provisional suite also passes, including the real
client outage/rejected-ACK/restart test, deferred retention, and injected
interruptions after event insertion and before job completion. Black and
strict mypy over the four changed production files pass. Tests remain
outside the repository's configured mypy scope; explicitly checking the
new capture test file reports untyped test helpers.

Native Claude acceptance and full verification of this new source are
pending. The original v6 duplicate-raw-event failure remains preserved.
