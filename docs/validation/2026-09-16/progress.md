# Functional local memory implementation record

Worktree: `D:/Development/mnemos/mnemos/.worktrees/functional-local-memory`
Branch: `wt/functional-local-memory`
Base: `5fcc925d5a90a087e5715c2fe6de56fa23a7d3a6`

This is a chronological checkpoint log. Later entries supersede earlier test
counts and open implementation items. An implementation or focused check does
not close a native-host, quality, performance, or pilot gate.

## September 19: same-session rewind identity collision

The [installed Hermes callback diagnostic](../2026-09-19/hermes-rewind-diagnostic-v2/README.md)
reproduces a remaining identity failure on `ea88dd4`: four initial messages
remain four queued records after a same-session rewind and identical replacement
of the last turn, where six logical occurrences are expected. Exact duplicate
replay remains idempotent, changed assistant content adds one ID, and a new
session adds four IDs. No service or model ran. The first zero-capture fixture
incorrectly reported passed; its failure and limited first rerun are preserved.
The reviewed successor records identity acceptance as false. A repair needs
the regenerated suffix boundary or native occurrence identity; rekeying all
surviving messages would create false observations.

T19 documentation now distinguishes the latest `ea88dd4` package from earlier
host results and records this negative result in the writing handoff. The plan's
Run commands point to existing test filenames; acceptance criteria are unchanged.

## September 19: native discovery and replay identity

The [successor Hermes preflight](../2026-09-19/hermes-native-discovery-preflight-v1/README.md)
passes actual plugin entry-point discovery and six-tool initialization under
runner `f10bf6c`. It also verifies the shared disposable Hermes home, which the
API replay helper must supply to preserve original occurrence timestamps.
The preceding helper omitted that initialization argument; no model run had
begun. The real host phase completes in 2.434 seconds without model or service
requests. Forty focused tests and configured type checks pass. The verified
runtime wheel and live host configuration are unchanged.

## September 19: native Hermes lifecycle setup

The [native lifecycle preflight](../2026-09-19/hermes-native-lifecycle-preflight-v1/README.md)
passes against the verified `ea88dd4` wheel and isolated patched Hermes host.
Runner `773738f` and its prospective protocol are committed. Forty focused
tests pass, including actual service response schemas; formatting and configured
type checks pass. The native preflight resolves its disposable project and all
six tools in 2.502 seconds without model or service calls. Full model-driven
lifecycle acceptance remains open. The live supplemental queue still owns the
GPU; its length-noise arm had 97 recorded observations at this checkpoint.
The prior two arms finished with 589/600 and 592/600 complete observations,
respectively. Partial observations remain failures, and retrieval follows in
the same queue. No model run was restarted or overlapped.

## September 19: correction retry and writing packet refresh

The [correction retry checkpoint](../2026-09-19/correction-retry-checkpoint-v1/README.md)
at `ea88dd4` passes 1,201 source tests with four skips, all six source/package
checks, 513 isolated installed cases with two matching skips, and seven replay
assertions. A completed correction no longer rejects its own unchanged retry
as stale. It recovers the original durable job after restart, retention expiry,
or a later correction without changing the newer fact. Fresh edits retain
version checks; payload collisions, scope violations, unsafe writes, and
forgotten targets remain rejected. Two queued corrections against the same
version demonstrate that durable acknowledgement and completed processing
are separate: the losing job exposes its error and does not rewrite the fact.

The first installed-check invocation supplied the wrong form of report path
and stopped at preflight. Its record is preserved; the corrected invocation
passed against the unchanged verified wheel. No model evaluation was restarted.
The full encoding arm has 589 complete and 11 partial observations; the
no-surprisal arm was still running at the latest dated status observation.
The writer handoff now includes the correction retry example alongside the
Hermes API recovery and negative graph evidence, with final acceptance still
open.

## September 19: development graph budget comparison

The [degree diagnostic](../2026-09-19/graph-degree-development-v2/README.md)
uses frozen development vectors on source `5e4b057`, testing every combination
of five cases and degrees 4, 8, and 16. All 15 measurements complete and pass
structural controls. Degrees 4/8 yield zero eligible cases; degree 16 yields
one path above the unchanged activation threshold, but no target reaches the
top five at any degree. No production setting changed. The first attempt's
report-parser error remains preserved, and the successor reran all planned
combinations in fresh storage. Exact inputs and raw observations are archived.

## September 19: installed Hermes API recovery and graph diagnosis

The [API recovery observation](../2026-09-19/hermes-api-recovery-v1/README.md)
passes on the verified `5e4b057` wheel. Three fresh WSL MemoryManager processes
use the real Windows bridge and outbox. Two outage captures survive process
exit, the required checkpoint is durably backed, and an ordinary request
drains the queue after explicit service restoration. Two event/job identities
and their original timestamps persist without duplication. Exact forgetting
survives another restart and transcript replay, with both controls unchanged.
This is fixture-provider API evidence, not native model or automatic daemon
recovery acceptance. Live settings and source remain unchanged.

The [graph diagnosis](../2026-09-19/graph-coverage-diagnosis-v1/README.md)
reconciles all 150 original held-out observations without rerunning inference:
every target is disconnected from the actual selected seed even when traversal
limits are ignored. Threshold or depth changes alone cannot repair those
graphs. The failed coverage result remains intact; association formation needs
separate development work before another prospective utility experiment.

## September 19: Hermes lifecycle checkpoint and writing handoff refresh

The [successor checkpoint](../2026-09-19/hermes-lifecycle-checkpoints-v1/README.md)
at `5e4b057` passes 1,195 source tests, four skips, all six source/package
checks, 507 installed cases, two matching skips, and seven replay assertions.
Installed Hermes API v2 passes 16 tool calls through the actual Windows/WSL
bridge. Four normal transcript occurrences survive repeated callbacks and
provider recreation. Four separate provisional receipts cannot produce facts.
Providers are fixtures; this is separate from the earlier patched CLI v4
real-model single-turn result. Live host configuration remains unchanged.

The earlier `b7d7500` candidate's full and installed suites passed, but later
regressions exposed rejected Hermes provisional evidence and a missing
required request ID in write schemas. The archive preserves both failed
installed regressions and the successor checks. Identity hashes bind submitted
fields, not host authorship; authenticated adapters remain trusted evidence
sources, and later real transcripts can create new normal occurrences.

The Substack research handoff now includes these repairs, bounded results, and
open questions. The independent real-model encoding/retrieval queue remains
active with its original frozen `fc72cc2` source, dataset, and protocols.
Full native lifecycle, model utility, Desktop, and pilot acceptance stay open.

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

The [failed lifecycle archive](lifecycle-serialization-failure/README.md)
now preserves the completed baseline, the last full checkpoint, manifests,
queue state, logs, and a read-only diagnostic of the failed observation's
database. Of the 36 checkpointed full observations, six are complete and
30 are partial. The final failed observation's in-memory model trace was
not saved. It is not included in any completed result. A clean frozen
checkout of `7dd3071` is now running the full suite and package checks.

## Full tests passed; archive formatting repair

Source7dd passed 754 tests with two optional skips. Black then found the
preserved Hermes discovery `payload.py` in the evidence tree. Its bytes are
unchanged; its archive filename is now `payload.py.txt`, and its evidence
index is updated. The [first validation attempt](provisional-checks-first-attempt/README.md)
remains preserved. The later typing, JavaScript, build, and package gates
were not reached in that attempt and need a new frozen verification.

The [extraction diagnosis](extraction-evidence-id-diagnosis.md) verifies that
all 30 partial full-profile observations cited existing fact IDs as source
evidence. The runtime rejected them correctly. This is separate from the
checkpoint serializer failure. Further development must distinguish source
citations from target-fact IDs and test actual source support, without
weakening the evidence guard or reusing partial observations as successful
measurements.

## Provisional capture package checkpoint

Frozen source `a38ee70` passed 754 tests with two optional skips, Black,
strict compatible mypy, JavaScript syntax, build, and Twine. The new wheel
passed fresh MCP-enabled installation, source/asset parity, and capture
restart/replay checks. Installed Claude adapter instrumentation also passed
the existing 10-second per-command bound. It does not exercise native Claude
or SessionStart. [Reports and original runner bytes](provisional-package-verification/README.md)
bind these results to the exact source and wheel hashes.

Native probes are next. The extraction schema/prompt candidate is isolated
from this package checkpoint and is undergoing a small, newly defined
development comparison with Ollama. The stopped lifecycle queue and all
negative results remain preserved; no held-out acceptance threshold changed.

## Native provisional identity and MCP installation binding

Native Claude v8 against the installed a38 wheel completed its canary turn,
captured one normal transcript-associated event per role, and kept the extra
unidentified prompt fallback deferred and capture-only. Maintenance completed
that job without derived facts or evidence; restart/replay passed. The original
three-raw-events failure remains visible. This uses instrumented explicit hook
settings, not plugin discovery. [Native record](provisional-native-claude-v8/README.md).

Generated MCP settings used a bare console-script command even though hooks
used the current Python interpreter. The MCP command now uses the same Python
installation and replaces obsolete launcher arguments. Regression checks cover
Codex, Cursor, and fallback Claude, including paths with spaces, retained
unrelated settings, and preview-only behavior. The affected host/control/settings
suite passed 40 tests; Black and compatible production mypy passed. This repair
is newer than the verified a38 wheel and needs a new package checkpoint.

The a38 installed wheel completed a native Codex health call once the probe
pinned PATH to that installation and checked the resolved launcher before
starting Codex. The response reported one seeded SQLite fact. No approval or
hook-trust override was used; no hook callbacks were observed. The earlier
cancelled run selected a different launcher on inherited PATH. Two probe-only
errors were also retained (missing helper and Path.casefold); neither is a
product failure. [Native MCP record](codex-health-installation-binding/README.md).

Hook command review also found a preexisting POSIX quoting gap. Interpreter
paths now use shell-aware quoting on POSIX; Cmd fallback rejects percent/quote
paths that cannot be represented safely. PowerShell literal quoting is
unchanged. Host tests pass (20 passed, two POSIX subprocess cases skipped on
Windows), with Black and compatible production mypy passing. These skips are
not POSIX execution evidence.

## Request-bound extraction schema and remaining model errors

The extraction repair now constrains source citations, current fact targets,
and exact target/version pairs. Complete decision alternatives require at least
one fact for `facts`, or zero facts and a non-whitespace reason for
`no_durable_facts`. Runtime guards remain unchanged. Nineteen new contract
tests pass. A broader 75-test check found one stale assertion requiring four
operations even without target facts; the corrected assertion now requires
only add/conflict in that case and passes. Target-operation coverage remains
in the new tests.

The [original eight-call diagnostic](extraction-contract-development-v1/README.md)
and [four-call schema repair check](extraction-contract-development-v2/README.md)
remain separate. The repaired candidate produced four schema/runtime-valid
responses, but matched the predefined exact case proxy only once. It selected
add for support and correction, and declined a clear preference. Valid citation
IDs do not prove good memory decisions. A bounded comparison with the existing
qwen3.5:9b installation is next; the large held-out queue remains stopped.

## Verified extraction package and Qwen thinking diagnostic

Frozen source `64471f5` passed 776 tests with four platform-specific skips,
Black, strict compatible mypy, JavaScript, build, and Twine. Fresh installation
passed replay and 53-Python-file/three-asset parity. Native Codex health passed
using the generated exact Python interpreter command, with inherited PATH,
no approval override, and no hook-trust bypass. Cleanup removed the copied
authentication and runtime descriptor. This remains a health-dispatch check,
not complete host capture acceptance.

Qwen3.5:9b with the current provider defaults produced four empty final
responses while returning a separate thinking field; all four failed parsing.
All stopped before the token limit. Replaying the same four prompts/schemas
directly with only `think:false` added produced four valid final responses and
two exact proxy passes. It still omitted the preference. The correction case
contrasts a new requirement with an existing actual deployment, so its add
operation is not conclusive evidence of a semantic error. Its frozen proxy
remains failed; a future disjoint case must state supersession explicitly.
[Diagnostic artifacts](qwen-thinking-control/README.md).

The direct request repair is not yet integrated provider behavior. An isolated
candidate is adding an explicit optional Ollama thinking setting and a typed
empty-final-response failure. No reasoning text is promoted into facts.

## Ollama provider integration and preference omission

The candidate provider completed the same four development cases through
`SleepDaemon.propose_facts` with `think=False`. All four passed runtime
validation; three matched the frozen exact proxy. Each case made one observed
provider call. This is actual candidate provider execution, separate from the
earlier direct HTTP diagnostic. It remains a four-case development result,
not a held-out quality estimate.
[Source binding and raw responses](ollama-thinking-provider-integration/README.md).

The model again omitted the editor preference, explaining that it was not a
durable system-state fact. That exclusion conflicts with Mnemos's intended
preference extraction. The actual runtime prompt names durable facts but does
not explicitly include preferences. A prompt clarification now names stated
preferences and continuing requirements, permits new topics, and preserves
the distinction between requirements and completed actions. Source, role,
scope, evidence, and version guards remain unchanged. A separate disjoint
development check will test that clarification, including questions,
quotations, hypothetical preferences, and assistant suggestions.

## Verified thinking configuration and fresh evaluation queue

Frozen source `d94ded6` passed 811 tests with four platform-specific skips,
Black, compatible strict mypy, JavaScript syntax, build, and Twine. Its wheel
passed fresh MCP-enabled installation, source/asset parity, and instrumented
restart/replay checks. [Package checkpoint](thinking-package-verification/README.md).
This adds no new native-host conversation claim.

The disjoint eight-call preference probe produced eight runtime-valid
responses. Its frozen exact text-match proxy remains 4/8. Separate qualitative
review against the authored criteria accepts seven cases: both preference
cases, explicit correction, and all four abstentions. The undeployed Redis
requirement fails because the model returns the old SQLite statement as a
conflict with an unsupported current-source citation. All 49 archived Python
source files match the frozen product after newline normalization.
[Raw observations and qualitative review](preference-contract-development-v1/README.md).

The new `real-model-v4` queue started against this frozen source, Qwen3.5:9b
digest `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`,
Nomic embedding digest `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`,
and `think=false`. It runs development mechanisms, lifecycle preflight, the
corrected latency measurement, held-out mechanisms, and all seven lifecycle
profiles with three repetitions. Output and checkpoints are new; prior
failures remain intact. At this checkpoint the development phase is running,
so no completed held-out result or latency improvement is claimed.

## Conflict presentation and native lifecycle follow-up

Offline replay of the unsupported conflict proposal confirms that canonical
deduplication merges it into the existing SQLite fact, preserves its ID and
version 1, changes its assertion status to conflict, and attaches the Redis
source as a second evidence record. It creates no Redis fact. The replay uses
injected recorded output, not another model call, and verifies the frozen
source import and hashes. Its simple-embedding query returns no facts.
[Persistence diagnostic](conflict-proposal-persistence-v2/README.md).

A separate retrievable-conflict regression reproduced missing assertion status
after SQLite restart. Commit `6ba17f5` exposes `assertion_status` in retrieval,
labels unresolved conflicts in host context, and counts the label in the shared
context budget. Fifty-nine affected service, retrieval, cross-host, MCP, and
benchmark tests pass, along with Black and strict production mypy. Full package
verification for this newer presentation change remains pending; the active
model queue continues against unchanged frozen `d94ded6`.

The verified d94 wheel also passed a native Codex lifecycle preflight. The
actual native turn reached `mnemos_store`, then received a client cancellation
without a successful service result. Correction, forgetting, and native
restart were not reached. No approval or hook-trust override was added.
Private auth and runtime descriptor cleanup was verified afterward.
[Native attempt](native-codex-lifecycle-v4/README.md).

The d94 development mechanism and lifecycle preflight phases have now
completed. Graph execution and controls passed all 15 development observations,
but none had a structural seed-to-target path under the frozen vectors and
mutual top-4 graph. This is a negative coverage result, not demonstrated
associative retrieval benefit. Held-out mechanisms are still running.

The corrected 10,000-fact measurement still fails warm recall: 1,088.094 ms
p95 against 500 ms. Durable HTTP capture passes at 64.279 ms p95. Removing
duplicate diagnostic writes did not eliminate metadata-update stalls.
[Completed performance phase](performance-corrected-v4/README.md).

## Verified conflict-display checkpoint and embedding follow-up

Frozen `6ba17f5` now passes the full suite: 812 tests passed and four skipped.
Black, compatible strict mypy, JavaScript syntax, build, and Twine also passed.
This closes package verification for the presentation repair. It does not add
installed-wheel or native-host lifecycle evidence and does not include the
subsequent access-metadata candidate.
[Verification archive](conflict-display-package-verification/README.md).

Source review found that the Ollama embedding path does not distinguish query
and document task prefixes. Nomic's model card specifies those roles. The
current queue remains evidence for its unchanged, unprefixed configuration;
the quality effect of a role-aware configuration is not measured yet.
[Source finding and implementation sequence](embedding-task-contract.md).

## Access-metadata candidate

Commit `2494e31` moves access counts, last-retrieved timestamps, and
reconsolidation due times into an authoritative small SQLite table. Schema 8
seeds legacy values once, preserves the existing pre-migration backup, and
seeds new raw or API inserts through a trigger. Nullable side-table values
remain authoritative after restart. Reads, stats, lexical projections,
reconsolidation clearing, and deletion follow the new table; durable writes
retain WAL/FULL and savepoint rollback behavior.

The new SQL-trace regression fails against frozen d94 because retrieval
updates the wide chunk row. On the candidate, 57 affected storage, service,
and retrieval tests passed, along with Black and strict production mypy.
The migration fixture was then strengthened to remove the new insert trigger
when reconstructing schema 7 and passed its separate rerun. Review found no
confirmed correctness defect. A fresh `store()` replacement still uses the
access fields provided by its caller, matching the previous API behavior.

Full frozen-source verification is running. A separately bound write-latency
comparison is prepared but not executed. The last accepted recall measurement
is still the failing d94 result, and no live database has been migrated.

## Embedding task-role integration

The working runtime now distinguishes query, document, and symmetric embedding
roles. Prefix settings default to empty strings, preserving the existing index
identity. Nonempty settings add an explicit versioned task contract, and a
contract change requires reindex generation and activation. Surprisal and sleep
support comparisons use the symmetric role; stored facts use the document
role; retrieval uses the query role.

The provider foundation and initial integration checks passed 19 tests. A later
runtime integration set passed 79 tests, and backup/reindex, documentation, and
basic benchmark checks passed 30. Strict production mypy passed for 49 source
files. These overlapping test sets are not an updated full-suite total.
Benchmark role-aware caches, frozen vectors, traces, and launch options remain
under integration. The real-model queue still uses frozen d94, without prefixes.
No quality improvement or new full-package acceptance is claimed from these
deterministic checks.

## Verified access-metadata package checkpoint

Frozen source `2494e31` passed 824 tests with four skips in 1,551.77 seconds.
Black, compatible strict mypy, JavaScript syntax, build, and Twine passed.
The archive preserves the executed runner, logs, test XML, source revision,
and verified distribution hashes. Its ten indexed blobs were verified before
commit. These checks ran while the graph diagnostic and real-model queue were
active, so their duration is not a performance measurement.
[Verification archive](access-metadata-package-verification/README.md).

The checkpoint includes schema 8 and excludes the later embedding-role and
conflict-collision work. A fresh installed-wheel lifecycle and the separately
bound metadata-write timing comparison have not run for this revision.

## Embedding integration review completed

Benchmark preparation now keeps query and document vectors separate for the
same text. Persistent caches bind the task contract, retain dimensions without
loading the provider after restart, and reject dimension drift across roles.
Role separation also applies to providers with native role methods and empty
prefixes. Symmetric calls retain their legacy flat-cache representation.
Launcher options, manifests, trace records, and performance provenance carry
the new contract. The frozen real-model queue remains unchanged.

Review caught and repaired a cache role-type annotation and an older in-flight
forget test fixture that did not forward the new role keyword. The fixture
still pauses embedding to exercise the same forget race. Thirty-five combined
embedding, benchmark-role, lifecycle, and conflict-collision checks passed.
Full repository Black passed 164 files, and compatible strict mypy passed all
70 source files. Full package verification for this new source is next.

## Conflict canonical-collision safeguard

InMemory and SQLite canonical upserts now reject incoming conflict status when
the existing canonical fact has another status, including missing legacy
status. SQLite holds one lock through lookup, validation, merge, and write;
the service's existing transaction rolls back fact, evidence, source-status,
and job-result changes together. Maintenance records the stable
`conflict_canonical_collision` error and retains the pending source for its
retry policy. New targetless conflicts and repeats of existing conflicts are
still accepted, with evidence deduplication.

Six new regressions cover canonical normalization, legacy rows, preserved
evidence after reopen, mixed-batch rollback, retryable failure, and valid
conflict writes. Thirty-one focused conflict, evidence, maintenance, and service
checks passed. The parent's subsequent 35-test combined integration run also
passed with the role-aware in-flight-forget fixture. Black and compatible
strict mypy passed. This closes the observed relabeling path; it does not prove
that other extracted claims are semantically supported by their sources.

## Graph capacity diagnostic completed

The corrected offline diagnostic ran the same five frozen development cases
once at each neighbor capacity: 4, 8, and 16. Eligible counts were 0/5, 0/5,
and 1/5; successful recall was 0/5 in every arm. The one eligible capacity-16
case had a two-hop activation path with energy 0.45954, but its target did not
appear in the five graph results. A retained path alone did not satisfy the
recall criterion.

All 75 constructed stores persisted the intended capacity. Activation-oracle
and insertion-order controls passed for all 15 observations. The diagnostic
used unchanged frozen d94 source and vectors, with no model calls. Production
capacity, datasets, and acceptance gates remain unchanged. Earlier v1 higher
arms changed capacity after construction and are excluded from conclusions.
[Raw evidence and limits](graph-capacity-diagnostic-v2/README.md).

## Completed metadata-write comparison

The fixed baseline/candidate/candidate/baseline comparison completed all four
arms. Each loaded 10,000 synthetic 768-dimensional facts and timed 100 durable
five-ID access updates under WAL/FULL. Reopened counters were all 100, and both
schema-8 candidates retained all 10,000 metadata rows.

The two candidate p95 values were 137.821 and 82.437 ms. Baseline p95 varied
from 1,985.947 ms in the first arm to 82.747 ms in the last arm. This does not
establish a reliable latency improvement. The host was not isolated, and these
are metadata writes rather than end-to-end recall requests. The 500 ms recall
gate remains open. Source, actual imports, helper hashes, and unchanged-file
checks bind every arm to frozen d94 or 2494 source; all raw measurements remain
preserved. [Comparison archive](access-metadata-comparison-v1/README.md).

The new [claim and evidence ledger](claim-evidence-ledger.md) maps the twelve
original capability promises to verified artifacts and remaining acceptance
gates. It also clarifies that configured embedding revision is an identity
label, not automatic verification of a mutable Ollama tag's digest.

## Backup during an active writer transaction

The additional T16 regression uses a separate SQLite connection and thread
barriers to hold two uncommitted row updates while the owning connection
creates a backup. After the writer commits, the live database contains both
new values, while the restored backup contains both prior committed values.
The restored registry, fact count, and SQLite integrity also pass.
`tests/test_backup_concurrent.py` and `tests/test_backup_restore.py` passed
all four tests. This uses temporary databases, not live memory data.

The evaluation-status page now identifies the active v4 queue and explicitly
labels the old v3 status as historical. Raw reports were not changed.

## Source-backed continuity fixture

The planned `benchmarks/scenarios/local-memory-continuity.jsonl` now records
synthetic source events and expected outcomes consumed by the authenticated
service test. Three cases rotate the teaching adapter through Codex, Claude
Code, and Hermes. They retrieve after teaching and restart, correct with a
stable fact ID and advancing version, and retrieve the correction after a
second restart. Each project has a positive retrieval control before all
three adapter contexts check exclusion in the other project.

Forget, source replay, maintenance, and a final restart leave both old and
corrected content absent while the unrelated project's canary remains
retrievable. The parent combined run of cross-host, MCP compatibility, and
documentation checks passed 18 tests in 57.38 seconds. Black passed for the
continuity and concurrent-backup tests. Providers remain synthetic, and no
native-host or real-model gate is promoted by these checks.

## Combined verification found a benchmark compatibility regression

Frozen source `9aba209` finished with 856 passed, four skipped, and one failed
test in 991.85 seconds. The performance harness directly called
`embed_document` on a legacy provider exposing only `embed`. An isolated
reproduction produced the same AttributeError. Commit `17c55ef` routes the
seed vector through the shared role dispatcher, retaining document-role
selection and the legacy fallback.

The existing access-write regression now checks both unprefixed and explicitly
prefixed payloads, retaining one metadata update per engine query and zero
follow-up measurement writes. The affected benchmark and embedding-role
tests passed 51 cases in 305.52 seconds. Black and strict mypy passed for the
changed files. The original full failure and separate passing focused XML are
[preserved](combined-package-failure-v1/README.md); package acceptance still
requires a fresh combined full run.

Claude 2.1.191 separately recognized the plugin and five hooks using its real
`--plugin-dir` metadata commands in a disposable configuration. The source
binding remained unchanged. List and details disagreed about MCP inventory;
neither hook execution nor MCP connection ran.
[Loader metadata evidence](claude-plugin-loader-v1/README.md).

## Full source pass and installed continuity failures

Frozen `312b8d6` passed 858 tests with four skipped, plus Black, strict mypy,
JavaScript syntax, build, and Twine checks. The fresh installed wheel then
passed 40 selected contract cases and failed two continuity cases. The
downstream replay check did not run. Source and installed results remain
separate in the [package checkpoint](combined-package-checkpoint-v2/README.md).

A valid isolated diagnostic reproduced a client timeout while the initial
fact remained persisted. One-fact retrieval metadata writes sometimes took
more than three seconds. A subsequent instrumented run passed both cases;
its slowest metadata write was dominated by a 368 ms SQLite commit. No
product repair occurred between these diagnostic runs, so the later pass
does not establish that the failures are fixed. See the
[timeout reproduction](installed-continuity-diagnostics-v1/README.md) and
[SQL timing record](installed-continuity-sql-profile-v3/README.md).

The continuity test now rejects degraded/error retrieval replies before
checking scope exclusion, and requires actual successful maintenance results.
Those three strengthened cases passed. The Claude plugin launcher also now
shares the managed runtime configuration between hooks and MCP, preserving
explicit settings instead of selecting plugin-local mock defaults. Its
focused integration checks passed; actual native plugin execution remains
an open gate.

A disposable SQLite comparison measured median durable commits of 2.040 ms
on C: and 70.805 ms on D: with identical WAL/FULL settings. It supports an SSD
fixture diagnostic, not a product latency claim. The
[comparison archive](storage-drive-calibration-v1/README.md) preserves all
samples and the executed helper. Full verification of frozen `82b7116` has
started with fresh C: source and temporary fixtures; that result and a fresh
installed-package run are pending. Previous D: failures remain preserved.

## SSD package and installed continuity checkpoint

The first C: full run passed 864 tests, skipped four, and failed one documentation
test because its ancestor-based website lookup could not locate the separate
D: checkout. The [failed run](ssd-source-check-failure-v1/README.md) is preserved.
The test now accepts an explicit website input; its assertions are unchanged.

Frozen `6600dc2` then passed 865 tests with four skips in 177.02 seconds, followed
by Black, strict mypy, JavaScript syntax, build, and Twine. The website file's
SHA-256 matched before and after verification. A fresh installed wheel passed
the same 42 selected contract cases in 23.22 seconds, with every loaded Mnemos
module verified to come from that installed environment. Synthetic Stop replay
also passed all seven positive assertions, including unchanged event identity
after restart and zero replay capture. Helper, source, input, and wheel bindings
were unchanged across installation and validation.

The [package archive](ssd-package-checkpoint-v1/README.md) records this C: result
separately from the D: failures. No relaxed timeout, weaker durability setting,
or product performance repair is claimed. Native plugin execution, full native
cross-host continuity, real-model utility, recall latency, and pilot gates remain
open. No live configuration or memory data was changed.

## Native Claude plugin execution observed

The verified `6600dc2` wheel passed a fresh installed adapter timing preflight.
Native Claude 2.1.191 then loaded the actual plugin and executed SessionStart
and UserPromptSubmit, but the v9 turn stopped at its budget limit before Stop.
Account-level connectors had been inherited despite the private configuration
directory. Its reported $0.365532 cost and failure remain preserved.

The v10 test disabled account MCP connectors only in its child environment,
using Claude's documented `ENABLE_CLAUDEAI_MCP_SERVERS=false` setting. The turn
completed in 10.567 seconds, reported $0.0291 cost, executed all three lifecycle
hooks, returned the seeded memory canary, and reported Mnemos connected with
fourteen tools. No native MCP tool call ran. Restart retained the captured
events, and a reconstructed Stop payload through the actual plugin replayed the
native transcript without adding another event. It was not a replay of the
original native hook payload.

The overall probe remains `not_observed`: three raw captures include one
intentionally unassociated provisional user event, so the exact-two gate and
aggregate user/transcript association check fail. The separate provisional
identity contract passes, including no facts or evidence from that event.
This preserves ambiguity rather than merging repeated identical prompts by
text. [Reports and executed helpers](native-plugin-checkpoint-v1/README.md).

## Graph ranking sensitivity separated from connectivity

The offline [strict graph-weight diagnostic](graph-weight-diagnostic-v2/README.md)
held five development cases, capacity-16 stores, candidates, and activation
fixed across weights 0.0, 0.1, 0.2, and 0.4. The one structurally eligible target
ranked 34, 33, 16, and 3; four other targets remained unreachable. Every candidate
was scored with its own energy, and the production top-five list matched the
recorded ordering for all twenty observations. The default weight remains 0.1.

The strict run verified clean d94 source and unchanged package/benchmark,
vector, and input-report hashes before and after. It made no provider calls.
The exploratory v1 report is preserved with its weaker provenance and unknown
prior attempt history. Neither run establishes held-out graph utility or closes
the coverage gate. Increasing ranking weight cannot repair a missing path.

## Native MCP read attempt and power interruption

The v11 native Claude read probe completed in 23.229 seconds and reported
$0.027801. All three lifecycle hooks, transcript association, provisional
extraction exclusion, durable restart, and reconstructed replay passed. The
MCP server remained pending, exposed no tools, and made no calls. The correct
seeded answer was available through hook context; it is not MCP retrieval
evidence. The [unchanged report and executed helper](../2026-09-17/native-claude-mcp-v11/README.md)
preserve the failed read contract separately from the earlier v10 connection.

The local machine restarted at 18:09 UTC during a user-confirmed power outage.
The evaluation process handle and worker PIDs disappeared while the saved queue
status still said running. A recovery preflight initially timed out reaching
Ollama before changing any files. The endpoint subsequently returned both
original model digests. The queue must pass its existing provenance checks
before resuming from preserved checkpoints; [evaluation status](evaluation-running.md)
tracks that recovery separately from quality acceptance.

Recovery completed at 18:29 UTC with the original runner, source, model digests,
and options. Every completed-phase artifact digest matched. The archived stale
lock was retired only after proving the old processes absent; the coordinator
resumed with `--resume` and skipped completed phases. The resumed coordinator
and mechanism worker were observed live. The checkpoint retained 62 completed
case/pair IDs, 112 graph rows, and 150 affective rows, including partial cases.
No held-out quality fields were inspected for recovery. See the
[interruption snapshot](../2026-09-17/queue-interruption-v1/README.md) and
[recovery evidence](../2026-09-17/queue-recovery-v1/README.md).

## Native Claude MCP reads observed

The v12 comparison changed permission mode from plan to default while retaining
the two exact allowed read tools, empty built-in tool set, plugin/runtime
bindings, private fixture, and fixed time/cost limits. The native turn completed
in 18.715 seconds and reported $0.0260907. It called health then retrieve with
correlated successful results. Health matched the fixture database, profile,
and project; retrieve returned the project-scoped canary. All bounded read
criteria passed, including capture identity, restart, and reconstructed replay.

Four normal transcript events and one provisional user event were retained.
The old exact-two raw gate remains false; every normal event was positively
associated and the provisional event remained extraction-ineligible. The
[report and executed helper](../2026-09-17/native-claude-mcp-v12/README.md)
preserve these separate outcomes. This comparison supports a mode-related
hypothesis without proving it the sole cause of the prior pending state.
Providers remain synthetic, so this does not establish semantic quality or
complete native correction/forget continuity.

## Provider failure log privacy

An operational audit found provider wrappers copying exception messages into
structured failure events. Eighteen synthetic regression cases reproduced a
private-content and authorization canary appearing in the log across Ollama,
OpenAI-compatible, and OpenRouter generation and embedding paths. The wrappers
now emit `error_type` instead of `error`, while rethrowing the same exception and
preserving fallback behavior. All eighteen cases pass, including ordinary and
configuration failures, single/batch embeddings, and credential fallback paths.

The combined provider, embedding, and Ollama checks passed 41 cases; Black and
strict Python 3.10 mypy passed the changed source files. The frozen model queue
and installed native-client fixture remain unchanged. This focused check does
not replace a new full/package verification of later service logging changes.

## Installed doctor configuration checks

The installed wheel from source `6600dc2` passed three isolated doctor cases:
an empty SQLite store reported `ready`, one legacy unscoped row reported
`degraded`, and explicit mock cognition reported `not_ready`. The configured
Ollama endpoint and non-loopback socket connections were blocked in the child
bootstrap. This verifies configuration readiness and legacy-row reporting;
it does not verify model reachability. The report checks installed Python
files against the wheel before and after each case.

The [doctor archive](../2026-09-17/installed-doctor-v5/README.md) includes
the executed helper, reports, and command output. Earlier attempts remain
separate, including two failed probes. Earlier helper bytes were not preserved
for every attempt; only v5 carries the full checked execution record.

## Managed service logging and omitted-provider startup

The daemon previously discarded console output without installing a persistent
log handler. It now owns a private rotating `service.log` next to its runtime
descriptor. Startup, retrieval timing, and provider/storage error categories
are retained with selected fields; exception text, query text, scope IDs, and
other content are excluded. Cleanup closes the handler before releasing the
process lock and preserves existing maintenance observers. A failed start
clears its readiness state before a retry.

The actual client auto-start test exposed another bug: any existing TOML file
was treated as an explicit choice of mock cognition when `llm.provider` was
omitted. Settings now retain whether that field was supplied. Omitted cognition
uses `UnavailableLLMProvider`; explicit mock remains rejected. Regression cases
cover global and project configuration. The subprocess test now starts the
default service with simple embeddings, commits and retrieves a fixture fact,
observes startup and retrieval log events, and confirms authenticated shutdown.
No live model or live database participates in that test.

The focused service/configuration/maintenance/provider run passed 76 cases.
After preserving the previous maintenance callback and adding the explicit
project-mock rejection case, the affected service-log/runtime checks passed
27 cases. Black passed. Full/package verification of these changes remains
separate from the earlier `6600dc2` checkpoint.

## Native Claude lifecycle v13: negative result

The [v13 archive](../2026-09-17/native-claude-lifecycle-v13/README.md) preserves
a failed native correction phase against the installed `6600dc2` package.
Three correlated native tools completed in 21.127 seconds, but the evaluator
did not establish queued correction, completed consolidation, or corrected
retrieval from their payloads. The forget phase did not run. A Stop diagnostic
also reported degraded capture despite the host labeling the hook successful.
Parser and hook diagnostics are being investigated; this attempt does not
close native lifecycle acceptance. Its report and executed helper remain
unchanged, and cleanup removed the copied auth file and runtime descriptor.

An [offline payload diagnosis](../2026-09-17/native-claude-v13-payload-diagnosis/README.md)
subsequently recovered the actual native tool responses: Claude wrapped JSON
inside a string-valued `result` field. The corrected parser establishes all
phase-one operation assertions, including the seeded fact ID, version 2,
replacement-content hash, and project scope. The stored stream also matches
the original run's text digest after Windows newline normalization. This is
derived evidence from the same attempt, not a new native run. UserPromptSubmit
and Stop still contain degraded diagnostics, and the restart/forget phases
remain unrun; the overall lifecycle result stays failed.

## Service/runtime full checkpoint: failed continuity request

The clean `e1fd56e` full suite finished with 897 passed, one failed, and four
skipped in 460.15 seconds. The synthetic Codex continuity case reached its list
step after successful remember and maintenance, then failed because the
response lacked `facts`. The failure output does not include the returned
response, so a timeout or other degraded response is a hypothesis pending a
focused diagnostic. The [original full run](../2026-09-17/service-runtime-checkpoint-failure-v1/README.md)
is preserved, and the build/install pipeline stopped at this failed gate.
