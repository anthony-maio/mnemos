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
