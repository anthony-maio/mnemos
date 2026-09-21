# Functional Local Memory Implementation Plan

Date: 2026-09-16
Status: Implementation in progress in `wt/functional-local-memory`. See `docs/validation/2026-09-16/progress.md` for measured results and open gates.

**Goal:** Deliver dependable local memory shared by Codex, Claude Code, and Hermes, including the five biomimetic modules, with evidence for every advertised capability.

**Architecture:** Preserve the composable Python engine and SQLite backend. Introduce a persistent local runtime that owns database writes, model execution, and maintenance. Host adapters capture events and request scoped context through that runtime; explicit MCP tools remain available. Captured evidence, accepted facts, and revisions have distinct lifecycles.

**Tech stack:** Existing Python, Pydantic, SQLite/FTS5/sqlite-vec, FastEmbed, Ollama, httpx, and MCP. Do not introduce a hosted database or require cloud inference. Keep additional provider dependencies optional.

## 1. Decisions and boundaries

Confirmed requirements:

- Local storage and local model processing by default; cloud providers are explicit opt-ins.
- Codex, Claude Code, and Hermes share memory for the same user while respecting project/workspace/global scope.
- Implement and evaluate surprisal gating, mutable RAG, affective routing, sleep consolidation, and spreading activation. Passing storage tests alone does not meet this goal.
- Produce an implementation plan now. Do not deploy, publish, change installed host configuration, or migrate live memories as part of planning.

Recommended defaults, not yet separately accepted product settings:

- Automatic capture only in enabled projects, plus explicit remember/correct/forget operations. No bulk historical import by default.
- Retain processed raw events for 30 days; retain accepted facts until corrected, forgotten, or explicitly expired. Never silently discard failed/unprocessed events under the processed-event retention rule.
- Retain a redacted supporting excerpt and source identity with each accepted fact after raw-event expiry. Show when the original source is no longer available.
- Start with SQLite as the only supported local runtime database. Restore Qdrant only as an optional compatibility extension after the core acceptance gates.
- No cloud fallback when a local model fails. Capture and existing recall remain available; enrichment waits visibly.
- No automatic execution of generated procedural tools. Team collaboration, remote sync, Neo4j, and universal desktop surveillance are outside this release.

The optional Qdrant extension addresses historical compatibility; it must not delay a dependable SQLite release. The other explicitly future research directions remain future work.

## 2. Audit baseline and evidence

Repository: `D:\Development\mnemos\mnemos`, local commit `5fcc925d5a90a087e5715c2fe6de56fa23a7d3a6`, declared version 0.7.0. Public upstream observed at `8033bd61c9c1249b8c51b1fc6d47f94d74b37fef`; PyPI currently serves 0.6.0. Installed 0.2.0 distribution metadata is stale and points to this checkout through an editable install. Do not treat the metadata as a separate verified running implementation.

The outer directory is a separate website checkout with pre-existing edits to `web/index.html` and `web/style.css`. Preserve those edits. No application changes were made during this audit.

Validation already performed in this task:

- `py -3.13 -m pytest -q tests/test_durable_memory_repair.py`: 6 passed.
- `py -3.13 -m pytest -q tests/test_engine.py tests/test_modules.py tests/test_storage_sqlite.py tests/test_hook_autostore.py`: 115 passed.
- Isolated probes reproduced fabricated default mock facts, cross-project prediction context, failed-event non-recovery after restart, forgotten-fact regeneration, and FastEmbed rejection by health checks.
- A four-node graph probe using edge weight 0.8, decay 0.2, threshold 0.3, and three hops returned three nodes; the seed accumulated energy from a return edge.
- A separate reconsolidation probe changed the stored React fact to Svelte while its graph node retained React; subsequently storing the identical Svelte fact produced two current rows. Both the stale graph and stale canonical-key consequences are runtime-reproduced.
- These are source and synthetic-runtime findings. No new real-host, real-model, two-week pilot, full-suite, or published-artifact acceptance is claimed.

### Capability inventory

| ID | Capability | Current evidence and gap | Delivery tasks |
| --- | --- | --- | --- |
| C01 | Persistent local memory | SQLite facts exist. Public episodes are process-local; local durable repairs are unreleased and retry handling is incomplete. Bare engine defaults are temporary/mock. | T01-T05 |
| C02 | Project/workspace/global isolation | Retrieval filtering exists; prediction history leaks between scopes, identities use names/defaults, and workspace membership is not modeled independently. | T02, T07, T10 |
| C03 | Surprisal-gated encoding | Distance and threshold machinery exist; predictions run on ingestion, explicit writes can be gated, history is not session-safe, and real-model utility is unproved. | T07, T18 |
| C04 | Mutable RAG | Rewriting/version history exist; retrieval query is used as correction evidence, queues have inconsistent completion behavior, and derived graph/canonical state needs synchronization. | T08, T10 |
| C05 | Affective routing | Classifier and weighted scorer exist; engine later combines ranks, top-1 bypasses the module, and fallback state can come from unrelated history. | T09, T18 |
| C06 | Sleep consolidation | Extraction exists; no persistent entrypoint starts the daemon, retries/atomic completion are incomplete, and marking episodes processed does not bound disk growth. | T03-T06, T11 |
| C07 | Associative retrieval | Sparse persisted edges exist; graph seed selection is unscoped, traversal uses shared mutable energy, and startup/full maintenance can scan too much data. Scoped SQLite retrieval also bypasses sqlite-vec. | T09-T10, T18 |
| C08 | Composable safe modules | Public module classes exist; all supported write paths need equivalent scope/safety contracts, including use outside the engine. | T02, T06-T11 |
| C09 | Inspectable, correctable memory | Viewer/history/feedback exist; explanations can differ from actual ranking, source-to-fact links are incomplete, and forget can resurrect content. | T06, T08, T09, T11, T15 |
| C10 | Guided setup/readiness | UI and host config exist; FastEmbed is missing from doctor, default cognition is mock, and host names/config targets require verification. | T01, T12-T15 |
| C11 | Cross-host continuity | Claude hooks exist; Codex instructions rely on agent compliance; Hermes adapter is absent. Current host APIs provide stronger integration paths. | T12-T14, T17 |
| C12 | Proven retrieval/compression advantages | Mock-based tests characterize mechanics, not extraction fidelity or practical superiority. Baselines, ablations, actual bytes, and held-out workflows are needed. | T16-T19 |

### Paper evidence that needs reconciliation

The March paper describes v0.2.0, mock-provider experiments, Qdrant support, and production readiness. Its 40% noise-reduction headline does not follow clearly from the reported default storage rates. A 0.3 cosine distance implies about 45.6 degrees, not 72; distance ranges to 2. A reported graph energy of 0.26 is below its 0.3 threshold. The compression sequence is linear in episode count when output stays at three facts. Treat these as reproducibility and documentation issues, not numerical implementation targets. Reassess competitor absolutes from dated primary sources. [Paper](https://making-minds.ai/papers/mnemos.pdf)

The website promises scoped local persistence, guided setup, adaptive retrieval, and automatic maintenance. Its runnable examples, host labels, defaults, and release claims must match the tested installed artifact. [Website](https://mnemos.making-minds.ai/)

## 3. Runtime and data contracts

### Ownership and transport

- Add one per-user `mnemos serve` runtime with a single writer queue and long-lived model instances. Core library objects remain usable with explicitly supplied providers for embedded applications and tests.
- Use a loopback HTTP JSON API on an OS-assigned port, advertised in a user-private runtime descriptor. Require a generated token; reject non-loopback binding, unexpected browser origins, and unauthenticated mutations. Never put the token in URLs or logs.
- Implement transport with the existing standard-library HTTP-server approach and httpx client. Handlers submit work to one owning asyncio loop; they do not open competing database writers. Use an OS process lock and hidden Windows process startup. No additional web framework is required for this release.
- Keep `mnemos-mcp` stdio compatibility as a thin client to this service. CLI and UI use the same service. Auto-start uses a process lock so simultaneous clients cannot create multiple writers.
- A small local append-only outbox buffers sanitized host events if the service is unavailable. Acknowledgement distinguishes queued locally from committed to the service. Redelivery is idempotent.
- CLI/MCP fallback is one bounded restart attempt, then explicit degraded status and an outbox for writes. Never open the live database as a competing direct-write fallback. Standalone embedded usage remains available with explicitly supplied storage outside the managed service database.
- For Windows clients and Hermes in WSL, use an explicit bridge command to the Windows runtime when loopback is unavailable. Do not expose a LAN listener or open the SQLite file across the Windows/WSL filesystem boundary.
- Persistent state and processing are local by default. An absent extraction model is an honest pending state, never an invitation to fabricate a response.

### Minimum additive interfaces

Keep existing MCP names and result fields compatible for one release. Add fields rather than silently changing string/JSON types. Add `schema_version` to new structured responses.

```python
class ScopeContext(BaseModel):
    user_profile_id: str  # resolved by local setup, not inferred from folder names
    project_id: str | None = None
    workspace_id: str | None = None
    session_id: str
    host: Literal["codex", "claude-code", "hermes", "manual"]

class CaptureEvent(BaseModel):
    event_id: str  # stable per originating host event
    context: ScopeContext
    occurred_at: datetime
    role: str
    content: str  # sanitized before persistence or model use
    source_kind: str
    source_ref: str | None = None

class FactProposal(BaseModel):
    content: str
    evidence_ids: list[str]
    operation: Literal["add", "support", "correct", "conflict"]
    target_fact_id: str | None = None
    expected_version: int | None = None

class RetrievalRequest(BaseModel):
    query: str
    context: ScopeContext
    allowed_scopes: list[str]  # default project + explicitly shared global
    top_k: int = 5
    max_context_tokens: int = 1200
    profile: Literal["baseline", "biomimetic"] = "baseline"
```

- Add service operations `capture`, `remember`, `retrieve`, `correct`, `forget`, `inspect`, `maintenance`, and `health`; version their schemas. Add `mnemos_capture` and `mnemos_correct` MCP tools without removing current tools.
- `remember` is an explicit requested write and bypasses surprise rejection, but never bypasses safety or scope checks. `capture` stores evidence for later processing. An assistant statement is attributed to the assistant, not silently elevated to a user preference or observed fact.
- Retrieval returns content, fact/version identity, source references, relevance components, truncation information, and capability/degradation status. It can update access metadata, but cannot modify fact content from a query alone.
- Salience remains explicit encoding metadata in the initial scoring contract. Do not imply it is a retrieval weight. An optional salience ranking contribution may be promoted only through the same held-out ablation gates as the other ranking changes.
- Resolve project identity through a local registry keyed to the canonical Git common directory or canonical non-Git root. Worktrees share an identity; unrelated clones remain distinct unless linked explicitly. Maintain Windows/WSL path aliases in the registry.
- Workspace membership is explicit and separate from project identity. Global writes require an explicit scope choice. Missing/ambiguous identity fails safely into a visible unassigned queue, never a shared `default` project.
- The registry maps a stable local user profile to projects, workspace memberships, and explicitly linked host profiles. Enabled hosts share the selected profile; a different Hermes profile is not silently linked. Validate workspace membership before accepting caller-provided context. Global means shared within this local user profile, not across every profile on the machine.
- Legacy unscoped facts stay available in a migration review view and are excluded from automatic scoped retrieval until assigned. Do not erase them or assume they are global.

### Storage and lifecycle

- Extend current tables through numbered migrations: event source/session identity, work jobs with retry leases, fact evidence links, revision records, suppression records, project registry, and embedding-space identity.
- Use unique source-event keys for duplicate delivery, not content alone. Identical statements in separate turns are distinct observations. Preserve case-sensitive code when normalizing fact content.
- Keep a stable fact ID across corrections. Recompute the content-deduplication key and refresh vector/FTS/graph state on accepted corrections. Do not use stale content hashes as semantic identity.
- Claim jobs with a lease; execute model calls outside transactions; atomically commit accepted results, revision/evidence links, and job completion. Retry provider failures with bounded exponential backoff. Exhausted jobs remain visible and manually retryable; restarts recover expired leases.
- Forget removes the selected fact and its retained evidence copies, queued descendants, revisions containing its content, vectors, and graph references. Record non-content suppression identities so replay of old events cannot recreate it. A new explicit remember can supersede that suppression. Historical backups are separately labeled and managed; do not claim forensic secure deletion.
- Backups use SQLite's backup API, include schema/model metadata, and pass restore/integrity checks before migration. Preserve the original database and require explicit application of ambiguous mappings.

## 4. Implementation task graph

Paths below are relative to `D:\Development\mnemos\mnemos`. Each task is a bounded change; large tasks should be split into the listed contract/test, implementation, and integration steps before coding. One owner at a time for shared engine/storage/types files.

For each task: write the behavior regression first, demonstrate the failure, implement the smallest complete change, run its focused checks, review the diff, and record the result. Do not add tests that only mirror implementation details. After a passed batch, make a focused checkpoint commit when commit authorization exists. Do not publish or merge as an implicit consequence.

### T01 - Establish a truthful install and runtime baseline

Depends on: none.

Files: `mnemos/runtime.py`, `mnemos/settings.py`, `mnemos/health.py`, `mnemos/engine.py`, `mnemos/__init__.py`, `tests/test_runtime.py`, `tests/test_health.py`.

Steps: add a configured-engine factory shared by application entrypoints; reserve bare/mock engine construction for explicitly labeled development use; prohibit mock-generated facts in normal profiles; support FastEmbed health and real readiness probes. Report package version, source revision when available, provider/model/cache, database path, and missing capabilities. Do not call configured credentials proof of inference.

Acceptance: clean config cannot invent preferences; the configured SQLite factory survives restart; missing models leave capture usable; FastEmbed is recognized; health distinguishes configured, reachable, model-ready, and failed states.

Run: `py -3.13 -m pytest -q tests/test_runtime.py tests/test_health.py tests/test_settings.py`.

### T02 - Enforce scope and identity across every computation

Depends on: T01.

Files: new `mnemos/scopes.py`, `mnemos/types.py`, `mnemos/engine.py`, `mnemos/modules/surprisal.py`, `mnemos/modules/affective.py`, `mnemos/modules/spreading.py`, `tests/test_engine.py`, new `tests/test_scope_context.py`.

Steps: implement the registry/context contract and membership validation as a pure resolver with config serialization first; partition session histories; carry context into prediction, state fallback, candidate search, graph traversal, extraction, and corrections. Remove duplicate independent scope-normalization rules. Add safe compatibility handling for explicit old scope IDs. T03 persists this tested registry contract with migrations; T02 does not depend on a not-yet-created database schema.

Acceptance: canary text from one project never enters another project's prompts or outputs; equal directory names do not collide; linked worktrees share memory; workspace inclusion requires membership; unresolved legacy scope is not global.

Run: `py -3.13 -m pytest -q tests/test_scope_context.py tests/test_engine.py tests/test_modules.py`.

### T03 - Add durable evidence and transactional job processing

Depends on: T02.

Files: `mnemos/types.py`, `mnemos/utils/storage.py`, new `mnemos/migrations.py`, new `mnemos/jobs.py`, `tests/test_durable_memory_repair.py`, new `tests/test_job_recovery.py`.

Steps: add additive schema migration and event identity; implement leased jobs, retry state, and atomic finalization; backfill existing pending/failed records without losing originals. Before any live upgrade, create and restore-check a consistent SQLite backup, using the foundation later expanded in T16. Exercise both in-memory contract tests and real SQLite transactions.

Acceptance: crash before/after model completion produces no lost event or duplicate fact; expired leases recover; provider outage followed by restart retries; separate repeated observations count separately; duplicate hook delivery counts once.

Run: `py -3.13 -m pytest -q tests/test_durable_memory_repair.py tests/test_durable_jobs.py tests/test_job_recovery_api.py tests/test_storage_sqlite.py`.

### T04 - Introduce the persistent local service and client

Depends on: T03.

Files: new `mnemos/service.py`, new `mnemos/client.py`, new `mnemos/outbox.py`, `mnemos/cli.py`, `mnemos/mcp_server.py`, new `tests/test_service.py`.

Steps: implement the authenticated loopback protocol and runtime descriptor; add singleton startup/stop/status; run a single writer/event loop with bounded work queues; turn application entrypoints into clients; keep the library API available for explicit embedding.

Acceptance: simultaneous startup creates one writer; service restart drains outboxes once; no model is loaded per hook; unavailable service never claims a committed write; unauthorized/browser cross-origin requests are rejected; host work continues with a visible degraded result.

Run: `py -3.13 -m pytest -q tests/test_service.py tests/test_mcp_server.py tests/test_cli.py`.

### T05 - Run reliable background maintenance

Depends on: T04.

Files: new `mnemos/maintenance.py`, `mnemos/modules/sleep.py`, `mnemos/service.py`, `mnemos/settings.py`, new `tests/test_maintenance.py`.

Steps: start a scheduler owned by the service; use a 30-second check, a default 60-second idle delay, and batches capped at 100 events per scope; session-end requests enqueue work without blocking on inference. Drain oldest eligible work first. Limit concurrency and expose queue age, retry count, last success, and last error.

Acceptance: consolidation happens without an agent calling a tool; active capture does not corrupt an in-flight snapshot; continuous activity cannot indefinitely starve queued old work; shutdown/restart retains pending work.

Run: `py -3.13 -m pytest -q tests/test_maintenance.py tests/test_durable_jobs.py tests/test_job_recovery_api.py` with a controlled clock, not real sleeps.

### T06 - Enforce evidence and safety at supported write boundaries

Depends on: T03; integrate service routes after T04.

Files: new `mnemos/memory_service.py`, `mnemos/memory_safety.py`, `mnemos/curation.py`, `mnemos/modules/surprisal.py`, `mnemos/modules/mutable_rag.py`, `mnemos/modules/sleep.py`, `tests/test_memory_safety.py`, new `tests/test_fact_evidence.py`.

Steps: centralize sanitized writes and structured fact proposals; require source references for extracted claims; preserve user/assistant/tool provenance; reject unsupported evidence IDs and cross-scope references. Every public write-capable module constructs the default firewall when none is supplied, so standalone module use remains supported and safe under the configured policy. Raw storage is a documented low-level primitive, not a safe ingestion API. Policy tests establish coverage of configured detectors, not a claim that regex can detect every possible secret or personal detail.

Acceptance: safety runs before disk, outbox, logs, or cloud requests; extraction cannot cite absent sources; assistant speculation is labeled; prompt-injection content remains data and cannot authorize global writes or change runtime settings.

Run: `py -3.13 -m pytest -q tests/test_memory_safety.py tests/test_memory_service.py tests/test_durable_jobs.py tests/test_modules.py`.

### T07 - Complete surprisal gating as a measurable feature

Depends on: T02, T06.

Files: `mnemos/modules/surprisal.py`, `mnemos/config.py`, `mnemos/types.py`, new `tests/test_surprisal_contract.py`.

Steps: compute predictions from earlier same-session evidence only; exclude the event being scored; support precomputed next-intent predictions in the worker and an on-demand fallback. Preserve raw distance in [0, 2], clamp salience separately, and report prediction/model/threshold/status. Gate automatic long-term encoding, not durable capture. Explicit remember bypasses the gate.

Acceptance: deterministic distance boundaries pass; critical explicit instructions survive low surprise; unavailable prediction stays unknown; no cross-session history or fabricated score; cached/on-demand modes produce auditable results.

Run: `py -3.13 -m pytest -q tests/test_surprisal_contract.py tests/test_modules.py`.

### T08 - Make reconsolidation evidence-based and reversible

Depends on: T03, T06.

Files: `mnemos/modules/mutable_rag.py`, `mnemos/types.py`, `mnemos/utils/storage.py`, `mnemos/engine.py`, new `tests/test_reconsolidation_contract.py`.

Steps: recall marks a fact eligible for later review but does not enqueue a query as correcting evidence. New same-scope evidence or explicit corrections produce proposals against an expected fact version. Store conflicting assertions separately when authority/time cannot be resolved. Commit revision, new embedding/content key, and invalidation of derived graph state atomically.

Acceptance: asking "did we switch to Rust?" cannot rewrite a Python preference; an explicit dated correction can; hypothetical/quoted instructions cannot; stale concurrent proposals cannot overwrite newer revisions; unchanged review does not make old content appear newly observed; job completion is consistent across live/restarted workers.

Run: `py -3.13 -m pytest -q tests/test_reconsolidation_contract.py tests/test_durable_memory_repair.py tests/test_storage_touch.py`.

### T09 - Unify retrieval scoring and affective routing

Depends on: T02, T06; final graph integration after T10.

Files: `mnemos/modules/affective.py`, `mnemos/engine.py`, `mnemos/inspectability.py`, `mnemos/config.py`, new `mnemos/retrieval.py`, new `tests/test_retrieval_contract.py`.

Steps: retain scored candidates through the engine instead of replacing scores with ranks. Add baseline and biomimetic profiles; in the latter, use the documented 0.7 semantic/0.3 state component with explicit, separately recorded graph and freshness contributions. Enable modules independently of top_k. Normalize lexical/vector results consistently; enforce a relevance floor and context budget. Record exact retrieval traces for inspection. Replace Python callbacks on the normal scoped path with indexed SQL scope predicates; run scoped vector and FTS candidate search before ranking. Keep a declared fallback when sqlite-vec is unavailable, and verify exact scoped top-k correctness rather than taking an unscoped top-k and filtering it afterward.

Acceptance: top_k=1 can exercise the full profile; missing state uses semantic-only scoring and a reported fallback; no unrelated prior state is borrowed; inspect matches the actual returned ranking; no-match queries can return empty; the context budget is respected. Adding unrelated projects cannot force all their records into Python for an ordinary scoped search. Emotional state never changes fact authority or scope.

Run: `py -3.13 -m pytest -q tests/test_retrieval_contract.py tests/test_inspectability.py tests/test_engine.py`.

### T10 - Correct and bound associative graph retrieval

The [standalone visit-budget repair](../validation/2026-09-20/spreading-visit-budget-v1.md)
reproduces three configuration failures and a passing explicit-override control.
Standalone calls now use the configured limit when no call override is supplied.
The engine already supplied that limit explicitly; existing retrieval negatives
are unchanged. The 54-test graph and 73-test source-association selections pass.
The full source/package checkpoint passes 2,158 tests with four skips and all
six checks; fresh installed-package verification is in progress.

Depends on: T02, T03; integrate revisions from T08.

Files: `mnemos/modules/spreading.py`, `mnemos/utils/storage.py`, `mnemos/engine.py`, new `tests/test_spreading_contract.py`.

Steps: select seeds and traverse the permitted subgraph: current project, explicitly permitted member workspace, and explicitly shared global facts within the same user profile. Exclude foreign project nodes even as intermediate hops. Make energy state query-local. Define algorithm v2 as maximum propagated path energy, `next=current*(1-decay)*edge`, with threshold applied consistently, max-hop and max-visited budgets, and no cycle amplification. Persist graph schema, edge-builder, and traversal versions. On incompatibility, rebuild derived edges in a new generation while baseline recall remains available; never silently reinterpret incompatible persisted state. Maintain nearest eligible neighbors incrementally; load required persisted neighborhoods rather than truncating the entire usable graph to the first startup page. Preserve a legacy algorithm only in reproduction fixtures if needed. Keep `decay_all()` as a documented standalone/legacy primitive; do not imply persistent natural decay when each retrieval starts a fresh activation state.

Acceptance: hand-computed chains/branches/cycles pass; four-node default chain does not falsely return a subthreshold node; insertion order and concurrent requests do not change outcomes; deleted/corrected facts cannot survive in cached graph results; scoped high-similarity foreign nodes cannot steal the seed; maintenance avoids all-pairs rebuilds.

Run: `py -3.13 -m pytest -q tests/test_spreading_activation.py tests/test_spreading_isolation.py tests/test_spreading_mutual_builder.py tests/test_storage_sqlite.py tests/test_engine.py`.

### T11 - Finish consolidation, retention, and forgetting

Depends on: T05-T10.

Files: `mnemos/modules/sleep.py`, `mnemos/memory_service.py`, `mnemos/utils/storage.py`, `mnemos/config.py`, new `tests/test_memory_lifecycle.py`.

Steps: extract source-backed semantic facts, merge exact repeats and proven equivalent assertions, and retain unresolved contradictions. Only mark evidence processed after accepted output or an explicit no-durable-facts decision. Prune eligible processed raw evidence on schedule while retaining fact support excerpts. Implement suppression-aware forget across all derived state.

Acceptance: retry cannot increase observation counts without a new source event; empty/malformed model output is not success; diverse sessions cannot collapse into the same canned facts; retention reduces live rows while preserving supported facts; forget survives replay, restart, and reconsolidation; failed evidence stays visible.

Run: `py -3.13 -m pytest -q tests/test_memory_lifecycle.py tests/test_memory_service.py tests/test_durable_jobs.py tests/test_maintenance.py`.

### T12 - Repair Claude Code capture and retrieval

Depends on: T04, T06, T09, T11.

Files: `mnemos/hook_autostore.py`, new `mnemos/adapters/claude_code.py`, `hooks/hooks.json`, `mnemos/hosts.py`, `.claude-plugin/plugin.json`, `tests/test_hook_autostore.py`, `tests/test_hosts.py`.

Steps: validate current host payloads; capture prompts and completed response evidence with stable source IDs; retrieve bounded context at start/prompt boundaries; enqueue maintenance at completion/compaction. Remove unconditional error suppression. Distinguish Claude Code configuration from Claude Desktop configuration. Preserve unrelated settings and provide a restorable config preview.

Acceptance: a real session records capture, later scoped recall, correction, and forget; interrupted and duplicate events are safe; Python/command quoting works on Windows; hook failure is visible without indefinitely blocking coding.

Run: `py -3.13 -m pytest -q tests/test_hook_autostore.py tests/test_hosts.py tests/test_plugin_packaging.py`, then the real-host script in T17.

### T13 - Add native Codex hooks

Depends on: T12's shared adapter contract.

Files: new `mnemos/adapters/codex.py`, `mnemos/hosts.py`, `mnemos/antigravity.py`, `skills/mnemos-codex/`, `docs/codex.md`, new `tests/test_codex_hooks.py`.

Steps: implement SessionStart/UserPromptSubmit recall, UserPromptSubmit/Stop capture, and compaction/end maintenance using the documented payload fields. Use prompt, last_assistant_message, session_id, and turn_id where available. Use command hooks for service startup, since an unavailable MCP server cannot start itself through an MCP tool hook. Treat Stop capture as best effort and return non-blocking valid JSON; never request a continuation. Do not rely on matchers for UserPromptSubmit or Stop. SessionEnd requests a final durable flush; PreCompact captures available evidence and PostCompact restores bounded relevant context. Crashes recover through previously acknowledged events, not an assumption that SessionEnd always runs. Do not depend on an undocumented transcript parser for basic operation. Negotiate installed host capabilities and retain an explicitly labeled MCP/instruction fallback for unsupported versions.

Acceptance: CLI and desktop are tested separately; stop hooks do not create continuation loops; manual MCP use and hook capture do not double-count; Windows command paths work; partial capture reports its limitation rather than claiming a full transcript.

Run: `py -3.13 -m pytest -q tests/test_codex_adapter.py tests/test_codex_capture_coverage.py tests/test_hosts.py tests/test_mcp_tier2_smoke.py`, then T17. Contract source: https://learn.chatgpt.com/docs/hooks.

### T14 - Add the Hermes memory provider

Depends on: T04, T06, T09, T11.

Files: new `mnemos/adapters/hermes/`, `pyproject.toml`, `mnemos/hosts.py`, new `docs/hermes.md`, new `tests/test_hermes_provider.py`.

Steps: publish the optional `hermes_agent.memory_providers` entry point; implement the MemoryProvider ABC, configuration, initialize, prefetch, sync_turn, session-end/shutdown, and explicit memory-write mirroring. Availability checks do not perform network I/O. Make sync_turn a bounded durable enqueue, with enrichment outside the host callback. Use session-provided cwd and hermes_home profile identity, not process cwd. Implement the optional v2 pre-compress checkpoint contract: report success only after sanitized evidence is durable in the service or local outbox; raise when a required checkpoint cannot be made. Do not automatically enable checkpoint-required mode or change Hermes compressor selection. Keep the module importable without Hermes installed through an optional integration boundary. Add the Windows/WSL bridge client and preview setup.

Acceptance: exact supported-version ABC/config contracts pass; profile/session scope does not follow unrelated process cwd; built-in memory remains intact; mirrored writes do not echo into repeated captures; service outage uses bounded local buffering; required checkpoint failure is truthful and replay-safe; real Hermes can recall a fact captured by each other host.

Run: `py -3.13 -m pytest -q tests/test_hermes_adapter.py tests/test_hermes_tools.py tests/test_hermes_transcript_identity.py tests/test_hermes_provisional_capture.py tests/test_service.py`, then T17. Contract source: https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin.

### T15 - Make onboarding and controls reflect actual operation

Depends on: T11-T14.

Files: `mnemos/control_plane.py`, `mnemos/ui_server.py`, `mnemos/ui/`, `mnemos/settings.py`, `tests/test_control_plane.py`, `tests/test_ui_server.py`.

Steps: expose project enable/pause, capture coverage, queue/retry state, fact evidence/revisions, correction/forget, provider locality, model download/setup status, and host-specific smoke results. Use one configured runtime across UI and hosts. Keep model selection explicit and require extraction of an input-specific test fact before marking cognition ready.

Acceptance: setup cannot silently choose mock cognition or send content to cloud; the user can determine what was captured, what became memory, and why; UI health agrees with CLI/MCP; unrelated host settings survive preview/apply and restoration.

Run: `py -3.13 -m pytest -q tests/test_control_plane.py tests/test_ui_server.py tests/test_health.py tests/test_hosts.py`; inspect the first-run and memory-management flows in a browser.

### T16 - Add backup, restore, and embedding migration

Depends on: T03, T11.

Files: new `mnemos/backup.py`, `mnemos/cli.py`, `mnemos/utils/storage.py`, `mnemos/runtime.py`, new `tests/test_backup_restore.py`.

Steps: implement consistent backups/restores, integrity checks, schema/version manifests, embedding model identity, and resumable reindex into a new index generation. Activate a new generation only when complete. Restore carries suppression records; warn that an old backup may predate later forget operations.

Acceptance: backup during writes restores to a valid committed snapshot; interrupted migration leaves the previous database usable; equal-dimension different embedding models are detected; restores do not silently merge conflicting project IDs or resurrect suppressed sources from the current registry.

Run: `py -3.13 -m pytest -q tests/test_backup_restore.py tests/test_storage_sqlite.py tests/test_runtime.py`.

### T17 - Verify the complete local and cross-host workflow

The [v36 Hermes lifecycle](../validation/2026-09-20/hermes-revision-lifecycle-v36/README.md)
passes six phases against the byte-verified 995d9c2 wheel: delivery, offline
buffering, restart replay, correction/deduplication, reviewed forgetting/replay,
and final restart without resurrection. All modules are disabled and embeddings
are simple. This replays saved native records through a temporary loopback
service; it does not prove a new interactive capture or full-module quality.

The [compaction growth fixture](../validation/2026-09-20/hermes-compaction-growth-v4/README.md)
passes 25 exact-origin checkpoints across repeated compaction, user editing and
reopen. Whole-session pruning removes its complete fixture graph. Physical
generations accumulate while the session stays open; no bounded storage policy
or upstream growth comparison is claimed. Earlier fixture projection and carrier
mistakes are preserved and explained separately from the corrected assertions.

The [compression/edit delivery check](../validation/2026-09-20/hermes-revision-delivery-v33/README.md)
preserves three identities across six capture phases and service restart:
original source, edited source, and provisional summary. The service creates
two extract jobs and one capture_only job; offline replay adds no duplicate
episode or job. This proves raw delivery from the isolated host candidate,
not extracted-fact correction or forgetting. Revision-snapshot retention bounds
and long-term storage growth remain unvalidated.

The [Hermes outbox delivery check](../validation/2026-09-20/hermes-outbox-delivery-v31/README.md)
delivers the exact v28 boundary capture through the current installed package's
authenticated loopback service into private SQLite. Offline replay stays queued;
restart acknowledges the same event without duplicating its episode or job.
This closes that bounded raw-delivery path, with all five modules disabled and
no background extraction. Failed scope-fixture and verifier attempts are retained.

The [current-package boundary probe](../validation/2026-09-20/hermes-current-package-boundary-v28/README.md)
passes actual CLI `/new` persistence and provider rotation with verified installed
product `05c71d2`: one valid outbox capture and zero captures for corrupted evidence.
It does not exercise service delivery or the full interactive CLI. The repaired
host's complete SessionDB suite records 1,427 passes, 11 skips, and the previously
observed baseline WAL failure. The full failure remains preserved.

The [provenance repair follow-up](../validation/2026-09-20/hermes-provenance-repair-v27/README.md)
reproduces and repairs unmarked compression output acquiring original authority,
edited hybrid components retaining stale ancestor references, and derived user
edits being promoted to originals. Its 32 focused cases cover classification,
restart, recopy, rollback, and tamper rejection. Parent originals remain intact;
hybrid revisions receive separate durable source records. The broad SessionDB
rerun and native delivery acceptance remain open at this checkpoint.

The [pre-compression source snapshot follow-up](../validation/2026-09-20/hermes-source-snapshot-v21/README.md)
passes 23 focused cases after persisting original evidence before compression,
advancing its lease watermark, and correcting carrier classification. One
fixture now uses its actual seeded transcript; a separate regression rejects
unrelated transcript claims. The complete SessionDB selection records 1,416
passes, 11 skips, and one WAL-test failure also seen in the earlier baseline.
Two further provenance risks require runtime reproduction. This remains an
isolated host candidate and does not close T17.

The [isolated Hermes boundary candidate](../validation/2026-09-20/hermes-boundary-projection-v15/README.md)
passes 96 of 99 combined cases. Row-ID and /new projection repairs pass their
focused checks; compression provenance and two legitimate evidence-update
paths remain failed at that checkpoint. These are isolated candidate results,
not a live host upgrade or complete capture acceptance.

The [native Claude embedding-outage v2 check](../validation/2026-09-20/native-claude-embedding-outage-v2/README.md)
observes one real tool retrieval during provider unavailability on verified
product `4bbab4f`. The scoped fact and explicit degraded status reach Claude;
the foreign control stays absent, seed state stays unchanged, and cleanup
passes. Vectors are synthetic fixtures and all cognition modules are disabled.
V1 remains failed on a verifier/MCP schema mismatch. Semantic quality,
extraction recovery, broader native resilience, Desktop, and pilot gates stay open.

The [Codex outage run](../validation/2026-09-20/native-codex-outage-v1/README.md)
adds one stopped-service raw-capture recovery on verified product `9722d92`.
Two native messages persist as two pending jobs after restart; raw identities,
receipt accounting, replay, a second restart, and cleanup pass. Extraction was
disabled, so provider/extraction recovery and broader acceptance remain open.

The [native text audit](../validation/2026-09-20/native-capture-coverage-v1/README.md) accounts for all 31 text-bearing messages in v7, including two suppression acknowledgements. It separately identifies eight repeated deliveries and four provisional prompts among 43 arrivals. This closes receipt accounting for that recorded lifecycle; the broader capture and failure-recovery gates remain open.

Latest native result: [three-host v7](../validation/2026-09-20/three-host-native-v7/README.md) passes all 11 phases on verified product `e6887de`: teaching, restarted recall, same-ID correction, all-host corrected recall, reviewed deletion, 12 suppressed source replays, and all-host final recall. Controls and cleanup pass. All five cognition modules were disabled; complete capture, Desktop, native outage recovery, full-module quality, and the pilot remain open.

Subsequent bounded outage result: [Claude v2](../validation/2026-09-20/native-claude-outage-v2/README.md)
observes two native messages buffered while the service is stopped, then
persisted after restart on verified product `a64afb7`. Replay, another restart,
physical source checks, and cleanup pass. Three pending jobs include a
provisional prompt, with zero facts. Provider/extraction recovery and broader
host reliability remain open. The earlier v1 default-store incident is preserved.

Previous native result: [three-host v5](../validation/2026-09-20/three-host-native-v5/README.md) passed teaching but failed Hermes recall on unsupported scope arguments. Generated shared guidance requests those exact arguments; all three provider requests contain that conflicting instruction. Worker startup and cleanup passed. Correction and later phases did not run. Full continuity remains open.

Previous native result: [three-host v4](../validation/2026-09-20/three-host-native-v4/README.md) passed Codex and Claude teaching, followed by Hermes recall of both facts after restart. Correction failed: rejected scope override, omitted inspection, and a pending correction job. The final fact remained at version 1; later phases did not run. Cleanup passed. Validation stopped before the runner's post-phase maintenance; pending status does not establish a production worker defect. V2 and v3 remain failed. T17 acceptance remains open.

Depends on: T12-T16.

Files: new `tests/test_cross_host_contract.py`, new `docs/validation/local-memory-hosts.md`, new `benchmarks/scenarios/local-memory-continuity.jsonl`.

Steps: automate host-independent capture/recall/correct/forget fixtures, then execute equivalent real sessions on the installed Codex CLI, Codex desktop, Claude Code, and Hermes versions. Exercise Windows plus WSL where applicable, cloud-blocked operation, model outage, process kill, concurrent sessions, and config rollback.

Acceptance: host A teaches a source-backed project convention; after restart hosts B/C retrieve it; B corrects it and all retrieve the current version; forgetting survives maintenance and replay. Project B never sees project A's canary. Record actual host versions and transcripts with secrets removed.

The [Hermes v10 cohort](../validation/2026-09-20/hermes-native-cohort-v10/README.md) now supplies one complete bounded lifecycle in three attempts, with both failures retained. All five cognition modules were disabled; general reliability remains open.

The [original three-host protocol](2026-09-19-three-host-continuity-protocol-v1.md) now has preserved successor attempts. V7 establishes one bounded teaching, recall, correction, forgetting, and replay lifecycle across all three hosts. Earlier failures remain preserved; broader acceptance gates remain open.

Run: `py -3.13 -m pytest -q tests/test_cross_host_contract.py tests/test_mcp_compatibility.py`. Real-host results remain NOT RUN until observed; protocol tests do not promote host tiers.

The [installed all-module outage v2](../validation/2026-09-20/all-module-provider-outage-v2/README.md)
preserves both native-source events and pending jobs across controlled provider
connection refusal, service restart and replay on product d937892. All flags
are enabled, but failure occurs at prediction before downstream modules run.
The first probe's incorrect one-attempt expectation remains preserved in v1.
A [real-provider continuation](../validation/2026-09-20/all-module-provider-recovery-preparation-v1/README.md)
was frozen from that database and has now [failed](../validation/2026-09-20/all-module-provider-recovery-terminal-v1/README.md).
The user-event extraction twice exhausted its token budget without a final
answer; the job exhausted its remaining retries and no facts were created.
The [response diagnosis](../validation/2026-09-20/all-module-provider-recovery-diagnosis-v1.json)
preserves the exact six provider outcomes. A separately frozen temperature-only
follow-up [also failed](../validation/2026-09-20/all-module-provider-recovery-temperature-one-terminal-v1/README.md)
on the same older installed build and original database. Both jobs completed,
but extraction retained a one-turn tool restriction as a lasting preference,
producing two facts instead of the required marker alone. Duration and scope
of extracted instructions need a repair and prospective regression cases;
no sampling or prompt change is adopted and recovery remains open.

Additional recovery gap: [active work can outlive its fixed lease](../validation/2026-09-20/long-job-lease-reproduction-v1/README.md).
A controlled-clock installed reproduction rejects successful completion at
301 seconds with no competing owner, then reclaims the same job. Fenced lease
renewal is now implemented with owner/token/expiry fencing, a non-accumulating
renewal deadline, and bounded processor cancellation. Fifteen focused tests pass,
including rejection of a late write from a processor that suppresses cancellation.
The [frozen source/package checkpoint](../validation/2026-09-20/lease-renewal-source-checkpoint-v1/README.md)
passes 2,019 tests with four skips and all six checks. The [expanded installed check](../validation/2026-09-20/lease-renewal-installed-failure-v1/README.md)
failed one MCP startup case (832 passed, four skipped). The recorded registry
and a controlled reproduction expose identity drift when Git discovery fails:
a repository must not silently become a different filesystem-based identity.
The resolver repair retries once then fails closed for repository paths;
the 30-case focused selection includes scope tests and all three MCP roundtrips.
Inherited Git path overrides can also select a foreign repository; the repair
filters those overrides and tests both GIT_DIR and GIT_COMMON_DIR. Ordinary
folders, bare repositories and dangling Git markers have explicit coverage. A new
full source/package and installed checkpoint is required. Recovery and quality
acceptance remain open.

### T18 - Prove the contribution of all five modules

The [precision proxy labels](../validation/2026-09-20/precision-proxy-labels-v1/README.md)
make the exact-text measurement explicit in both precision aliases and both
summary populations. Semantic factual precision stays NOT MEASURED. Forty-nine
focused and adjacent tests pass; no scores, thresholds or historical results
change, and no new semantic quality or full-suite result is claimed.

The [alternative seed structural audit](../validation/2026-09-20/alternative-graph-seed-audit-v1/README.md)
checks the saved five-case graph development traces. Top-three and top-five seed
sets add no structural target coverage over the chosen single seed. Top-ten
adds coverage in one prefixed case; this is an optimistic path bound, not an
executed multi-seed retrieval or recall improvement. Scope policy, path ranks
and independent reachability checks pass. Original graph quality failures remain.

T17 follow-up: the [revision lifecycle v34](../validation/2026-09-20/hermes-revision-lifecycle-v34/README.md)
passes raw delivery and restart replay, then fails explicit correction when
the destination text already exists as captured evidence. The durable job
remains pending with RuntimeError; the original target stays unchanged.
The expected-version canonical-collision contract requires review before a
repair. Forgetting and final replay did not run. All modules were disabled;
the preserved failure is structural, not a semantic extraction measurement.

The subsequent [captured correction merge](../validation/2026-09-20/captured-correction-merge.md)
repairs the unverified captured-destination case in source, preserving the
target ID and transferred evidence through correction, rollback, restart and
reviewed forgetting. The expanded suite passed 74 tests; 13 focused tests pass
after the final retired-ID and pending-job guards. Full-suite and rebuilt-wheel
Hermes verification are now recorded by the [installed v35 lifecycle](../validation/2026-09-20/hermes-revision-lifecycle-v35/README.md):
all six phases pass with all five modules disabled. Source 26c9ddb passes 2,012
tests with four skips and all six source/package checks. The [broader installed
contract suite](../validation/2026-09-20/captured-merge-installed-checkpoint-v2/README.md)
passes 811 cases with four matching skips. Asserted-destination collisions remain rejected.

The [prompt-blinded semantic review](../validation/2026-09-20/extraction-authored-semantic-review-v1/README.md)
finds better qualification retention in the candidate's successful outputs,
but its runtime failure remains. The [thinking-disabled diagnostic](../validation/2026-09-20/extraction-thinking-off-diagnostic-v1/README.md)
completes 40 calls without runtime errors while losing required facts and
attaching unrelated source events to the existing control fact. That setting
is not adopted. The [12,288-token comparison](../validation/2026-09-20/extraction-larger-budget-v1/README.md)
has completed all 40 calls: production 20 runtime-valid; candidate 19 valid
and one empty final response after exhausting the larger budget. All parsed
statuses and ordered proposals equal the original 8,192-token run. The larger
budget is not adopted; other inputs and original criteria stayed fixed.

The [terminal authored comparison](../validation/2026-09-20/extraction-authored-holdout-terminal-v1/README.md)
records all 40 calls. Production has 20 runtime-valid responses; the history-only
candidate has 19 and one empty final response. Every call and the original
pre-execution rubric ambiguities are preserved. Semantic review is separate;
runtime validity does not establish correctness or close T18.

The [Qwen3.5 service follow-up](../validation/2026-09-20/historical-claim-service-v3-verified/README.md)
passes all seven bounded development cases on source `05c71d2`, including three
explicit correction/restart/forget/replay lifecycles and four abstention controls.
Seven SQLite backups pass integrity and foreign-key checks. This is one
repetition of previously observed cases with only sleep extraction enabled.
The [production-prompt comparison](../validation/2026-09-20/history-production-prompt-v1/README.md)
records 36 runtime-valid responses but retains semantic failures. The separate
[20-case comparison preparation](../validation/2026-09-20/extraction-authored-holdout-preparation-v1/README.md)
freezes both prompt arms and records rubric ambiguities before execution.
Its execution is now complete in the terminal record above; no prompt adoption
or T18 acceptance is claimed.

Subsequent [historical-claim service development](../validation/2026-09-20/historical-claim-service-v1/README.md)
completes three positive correction/restart/retrieval/forget/replay lifecycles
with real Qwen2.5 and Nomic, sleep extraction enabled, and the other modules
disabled. Three of four zero-fact controls fail by retaining qualified
fiction, hypothetical, and assistant-suggestion memories. This candidate-only
development result does not close T18 or justify adopting the prompt.

The [frozen fixture audit](../validation/2026-09-20/retrieval-fixture-audit-v1/README.md) identifies scope-label and explicit-body conflicts in the original retrieval evaluation. Preserve the original negatives. The [v4 repair](../validation/2026-09-20/contract-v4-preparation-v1/README.md) is now frozen on `3d09658`, with unchanged minimums, scoring, and gates. Its [terminal seven-profile matrix](../validation/2026-09-20/contract-v4-terminal-v1/README.md) contains 4,150 complete and 50 partial observations out of 4,200 and does not pass acceptance. This repaired synthetic input must not be described as unseen heldout data. The [missing-target diagnosis](../validation/2026-09-20/contract-v4-missing-target-diagnosis-v1.json)
binds all 50 partial observations to 11 scenarios: automatic historical-note
extraction abstains before an explicit correction targets that source. This
locates the missing target upstream of correction; no accepted target existed
to mutate. Preserve the failure and criteria rather than inventing a target.
The [durability comparison](../validation/2026-09-20/extraction-durability-development-terminal-v1/README.md)
completed all 52 calls without runtime errors. Its [semantic review](../validation/2026-09-20/extraction-durability-semantic-review-v1.md)
rejects adoption: fewer transient instructions were retained, but historical
facts, independent markers and policy qualifications were lost. Production
also retains unwanted temporary instructions. The [v3 follow-up](../validation/2026-09-20/extraction-durability-history-terminal-v3/README.md)
completed 92 calls across two arms on 46 reused development cases, including
the previous history set. Production had 45 valid outputs and one error; the
candidate had 46 valid outputs. Runtime validity does not prove extraction quality. V2 was prepared
but never executed. No prompt is adopted; this comparison cannot close T18 alone.

Depends on: T07-T11, T17.

Files: `mnemos/benchmark.py`, new `benchmarks/ablation.py`, new `benchmarks/datasets/local-memory-heldout-v1.jsonl`, new `tests/test_ablation_contract.py`, `docs/retrieval-benchmarks.md`.

Steps: preserve deterministic mechanism fixtures, add real local-model runs, and compare scope-aware vector, lexical/hybrid, full pipeline, and one-module-disabled variants on identical data and budgets. Freeze labels and protocol before threshold tuning: at least 200 held-out session scenarios, 100 critical facts, 100 answerable queries, 100 no-answer queries, and 50 matched semantic/state query pairs, spanning all three hosts and multiple projects. Keep a disjoint development set for tuning. Run at least three real-model repetitions and report Wilson intervals for proportions plus paired bootstrap intervals for profile differences. Gates below apply to point estimates; intervals and sample counts must accompany them. Log model revisions, embeddings, dataset hashes, feature flags, hardware, warm/cold timings, and raw outputs. Measure extraction precision, critical-fact retention, retrieval accuracy, stale-fact rate, no-answer accuracy, token cost, actual row/byte growth, and latency. The existing harness fixes MockLLM but can honor an embedding configuration despite its simple default; label actual provider selection rather than assuming it.

Acceptance: each module can be activated independently, including at top-1; the full path is actually executed; compression is reported with retained-information accuracy; surprise filtering is compared with simple length/noise filtering; affective tests hold semantic content constant; graph tests require relevant associative recall, not merely more nodes. Report neutral/negative results without rewriting the dataset to obtain a desired lift.

Run: `py -3.13 -m pytest -q tests/test_local_memory_benchmark.py tests/test_retrieval_comparison.py tests/test_retrieval_comparison_run.py tests/test_benchmark.py`, followed by versioned baseline/full/ablation runs that generate JSON artifacts. Add `--dataset`, `--profile`, `--disable-module`, `--provider`, `--repetitions`, and `--output` to the new runner.

The [temperature-only diagnostic preparation](../validation/2026-09-20/extraction-temperature-one-preparation-v1/README.md)
freezes a separate 40-call development comparison before execution. It retains
the original prompt arms and changes only temperature from zero to 1.0. This
is not a full recommended-sampling configuration or unseen validation.
The [terminal comparison](../validation/2026-09-20/extraction-temperature-one-terminal-v1/README.md)
completed all 40 calls with zero runtime errors. Twelve parsed outputs changed
from the temperature-zero baseline; that count is not an accuracy measurement.
The positive-case review is byte-matched. The [completed negative review](../validation/2026-09-20/extraction-temperature-negative-review-v1.md) preserves candidate N07 unwanted retention and the preregistered N10 ambiguity. No prompt or sampling change is adopted; unseen validation remains open.
The [fixed-candidate graph-weight bound](../validation/2026-09-20/graph-weight-bound-audit-v1/README.md)
finds only three of 30 recorded combinations could reach top five under its
optimistic assumptions; 15 lack the target candidate altogether. This is a
structural diagnostic, not retrieval improvement or a changed graph default.

The completed [v3 durability subset](../validation/2026-09-20/extraction-durability-history-v3-durability-subset/README.md)
preserves all 52 calls across 26 development cases. The candidate drops the
required marker in the mixed ongoing-policy case; production loses it in the
temporary-instruction variant. No adoption follows from these partial results.
The history portion completed under the unchanged frozen protocol. The
[terminal record](../validation/2026-09-20/extraction-durability-history-terminal-v3/README.md)
preserves 92 calls with one production error. The [full semantic review](../validation/2026-09-20/extraction-durability-history-semantic-review-v3.md)
rejects candidate adoption: required marker and open-consideration omissions
and unwanted fictional retention remain. The [14B comparison](../validation/2026-09-20/extraction-model-comparison-14b-terminal-v1/README.md)
completed all 92 calls without runtime errors. Its [semantic review](../validation/2026-09-20/extraction-model-comparison-14b-semantic-review-v1/README.md)
rejects both prompts: marker loss, unsupported user-preference inference and
unwanted retention persist. Candidate N08 also cites an unrelated source as
support for an existing control fact. No fresh corpus calls have been made.

The [fresh v2 corpus review](../validation/2026-09-20/fresh-corpus-semantic-review-v2/README.md)
records ineffective corrections, incorrect old-source links, neutral state
paraphrases and singleton retrieval pools. New drafts are being revised before
inference; none is frozen or accepted. Zero exact overlap does not establish
semantic independence.

The [v7 preparation](../validation/2026-09-20/fresh-corpus-preparation-v7/README.md)
repairs question semantics, source links and project-name consistency. Its new
v6 contract supports exact original-source replay after forget and restart,
rather than treating a post-restart question as replay. It has 200 heldout
sessions, 700 critical facts, 200 answerable queries, 220 no-answer queries and
80 matched state pairs. One hundred ranking pools have six current facts.
Repeated families and short sessions limit generalization; graph benefit is
not established. No evaluated-model calls have been made on this corpus.
Source c1e2217 passes 22 replay tests and 78 combined focused checks; full
source/package verification passes all six checks with 2,081 tests and four skips
in the [replay checkpoint](../validation/2026-09-20/evaluation-replay-v6-source-checkpoint-v1/README.md). Original model outcomes
and acceptance thresholds remain unchanged.

The [extraction evidence scoring repair](../validation/2026-09-20/extraction-evidence-resolution-v1/README.md)
passes 113 focused and adjacent tests, Black and compatibility mypy on e082790.
Every claimed source must now resolve before the exact-text proxy accepts a
fact. Earlier scores remain bound to their original code; this is not semantic
precision or a new full-suite claim. Packaged engine code is unchanged.

The [source-only assertion inventory](../validation/2026-09-20/assertion-inventory-preparation-v1/README.md)
is frozen at candidate 60f79ca and completed 46 reused development cases
on local Qwen2.5:14b with no runtime errors. The [terminal record](../validation/2026-09-20/assertion-inventory-terminal-v1/README.md)
and [semantic review](../validation/2026-09-20/assertion-inventory-semantic-review-v1/README.md)
preserve improved marker retention and remaining eligibility/qualification
failures. The candidate is not adopted. Sixteen boundary tests, formatting, compatibility mypy
and all 46 offline prompt checks pass. It withholds existing facts during
extraction, requires exact source quotations and distinguishes retained kinds
from temporary instructions and suggestions. It cannot yet reconcile targets. The separate [scope repair](../validation/2026-09-20/assertion-inventory-scope-review-v1.md)
passes 29 tests and all 46 frozen source-identity checks; original run bytes
remain unchanged.
Literal quote/ID validity is not semantic support; no production adoption or
fresh held-out inference is claimed. The separate launcher configuration change
is validated by [75 focused checks](../validation/2026-09-20/heldout-generation-settings-v1.md).

The [qualified inventory v2 preparation](../validation/2026-09-20/assertion-inventory-preparation-v2/README.md)
freezes the next 46-case diagnostic at 1975f5e. It completed on the same
local model/settings, with qualifications attached to their assertions and
source-bound retention-request quotations for unresolved ideas. Forty-one
combined tests, formatting, compatibility mypy and all 46 offline source checks
pass. Missing/wrong external preparation digests fail before execution. This
remains an unadopted extraction experiment. The [terminal record](../validation/2026-09-20/assertion-inventory-terminal-v2/README.md)
preserves 41 runtime-valid outcomes and five errors. The [semantic review](../validation/2026-09-20/assertion-inventory-semantic-review-v2/README.md)
rejects adoption: one continuing policy is lost, transient instructions leak
into an attached qualification, and unrequested speculation is retained.
All five errors are contract rejections, not provider outages. Literal quotations do not establish
semantic support or request intent. The [independent v1 review](../validation/2026-09-20/assertion-inventory-independent-review-v1.md)
preserves additional inventory-completeness and qualification findings.

The separate [v3 contract repair and offline review](../validation/2026-09-20/qualified-inventory-v3-review-v1.md)
passes 51 candidate tests and revalidates all 46 original response texts without
new inference. All 41 previously valid outputs remain unchanged; four former
errors pass the revised contract and one still fails exact quotation checking.
Original v2 results and semantic criteria remain unchanged. Extraction quality,
qualification attachment and target reconciliation remain open.

The [Qwen3.5:9b v3 preparation](../validation/2026-09-20/assertion-inventory-preparation-v3/README.md)
freezes the same 46 development cases, exact model digest and source hashes,
thinking enabled, temperature 1.0 and 8192 output tokens. All offline checks
pass. The [terminal review](../validation/2026-09-20/assertion-inventory-semantic-review-v3/README.md) covers all 46 cases: 31 runtime-valid outputs, 13 empty-final budget failures and two source-role rejections. Unnecessary fixture retention and qualification omissions remain; the candidate is unadopted. Its model, thinking mode and prompt differ from
the 14B experiment; no single-component causal claim or fresh held-out exposure
is implied. No production default changed.

The [thinking-disabled follow-up preparation](../validation/2026-09-20/assertion-inventory-preparation-v4/README.md)
freezes the same 46 cases, v3 candidate, model and generation inputs except
think=false. All offline paired prompts and schemas match. The launcher rejects
missing/wrong digests and an unfinished parent before generation. It also binds
the parent's terminal observations. This changes the provider route as well as
thinking configuration, so no isolated reasoning claim is planned. The follow-up
completed all 46 cases: 40 runtime-valid inventories and six validation rejections, with no empty-final failures. The [paired review](../validation/2026-09-20/assertion-inventory-comparison-v3-v4/README.md) preserves all results. Retention-intent errors, unsupported paraphrases and qualification loss remain; neither configuration is adopted. Original results, rubrics and production defaults are unchanged.

The [source-grounded review v5 preparation](../validation/2026-09-20/assertion-review-preparation-v5/README.md)
adds a separately recorded model pass over every frozen v4 draft, including all
six rejected drafts. Its new candidate passes 70 focused tests, formatting and
Python 3.10 type checks; all 46 offline bindings and both digest guards pass.
The [completed review](../validation/2026-09-20/assertion-review-semantic-v5/README.md)
covers all 46 cases: 39 runtime-valid outputs and seven validation rejections.
All six original rejections persist and one valid draft becomes a rejection.
P01 commentary and one missing approval boundary improve, but P06 permission
inflation, N02 unwanted retention and qualification defects remain. The extra
46 review responses generated 13,813 output tokens, in addition to draft cost.
The reviewer remains unadopted; no production default or earlier outcome changed.

The [extractive selection v6 review](../validation/2026-09-20/source-selection-semantic-v6/README.md)
covers all 46 unchanged cases with no runtime/validation errors. Every one of
its 23 rendered excerpts independently verifies as an exact original source
slice with matching identity, scope, role, time and content hash. The candidate
passes 32 boundary tests and 102 combined extraction tests. It prevents invented
memory wording but still omits continuing policies, corrections, a mixed-message
marker and a manuscript requirement, while retaining some temporary/inspection
material. It remains unadopted; runtime validity is not retention accuracy.

The [unchanged-selector Qwen2.5:14b comparison](../validation/2026-09-20/source-selection-semantic-v7/README.md)
completes all 46 cases: 44 runtime-valid, two non-user authority rejections and
35 exact source excerpts with verified provenance. It recovers several omitted
policies, corrections, the one-turn marker and manuscript requirement, but retains
an options question, fictional/hypothetical material and inspection fixtures.
It also loses the independent marker in the continuing-policy case. Both model
configurations remain unadopted; no criteria or previous outcomes changed.
The [full isolated source/package check on 6e8456b](../validation/2026-09-20/extractive-source-checkpoint-v1/README.md)
passes 2,209 tests with four skips and all six checks. Wheel bytes match the
previous installed checkpoint; experimental selectors remain source-only.

The [temperature-zero control v8](../validation/2026-09-20/source-selection-semantic-v8/README.md)
finishes 46 cases with 44 valid outputs, two authority rejections and 34 verified
source excerpts. Forty-four outcomes match v7 exactly. One inspection fixture
is omitted and one historical report gains a correct retention-request reference.
The remaining unwanted retention and marker omission persist. No candidate is
adopted; this is reused-development evidence, not fresh acceptance.

Future integration must preserve the current `SleepDaemon.propose_facts` ->
`LocalMemoryService.process_job` -> transactional `commit_fact_and_job` path.
Source-only excerpts cannot directly become support/correction operations.
Reconciliation still needs the scoped target snapshot, explicit user correction
evidence and expected-version checks, including changes while embedding awaits.
No experimental selector has been connected to production writes.

The [explicit-discard classification v9 preparation](../validation/2026-09-20/source-classification-preparation-v9/README.md)
freezes all 46 original cases and rubrics before execution. The experimental
classifier adds question, temporary-instruction, quoted-material, unrequested-
speculation and unaccepted-suggestion labels; these remain auditable but emit
no memory. Source, scope, authority and mutation guards still apply. The focused
selector/classifier suite passes 57 tests, with formatting and Python 3.10 type
checking passing. The [completed local Qwen2.5:14b comparison](../validation/2026-09-20/source-classification-semantic-v9/README.md)
has 45 valid outputs, one authority rejection and 29 verified original-source
excerpts. It discards the options question, fiction, hypothetical report and date
fixture retained in v8. The mixed-policy marker omission persists, and several
requirements are incorrectly cited as retention requests. Prompt and schema
change together; no fresh quality acceptance or production adoption is claimed.

The [source-accounting v10 preparation](../validation/2026-09-20/source-coverage-preparation-v10/README.md)
freezes the same 46 cases, rubrics, model settings and classification schema.
An updated prompt and post-validation require every nonblank source unit to be
covered by a retained or discarded range. This exposes silent omission without
certifying the assigned labels. All 64 focused tests pass, including a covered
but wrongly discarded example; formatting and Python 3.10 type checking pass.
The [completed comparison](../validation/2026-09-20/source-coverage-semantic-v10/README.md)
has 42 valid outputs, four rejections and 32 verified source excerpts. It exposes
the omitted marker as an error without recovering it; other regressions discard
the benchmark-date boundary, detach approval qualifications and retain a
temporary tool instruction. This candidate is not adopted. Complete coverage
does not replace the original semantic criteria.

The [Gemma comparison v11](../validation/2026-09-21/source-classification-preparation-v11/README.md)
changes only the requested model from v9 to the installed Gemma3:12B digest.
All 46 paired prompts and schemas match; cases, rubrics, classifier bytes and
generation settings remain fixed. The [completed comparison](../validation/2026-09-21/source-classification-semantic-v11/README.md)
has 43 valid outputs, three rejections and 31 source-verified excerpts. Some
request-reference errors disappear, but temporary instructions and fixtures are
retained and mixed-message/range failures remain. Candidate not adopted. A full
isolated [source/package checkpoint on f5c58f1](../validation/2026-09-21/classification-source-checkpoint-v1/README.md)
passes 2,241 tests with four skips and all six checks, including the newer
classifier and coverage contracts. The wheel remains byte-identical to the
previous installed checkpoint; no model quality or native acceptance follows.

The [Gemma 27B comparison v12](../validation/2026-09-21/source-classification-preparation-v12/README.md)
uses the same v9 classifier, prompt, schema, settings and 46 original cases.
Only the requested model changes to the installed digest bound in preparation.
All offline paired checks pass. The [completed comparison](../validation/2026-09-21/source-classification-semantic-v12/README.md)
has 43 valid outputs, three failures and 34 source-verified excerpts. It recovers
the mixed-policy marker but duplicates policy meaning and retains temporary
formatting and test instructions. The first case exhausted provider retries;
that failure remains in the result. The candidate is not adopted. The loading log reports
45 of 63 layers offloaded to the GPU, so this is not an all-GPU configuration
or an isolated latency benchmark. No fresh quality acceptance is claimed.

The experimental `source_role_binding.py` wrapper binds each source ID to its
allowed memory kinds in the structured decoding schema. Non-user sources can
produce attributed history or discard classifications; runtime authority and
mutation guards remain active even when a provider ignores the schema. The
prompt and rendered output are compared directly with the original classifier.
The combined selector/classifier/role-binding suite passes 66 tests. This is a
mechanical constraint, not evidence of correct durability judgments. The
[v13 preparation](../validation/2026-09-21/source-role-binding-preparation-v13/README.md)
freezes the same 46 v9 cases, model settings and rubrics. Offline paired calls
confirm unchanged prompts and source-specific schema branches; two launch
guards reject missing or incorrect preparation digests. The [completed comparison](../validation/2026-09-21/source-role-binding-semantic-v13/README.md)
has 46 runtime-valid outputs and 30 source-verified excerpts. Forty-five outcomes
match v9 exactly; P07 now retains the qualified assistant report as attributed
history instead of failing authority validation. The earlier semantic mistakes
remain, including marker omission and false retention-request references.
Production integration and fresh quality acceptance remain unexecuted.

The experimental `source_reconciliation.py` separates candidate selection from
target decisions without writing facts. It reconstructs source candidates through
the existing renderer, binds targets to scoped snapshots and observed versions,
accounts for every candidate and rejects input mutation after the provider call.
The combined selector/classifier/role-binding/reconciliation suite passes 84
tests, formatting and Python 3.10 module type checking. No model comparison or
production integration has run for reconciliation. Its conservative handling of
`correction_or_rejection` can leave a standalone new rejection unresolved when
there is no target. That distinction must be resolved before production adoption;
an unresolved correction cannot silently become a new fact or a no-facts success.

The [v14 instructional-example comparison](../validation/2026-09-21/source-instruction-examples-preparation-v14/README.md)
changes only the v13 prompt by adding three worked examples. All 46 paired
offline calls preserve the complete role-bound schema and exact real source
payload. The combined experimental suite passes 89 tests; formatting and Python
3.10 module type checking pass. The prepared source is 179b189. The
[completed comparison](../validation/2026-09-21/source-instruction-examples-semantic-v14/README.md)
has 46 runtime-valid outputs and 31 source-verified excerpts. Thirty-three
outcomes match v13; the marker omission is repaired, but overlapping policy
meaning and false uncertainty/request metadata remain. A standing policy also
gains an incorrect uncertainty label. Candidate not adopted. These examples
target known development failure classes; this is not fresh acceptance.

### T19 - Reconcile release, documentation, and research claims

Production extraction now snapshots source events before constructing the prompt
and rejects a response if those events change while the provider is awaited.
Five regression cases reproduced the missing guard, including an assistant
source changed to user that previously authorized a correction. The repaired
path passes 62 focused extraction, lifecycle, correction and service tests,
formatting and Python 3.10 type checking. The [full source/package checkpoint](../validation/2026-09-21/source-evidence-snapshot-checkpoint-v1/README.md)
on 56687e0 passes 2,255 tests with four skips and all six checks. Its changed wheel
passes the [isolated installed check](../validation/2026-09-21/source-evidence-snapshot-installed-v1/README.md):
898 passed cases with four matching skips, installed origins, wheel/source hashes
and the replay probe. This includes all five source-mutation regressions; it is
not a new native-host or real-model quality result.

Recovery follow-up: the [preserved failure diagnosis](../validation/2026-09-20/all-module-provider-recovery-diagnosis-v1/README.md)
identifies two empty structured final responses at the 8192-token limit. The
[provider-only control](../validation/2026-09-21/recovery-extraction-no-thinking-preparation-v1/README.md)
reconstructs both saved-event requests from the original d937892 wheel and
freezes them before running with think=False. No store is opened. This also
changes chat to generate; it is not an isolated reasoning experiment. The
control [completed with two valid empty proposals](../validation/2026-09-21/recovery-extraction-no-thinking-terminal-v1/README.md).
The user marker was omitted. The [latest-wheel lifecycle follow-up](../validation/2026-09-21/all-module-provider-recovery-terminal-v2/README.md)
completed both pending jobs on attempt four but produced zero facts, failing the
unchanged one-marker assertion. Later scope/correction/restart/forget phases did
not run. Service cleanup passed. Neither result closes full-module recovery.

The [source distribution repair](../validation/2026-09-20/sdist-payload-exclusion-v1/README.md)
reduces the observed source archive from 399.8 MB to 40.7 MB by excluding bulky
validation payloads while preserving them in the repository. All required
archive-backed fixtures retain their bytes, and 50 affected tests pass from the
extracted distribution. Its rebuilt wheel is byte-identical to the verified
wheel below. This packaging check on 232d0da is separate from the full source
suite on a128ef7; no new model result or release publication is implied.

Historical fully verified source [a128ef7](../validation/2026-09-20/precision-proxy-source-checkpoint-v1/README.md)
passes 2,158 tests with four skips and all six source/package checks. It includes
the support-version, standalone graph-budget and benchmark precision-label repairs.
Its rebuilt wheel is byte-identical to 995d9c2. Those bytes passed
the [installed check](../validation/2026-09-20/spreading-budget-installed-checkpoint-v1/README.md):
893 selected cases with four matching skips and the replay probe. It includes
lease renewal, cancellation fencing, project identity and inherited Git
environment isolation. The previous installed failure is preserved alongside
its repair. Earlier verified d937892 and Hermes v35 evidence remain bound to
their original checkpoints; Hermes v35 uses product 26c9ddb.
Native host integration, full-module quality and final acceptance remain open.
The [larger-budget extraction diagnostic](../validation/2026-09-20/extraction-larger-budget-v1/README.md)
preserves the same parsed outputs and failure as the original comparison;
increasing the budget did not resolve the observed extraction defect.

Depends on: T17, T18.

Files: `README.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `docs/client-compatibility-matrix.md`, `docs/release-checklist.md`, `docs/retrieval-benchmarks.md`, `docs/handoffs/substack-local-memory.md`, `tests/test_docs_consistency.py`; website changes in a separately reviewed checkout.

Steps: build and test the actual wheel from one commit; update runnable examples, support tiers, optional dependencies, and measured limitations. Produce a dated claim/evidence ledger and a paper errata/revision draft. Locate the actual paper publishing source before editing it; do not recreate it in an unrelated repo. Verify competitor comparisons with pinned primary evidence or remove absolute statements. Preserve historical paper versions and attach dated corrections.

Acceptance: install instructions match the release artifact; no synthetic result is presented as host or production proof; only completed pilot results close readiness gates; all published numbers have reproducible artifacts. Website/PyPI/paper publication is a distinct, explicitly authorized step.

Additional deliverable requested September 19: finalize a research and writing handoff for a roughly 2,500-word Substack article in Anthony's voice. The [handoff](../handoffs/substack-local-memory.md) must identify the final verified checkpoint, original promises, concrete repairs, measured results and failures, source locations, unresolved claims, and a suggested narrative. Apply Anthony's requested `writing-in-anthonys-voice` skill. The handoff is complete only after its evidence and open items are reconciled with the final validation state; the article itself and publication are separate work.

The latest working export is `mnemos-substack-research-packet-2026-09-20-v29.zip`
in the workspace's `deliverables` directory. Its [verification record](../validation/2026-09-20/substack-packet-v29/README.md)
binds source snapshot 60d572c, all 6,593 manifest-indexed files and all 140 local
handoff links. It includes the voice skill, current product checks, Hermes v35,
rejected extraction experiments and graph diagnostics. The temperature-only
run is represented by preparation, not a terminal result. Refresh at final
acceptance; this working packet does not close engineering or research gates.

Run final repository checks in an isolated environment: `python -m pytest -q`, `python -m black --check .`, `python -m mypy .`, `python -m build`, `python -m twine check dist/*`. Run the built wheel outside the source tree so imports cannot accidentally validate the checkout.

### T20 - Optional compatibility and procedural research extensions

Depends on: T19; does not block the SQLite local-memory release.

Files: new `mnemos/utils/qdrant_storage.py`, `pyproject.toml`, provider-specific tests; `mnemos/modules/sleep.py` and separate procedural-artifact tests.

Steps: if completing the historical backend surface, provide an optional Qdrant vector adapter while SQLite remains the authoritative fact/evidence/job ledger. Use versioned, idempotent index updates and reject stale index versions on read. This is an explicit backend-contract revision, not a claim of standalone Qdrant durability. Do not reintroduce it as a default service dependency. Retain proceduralization as generated, inspectable code artifacts with provenance; parsing or compilation is validation, not permission to execute.

Acceptance: optional installs leave SQLite unchanged; Qdrant contract tests cover persistence/isolation/forget; procedural artifacts are never executed during generation, retrieval, or maintenance. Do not call a backend infinitely scalable or declare generated tools safe from syntax checks alone.

## 5. Execution batches and release gates

Sequential batches:

1. Correctness foundation: T01-T03 and T06. Stop if any reproduced data-integrity/scope failure remains.
2. Persistent operation: T04-T05; use the backup prerequisite established in T03 for any schema upgrade.
3. Full memory behavior: T07, T08, T10, T09, then T11. Integrate scoring, revisions, and graph state before host expansion.
4. Integrations and controls: T12-T15, then complete T16-T17.
5. Measured release: T18-T19. T20 is separate optional work.

Only T13 and T14 may run in parallel after the shared adapter contract is stable, and only with separate file ownership. Shared host/type/config edits are integrated serially. Review every batch before moving on; inspect failures rather than weakening acceptance tests.

Initial acceptance targets, to be frozen before held-out evaluation:

- Zero scope canary leaks, fabricated default facts, lost acknowledged events, or forgotten-fact resurrection in deterministic adversarial and crash tests.
- At least 95% supported-fact precision and 95% retention of labeled critical facts on the held-out local-memory scenarios; report sample counts and uncertainty.
- At least 90% Recall@5 for answerable held-out queries and 90% correct abstention for no-answer queries. Compare all profiles under the same scope and context budget.
- The full profile must not reduce overall retrieval accuracy by more than two percentage points against the baseline. Enable it by default only when it also produces a reproducible benefit in its targeted scenarios within the latency budget; otherwise ship it as evaluated opt-in functionality with the measured result disclosed.
- On the recorded target machine, warm recall p95 <= 500 ms at 10,000 accepted facts and hook durable-capture p95 <= 100 ms. Report cold start/model load separately. If the selected local model cannot meet the recall budget, classify state asynchronously or accept explicitly supplied state; do not silently send queries to cloud.
- Run two weeks of daily use on at least two projects. The [prepared pilot protocol](2026-09-20-local-memory-daily-use-pilot.md) and empty record template are NOT STARTED; preparation does not satisfy this gate. External/general-availability readiness additionally requires the documented external pilot evidence; a single-user internal trial does not satisfy it.

All targets above are proposed engineering gates, not claims of current performance. A failing gate produces a documented limitation and repair task, never an automatic threshold adjustment.

First execution instruction: implement T01 only in an isolated implementation checkout, prove its focused tests, and review the patch before starting T02. Preserve live configuration and databases until backup/migration tasks are independently validated.
