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

Run: `py -3.13 -m pytest -q tests/test_durable_memory_repair.py tests/test_job_recovery.py tests/test_storage_sqlite.py`.

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

Run: `py -3.13 -m pytest -q tests/test_maintenance.py tests/test_job_recovery.py` with a controlled clock, not real sleeps.

### T06 - Enforce evidence and safety at supported write boundaries

Depends on: T03; integrate service routes after T04.

Files: new `mnemos/memory_service.py`, `mnemos/memory_safety.py`, `mnemos/curation.py`, `mnemos/modules/surprisal.py`, `mnemos/modules/mutable_rag.py`, `mnemos/modules/sleep.py`, `tests/test_memory_safety.py`, new `tests/test_fact_evidence.py`.

Steps: centralize sanitized writes and structured fact proposals; require source references for extracted claims; preserve user/assistant/tool provenance; reject unsupported evidence IDs and cross-scope references. Every public write-capable module constructs the default firewall when none is supplied, so standalone module use remains supported and safe under the configured policy. Raw storage is a documented low-level primitive, not a safe ingestion API. Policy tests establish coverage of configured detectors, not a claim that regex can detect every possible secret or personal detail.

Acceptance: safety runs before disk, outbox, logs, or cloud requests; extraction cannot cite absent sources; assistant speculation is labeled; prompt-injection content remains data and cannot authorize global writes or change runtime settings.

Run: `py -3.13 -m pytest -q tests/test_memory_safety.py tests/test_fact_evidence.py tests/test_modules.py`.

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

Depends on: T02, T03; integrate revisions from T08.

Files: `mnemos/modules/spreading.py`, `mnemos/utils/storage.py`, `mnemos/engine.py`, new `tests/test_spreading_contract.py`.

Steps: select seeds and traverse the permitted subgraph: current project, explicitly permitted member workspace, and explicitly shared global facts within the same user profile. Exclude foreign project nodes even as intermediate hops. Make energy state query-local. Define algorithm v2 as maximum propagated path energy, `next=current*(1-decay)*edge`, with threshold applied consistently, max-hop and max-visited budgets, and no cycle amplification. Persist graph schema, edge-builder, and traversal versions. On incompatibility, rebuild derived edges in a new generation while baseline recall remains available; never silently reinterpret incompatible persisted state. Maintain nearest eligible neighbors incrementally; load required persisted neighborhoods rather than truncating the entire usable graph to the first startup page. Preserve a legacy algorithm only in reproduction fixtures if needed. Keep `decay_all()` as a documented standalone/legacy primitive; do not imply persistent natural decay when each retrieval starts a fresh activation state.

Acceptance: hand-computed chains/branches/cycles pass; four-node default chain does not falsely return a subthreshold node; insertion order and concurrent requests do not change outcomes; deleted/corrected facts cannot survive in cached graph results; scoped high-similarity foreign nodes cannot steal the seed; maintenance avoids all-pairs rebuilds.

Run: `py -3.13 -m pytest -q tests/test_spreading_contract.py tests/test_storage_sqlite.py tests/test_engine.py`.

### T11 - Finish consolidation, retention, and forgetting

Depends on: T05-T10.

Files: `mnemos/modules/sleep.py`, `mnemos/memory_service.py`, `mnemos/utils/storage.py`, `mnemos/config.py`, new `tests/test_memory_lifecycle.py`.

Steps: extract source-backed semantic facts, merge exact repeats and proven equivalent assertions, and retain unresolved contradictions. Only mark evidence processed after accepted output or an explicit no-durable-facts decision. Prune eligible processed raw evidence on schedule while retaining fact support excerpts. Implement suppression-aware forget across all derived state.

Acceptance: retry cannot increase observation counts without a new source event; empty/malformed model output is not success; diverse sessions cannot collapse into the same canned facts; retention reduces live rows while preserving supported facts; forget survives replay, restart, and reconsolidation; failed evidence stays visible.

Run: `py -3.13 -m pytest -q tests/test_memory_lifecycle.py tests/test_fact_evidence.py tests/test_maintenance.py`.

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

Run: `py -3.13 -m pytest -q tests/test_codex_hooks.py tests/test_hosts.py tests/test_mcp_tier2_smoke.py`, then T17. Contract source: https://learn.chatgpt.com/docs/hooks.

### T14 - Add the Hermes memory provider

Depends on: T04, T06, T09, T11.

Files: new `mnemos/adapters/hermes/`, `pyproject.toml`, `mnemos/hosts.py`, new `docs/hermes.md`, new `tests/test_hermes_provider.py`.

Steps: publish the optional `hermes_agent.memory_providers` entry point; implement the MemoryProvider ABC, configuration, initialize, prefetch, sync_turn, session-end/shutdown, and explicit memory-write mirroring. Availability checks do not perform network I/O. Make sync_turn a bounded durable enqueue, with enrichment outside the host callback. Use session-provided cwd and hermes_home profile identity, not process cwd. Implement the optional v2 pre-compress checkpoint contract: report success only after sanitized evidence is durable in the service or local outbox; raise when a required checkpoint cannot be made. Do not automatically enable checkpoint-required mode or change Hermes compressor selection. Keep the module importable without Hermes installed through an optional integration boundary. Add the Windows/WSL bridge client and preview setup.

Acceptance: exact supported-version ABC/config contracts pass; profile/session scope does not follow unrelated process cwd; built-in memory remains intact; mirrored writes do not echo into repeated captures; service outage uses bounded local buffering; required checkpoint failure is truthful and replay-safe; real Hermes can recall a fact captured by each other host.

Run: `py -3.13 -m pytest -q tests/test_hermes_provider.py tests/test_service.py`, then T17. Contract source: https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin.

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

Depends on: T12-T16.

Files: new `tests/test_cross_host_contract.py`, new `docs/validation/local-memory-hosts.md`, new `benchmarks/scenarios/local-memory-continuity.jsonl`.

Steps: automate host-independent capture/recall/correct/forget fixtures, then execute equivalent real sessions on the installed Codex CLI, Codex desktop, Claude Code, and Hermes versions. Exercise Windows plus WSL where applicable, cloud-blocked operation, model outage, process kill, concurrent sessions, and config rollback.

Acceptance: host A teaches a source-backed project convention; after restart hosts B/C retrieve it; B corrects it and all retrieve the current version; forgetting survives maintenance and replay. Project B never sees project A's canary. Record actual host versions and transcripts with secrets removed.

Run: `py -3.13 -m pytest -q tests/test_cross_host_contract.py tests/test_mcp_compatibility.py`. Real-host results remain NOT RUN until observed; protocol tests do not promote host tiers.

### T18 - Prove the contribution of all five modules

Depends on: T07-T11, T17.

Files: `mnemos/benchmark.py`, new `benchmarks/ablation.py`, new `benchmarks/datasets/local-memory-heldout-v1.jsonl`, new `tests/test_ablation_contract.py`, `docs/retrieval-benchmarks.md`.

Steps: preserve deterministic mechanism fixtures, add real local-model runs, and compare scope-aware vector, lexical/hybrid, full pipeline, and one-module-disabled variants on identical data and budgets. Freeze labels and protocol before threshold tuning: at least 200 held-out session scenarios, 100 critical facts, 100 answerable queries, 100 no-answer queries, and 50 matched semantic/state query pairs, spanning all three hosts and multiple projects. Keep a disjoint development set for tuning. Run at least three real-model repetitions and report Wilson intervals for proportions plus paired bootstrap intervals for profile differences. Gates below apply to point estimates; intervals and sample counts must accompany them. Log model revisions, embeddings, dataset hashes, feature flags, hardware, warm/cold timings, and raw outputs. Measure extraction precision, critical-fact retention, retrieval accuracy, stale-fact rate, no-answer accuracy, token cost, actual row/byte growth, and latency. The existing harness fixes MockLLM but can honor an embedding configuration despite its simple default; label actual provider selection rather than assuming it.

Acceptance: each module can be activated independently, including at top-1; the full path is actually executed; compression is reported with retained-information accuracy; surprise filtering is compared with simple length/noise filtering; affective tests hold semantic content constant; graph tests require relevant associative recall, not merely more nodes. Report neutral/negative results without rewriting the dataset to obtain a desired lift.

Run: `py -3.13 -m pytest -q tests/test_ablation_contract.py tests/test_benchmark.py`, followed by versioned baseline/full/ablation runs that generate JSON artifacts. Add `--dataset`, `--profile`, `--disable-module`, `--provider`, `--repetitions`, and `--output` to the new runner.

### T19 - Reconcile release, documentation, and research claims

Depends on: T17, T18.

Files: `README.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `docs/client-compatibility-matrix.md`, `docs/release-checklist.md`, `docs/retrieval-benchmarks.md`, `docs/handoffs/substack-local-memory.md`, `tests/test_docs_consistency.py`; website changes in a separately reviewed checkout.

Steps: build and test the actual wheel from one commit; update runnable examples, support tiers, optional dependencies, and measured limitations. Produce a dated claim/evidence ledger and a paper errata/revision draft. Locate the actual paper publishing source before editing it; do not recreate it in an unrelated repo. Verify competitor comparisons with pinned primary evidence or remove absolute statements. Preserve historical paper versions and attach dated corrections.

Acceptance: install instructions match the release artifact; no synthetic result is presented as host or production proof; only completed pilot results close readiness gates; all published numbers have reproducible artifacts. Website/PyPI/paper publication is a distinct, explicitly authorized step.

Additional deliverable requested September 19: finalize a research and writing handoff for a roughly 2,500-word Substack article in Anthony's voice. The [handoff](../handoffs/substack-local-memory.md) must identify the final verified checkpoint, original promises, concrete repairs, measured results and failures, source locations, unresolved claims, and a suggested narrative. Apply Anthony's requested `writing-in-anthonys-voice` skill. The handoff is complete only after its evidence and open items are reconciled with the final validation state; the article itself and publication are separate work.

The working export is `mnemos-substack-research-packet-2026-09-19.zip` in the workspace's `deliverables` directory. It includes the handoff, Anthony's selected voice skill, source documents, selected validation reports, regression examples, and a manifest. Refresh the packet at final acceptance; the current snapshot explicitly retains open engineering and research gates.

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
- Run two weeks of daily use on at least two projects. External/general-availability readiness additionally requires the documented external pilot evidence; a single-user internal trial does not satisfy it.

All targets above are proposed engineering gates, not claims of current performance. A failing gate produces a documented limitation and repair task, never an automatic threshold adjustment.

First execution instruction: implement T01 only in an isolated implementation checkout, prove its focused tests, and review the patch before starting T02. Preserve live configuration and databases until backup/migration tasks are independently validated.
