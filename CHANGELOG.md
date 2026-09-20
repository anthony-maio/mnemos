# Changelog

## Unreleased

- Compatible Hermes hosts can supply a separate durable evidence snapshot for ordinary turns and session end. An explicitly empty snapshot never falls back to model-facing text. The adapter validates native identity and summary classification before enqueueing; complete host compaction integration remains under validation.

- The Python affective classifier now honors an explicit model selection on supported providers without changing the shared provider's model. Omitted selection inherits the shared model; unsupported overrides and blank names fail visibly. Saved configurations containing the formerly ignored `llama3` value now request that model; omit it to inherit. See docs/configuration/affective-classifier-model.md.

- Hermes capture accepts validated original-message provenance from compatible hosts, preserving identity and forget suppression across copied rows. Marked compression summaries remain capture-only evidence; changing a native carrier's classification cannot create another observation. Complete host-side provenance support remains under validation.

- Existing SQLite memories remain searchable through scoped lexical retrieval after a transient embedding failure. Responses, context, and host diagnostics identify the degraded mode, including empty results; semantic scores stay unmeasured. Configuration and database errors still propagate.

- Client-created services now honor `MNEMOS_AUTO_START`, including bridge and managed CLI requests. Disabling startup preserves local capture buffering and access to an existing owner. Disposable native phases also pin their database and runtime paths.

- Configurations that omit an LLM provider now start with cognition unavailable instead of being mistaken for an explicit mock selection. Explicit mock cognition remains rejected by the managed runtime.
- Managed services now keep private, rotating JSON logs for startup, retrieval timing, and failure categories, with exception text and user content excluded from the file log.
- Provider failure logs now record exception types instead of exception messages, keeping request content and credentials out of those structured events while preserving exceptions and retry behavior for callers.
- Replaced insertion-dependent graph construction with persisted mutual top-k preferences in builder v4, including empty-source restart and bounded recovery. Automatic standalone rebuilds replace existing adjacency; explicit manual edges remain available.
- Fixed structured extraction alternatives so each decoder branch requires fact content and source evidence. Baseline and full Ollama development checks now preserve correction and forgetting across restart.
- Added a shared authenticated loopback service for CLI, MCP, UI, and host adapters, with one SQLite writer, durable jobs, recovery leases, private runtime files, and sanitized capture outboxes.
- Added registered profile/project/workspace identities, scoped retrieval and graph traversal, source evidence, expected-version correction, and forget suppression across restart and replay.
- Added typed local-model extraction, independent module switches, baseline retrieval by default, bounded graph rebuilding, and measured score traces. Recall no longer silently rewrites facts.
- Added Codex lifecycle hooks, updated Claude transcript capture, and an optional Hermes provider with an explicit Windows/WSL bridge. Native host acceptance remains separately recorded.
- Added project enable/pause and fact inspection/correction/forget in the UI, input-specific provider probing, consistent backup/restore, resumable offline reindex, and authenticated service-stop.
- Added schema-migration backups, FULL SQLite synchronization, atomic fact/vector writes, and explicit embedding identity checks.
- Added real-model evaluation and 10,000-fact performance harnesses. Passing mechanism tests does not establish a biomimetic quality advantage; negative measurements remain part of the record.
- These changes are unreleased. See docs/validation/2026-09-16/progress.md for observed validation and open acceptance gates.

## 0.7.0 -- 2026-04-21

Closes the auto-memory loop in Claude Code: retrieval is now deterministic and invisible to the model, not dependent on the model choosing to prime itself.

### Added
- `mnemos-cli retrieve-hook` subcommand emits Claude Code `hookSpecificOutput.additionalContext` JSON so hooks can inject relevant memories directly into the model context
- `SessionStart` hook wired in `hooks/hooks.json` -- primes each new session with project-scoped memories
- Second handler on `UserPromptSubmit` -- query-aware retrieval on every turn, running alongside the existing autostore write
- `MNEMOS_AUTO_RETRIEVE` env var (default on) so users who want writes without reads can disable auto-priming
- `FastembedProvider` -- local, quantized ONNX embeddings via `fastembed`. CPU-only, no daemon, no API keys, fully offline after first-run model download (~100MB)

### Changed
- `hooks/hooks.json` now delivers both sides of the memory loop at install time (write via `autostore-hook`, read via `retrieve-hook`) with no manual `settings.json` edits required
- **Default embedding provider switched from `simple` (TF-IDF) to `fastembed`** -- a fresh install now produces real semantic retrieval out of the box, no configuration required
- `ollama`, `openai`, `openclaw`, and `openrouter` providers remain supported as power-user swaps via `MNEMOS_EMBEDDING_PROVIDER`
- `fastembed>=0.4` is now a hard dependency so the default path works with zero configuration; accepted tradeoff is a larger wheel footprint on uncommon platforms

### Fixed
- `mnemos-cli autostore-hook` now queues idempotent episodic events instead of writing semantic memory directly, so duplicate hook fires collapse cleanly and consolidation works across process restarts
- `mnemos-cli repair-db` migrates bloated SQLite stores into a clean target DB, exact-dedupes canonical chunks, and rebuilds FTS plus `sqlite-vec` once instead of carrying shadow-table bloat forward
- surprisal and affective fields now stay null when unmeasured instead of serializing fake midpoint defaults
- retrieval now queues durable reconsolidation jobs and later consolidation updates the same chunk in place with revision history
- repaired SQLite stores now persist typed `SEMANTIC_SIMILAR` edges, and the runtime can infer the correct simple-embedder dimension from an existing SQLite vector store during CLI and MCP startup

## 0.6.0 — 2026-03-31

Beta update focused on making SQLite the only shipped persistence path and exposing recall-gated plasticity on the live runtime.

### Changed
- removed dead Qdrant and Neo4j runtime code, migration flags, profile docs, and backend packaging extras so Mnemos now has one persistent storage story: SQLite
- simplified `mnemos-cli migrate-store` to SQLite-to-SQLite copy and upgrade workflows
- updated MCP, skill, and profile docs to describe the shipped local database path without optional backend branching

### Added
- optional recall-gated plasticity during sleep consolidation, with config knobs, MCP surfacing, and regression coverage

## 0.3.1 — 2026-03-15

Beta update focused on unifying Mnemos around a single local SQLite backend and validating retrieval quality on that shipped path.

### Changed
- public runtime/configuration/docs now present SQLite as the definitive shipped backend, with a single local-file persistence story across hosts
- CLI profile/doctor flows now align to the SQLite-only path, including the simplified `default` profile and `--chunk-threshold` doctor gate
- benchmark tooling now targets `memory` and `sqlite` store modes, and the claim-driving benchmark is green on the SQLite release path
- Claude plugin bootstrap now installs only MCP support rather than backend-specific storage extras
- public API exports and compatibility docs no longer present Qdrant and Neo4j as first-class runtime backends

### Added
- expanded SQLite graph-store validation covering sparse edge persistence, FTS, and `sqlite-vec` indexing in the shipped local database path

## 0.3.0 — 2026-03-12

Public beta update focused on shared persistence, host soft-auto workflows, and OpenClaw onboarding.

### Changed
- Neo4j-backed persistence is now wired into runtime settings, health checks, and live shared-config validation
- Neo4j reads and schema setup are hardened for real server usage, including safer optional-property reads and constraint handling
- Codex soft-auto guidance now ships as a stronger `AGENTS.md` pack, with an optional Codex Automation prompt for scheduled Mnemos hygiene checks
- Cursor soft-auto now includes a generated `.cursor/rules/mnemos-memory.mdc` rule alongside `.cursor/mcp.json`
- docs, UI copy, and compatibility notes now explicitly position hard auto-capture as host-dependent outside Claude Code

### Added
- `Neo4jStore` experimental backend with validated write/read/delete smoke coverage against a live Neo4j instance
- `mnemos-cli migrate-store` for Qdrant/SQLite/Neo4j backfill and migration workflows
- shared antigravity artifact generation for generic policy text, Codex AGENTS blocks, Codex automation prompts, and Cursor rules
- publishable OpenClaw / ClawHub skill guidance in `skills/mnemos-memory`
- `docs/clawhub-skill.md` for skill packaging/publishing and positioning
- regression coverage for Cursor rule generation and multi-artifact host integration previews

## 0.2.0 — 2026-03-10

Public beta release.

### Changed
- scoped consolidation now preserves `scope` and `scope_id`, so project and workspace memories do not bleed during sleep consolidation
- retrieval now persists access touches and keeps remote embedding providers off the async event loop
- Qdrant scoped retrieval now uses native payload filters instead of full collection scans in the normal engine path
- Tier 1 MCP support is now validated end-to-end for Claude Code, Claude Desktop, and generic stdio hosts
- README, website, and release docs now lead with reliable scoped memory for coding agents instead of category-defining replacement claims

### Added
- `mnemos_health` and `mnemos-cli doctor` now surface legacy unscoped chunk counts and scope-isolation readiness
- Codex setup docs and `mnemos-cli antigravity codex` for MCP + `AGENTS.md` workflows
- OSS trust docs: `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, and `SUPPORT.md`
- public OSS intake templates for bugs, feature requests, host compatibility reports, and pull requests
- public release packaging guidance in `docs/public-release-package.md`

## 0.1.0 — 2026-03-05

Initial release.

### Features
- **SurprisalGate** — predictive coding memory gate (only encodes prediction errors)
- **MutableRAG** — memory reconsolidation (facts evolve on retrieval)
- **AffectiveRouter** — emotional state-dependent memory routing
- **SleepDaemon** — hippocampal-neocortical consolidation (episodic to semantic)
- **SpreadingActivation** — graph-based associative context retrieval
- **MnemosEngine** — orchestrator composing all five modules
- **MCP Server** — 7 tools for Claude Code / Cursor / Windsurf integration
- **CLI** — `mnemos-mcp` for MCP stdio transport, `mnemos-cli` for shell commands
- **Storage backends** — InMemoryStore, SQLiteStore
- **LLM providers** — MockLLMProvider, OllamaProvider, OpenAIProvider
- **SimpleEmbeddingProvider** — TF-IDF + random projection (zero external deps)
