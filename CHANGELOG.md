# Changelog

## Unreleased

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
