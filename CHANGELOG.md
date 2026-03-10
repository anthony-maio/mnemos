# Changelog

## Unreleased

## 0.2.0 — 2026-03-10

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
