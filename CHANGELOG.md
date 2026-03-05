# Changelog

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
