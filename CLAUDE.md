# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**mnemos** — a biomimetic memory architecture library for LLMs. Implements five neuroscience-inspired memory mechanisms: surprisal-gated encoding, mutable RAG, affective routing, sleep consolidation, and spreading activation. Python 3.10+, MIT licensed, alpha (0.1.0).

## Repository Layout

```
mnemos/        → Source package
  modules/     → Five biomimetic memory modules
  utils/       → LLM, embedding, and storage provider abstractions
  cli.py       → Shell CLI for hooks and automation
  mcp_server.py → MCP server for agent integration
tests/         → pytest suite (3 files: engine, modules, types)
examples/      → Runnable demos (basic_usage, full_pipeline, mcp_agent_demo)
docs/          → MCP_INTEGRATION.md, claude-code-hooks.json
# Website: https://mnemos.making-minds.ai (separate mnemos-web repo)
```

## Development Commands

```bash
# Install (editable + dev deps)
pip install -e '.[dev]'

# Run all tests
pytest

# Run a single test file
pytest tests/test_engine.py

# Run a single test by name
pytest tests/test_modules.py -k "test_surprisal"

# Format
black .

# Type check (strict mode)
mypy .

# Run MCP server (stdio transport)
mnemos-mcp

# CLI commands (shell-friendly, defaults to SQLite)
mnemos-cli store "some content"
mnemos-cli retrieve "query" --top-k 5
mnemos-cli consolidate
mnemos-cli stats
```

Tests use `MockLLMProvider + SimpleEmbeddingProvider + InMemoryStore` — no external services needed. Async tests auto-detected via `asyncio_mode = "auto"`.

## Architecture

**MnemosEngine** (`engine.py`) orchestrates five modules through three paths:

- **Encode** (`process()`): Input → SurprisalGate (filter novelty) → AffectiveRouter (tag emotion) → Store + SleepDaemon buffer
- **Decode** (`retrieve()`): Query → AffectiveRouter (classify) → SpreadingActivation (graph propagation) → AffectiveRouter (re-rank) → MutableRAG (reconsolidate stale facts) → Results
- **Consolidate** (`consolidate()`): SleepDaemon extracts semantic facts from episodic buffer → writes to store → prunes

**Five modules** (all in `modules/`, all implement `async process()`, `async retrieve()`, `get_stats()`):

| Module | File | Role |
|---|---|---|
| SurprisalGate | `surprisal.py` | Predictive coding gate — only encodes prediction errors |
| MutableRAG | `mutable_rag.py` | Reconsolidation — updates stale memories on recall |
| AffectiveRouter | `affective.py` | State-dependent retrieval via emotional context |
| SleepDaemon | `sleep.py` | Episodic→semantic consolidation (hippocampal replay) |
| SpreadingActivation | `spreading.py` | Graph-based associative retrieval |

**Provider abstractions** (`utils/`):
- `llm.py` — MockLLMProvider, OllamaProvider, OpenAIProvider
- `embeddings.py` — SimpleEmbeddingProvider + cosine_similarity()
- `storage.py` — InMemoryStore, SQLiteStore (abstract: MemoryStore)

**Core types** (`types.py`): MemoryChunk, CognitiveState, Interaction, ProcessResult, ActivationNode, ConsolidationResult

**Configuration** (`config.py`): Pydantic models. Top-level `MnemosConfig` composes per-module configs. Both `mnemos-mcp` and `mnemos-cli` read `MNEMOS_*` environment variables.

## Entry Points

- `mnemos-mcp` → `mcp_server.py:main` — MCP server (stdio transport, for Claude Code/Cursor/Windsurf)
- `mnemos-cli` → `cli.py:main` — Shell CLI (for hooks, cron, automation; defaults to SQLite)

## Code Style

- Black formatting, 100-char line length
- Strict mypy
- Fully async public API
- Logging via `logging` module (gated by `config.debug`)
- Build backend: hatchling
