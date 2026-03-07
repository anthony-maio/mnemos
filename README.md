# mnemos (μνῆμος)

**Biomimetic memory architectures for LLMs — built on how the brain actually works.**

[![PyPI version](https://img.shields.io/pypi/v/mnemos-memory.svg)](https://pypi.org/project/mnemos-memory/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Website](https://img.shields.io/badge/Website-mnemos.making--minds.ai-purple.svg)](https://mnemos.making-minds.ai)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io)

---

## The Problem

Current LLM memory falls into two failure modes: **context stuffing** (cramming millions of tokens into RAM until it costs a fortune) or **standard RAG** (dumping everything into an append-only vector database and hoping cosine similarity is enough).

Both treat memory as a hard drive — static, lossless, and disorganized. A mundane greeting gets the same embedding priority as a production outage. A fact from 2024 sits quietly contradicting the updated fact from 2025, and the LLM burns context tokens resolving the conflict on every single retrieval.

The result: bloated stores, stale data, no associative reasoning, and zero emotional context.

---

## The Insight

Human memory is efficient precisely because it does the opposite. It is:

- **Reconstructive, not reproductive** — every recall rewrites the trace with current context
- **Surprisal-gated** — the brain only encodes events that violate predictions
- **State-dependent** — fear retrieves fear, urgency retrieves crisis solutions
- **Lossy by design** — sleep compresses episodic logs into semantic abstractions and discards the rest
- **Associative** — remembering "server" pre-activates "AWS," "downtime," and "nginx" before you ask

mnemos implements all five of these mechanisms as composable Python modules, grounded in the neuroscience literature. They can be used independently or composed together through `MnemosEngine`.

---

## Five Architectures

### 1. `SurprisalGate` — Predictive Coding Memory Gate

> *Inspired by Friston's Active Inference / Predictive Processing: the brain encodes only prediction errors, not the expected.*

A fast LLM continuously predicts the user's next intent. When a new input arrives, mnemos computes the cosine distance between the prediction embedding and the actual input embedding. Low divergence (expected input) → discarded. High divergence (genuine surprise) → stored with a salience weight proportional to the prediction error.

This eliminates context bloat at the ingestion layer. A routine "sounds good" never touches long-term memory. "My production database just corrupted" gets maximum salience.

```python
from mnemos import MnemosEngine, MnemosConfig, SurprisalConfig, Interaction

engine = MnemosEngine(
    config=MnemosConfig(
        surprisal=SurprisalConfig(threshold=0.3)  # ~72° in embedding space
    )
)

result = await engine.process(Interaction(role="user", content="sounds good"))
# result.stored → False  (low surprisal, discarded)

result = await engine.process(Interaction(role="user", content="I'm migrating from Python to Rust"))
# result.stored → True   (high surprisal, salience ≈ 0.87)
# result.reason → "Surprisal 0.87 exceeds threshold 0.3"
```

---

### 2. `MutableRAG` — Memory Reconsolidation

> *Inspired by the destabilization-restabilization cycle: every act of recall makes the memory labile, allowing new context to be integrated before re-encoding.*

Standard RAG is append-only. When a user says "I use React" in 2024 and "I'm migrating to Rust" in 2026, both facts live forever. Every retrieval forces the LLM to spend tokens resolving the contradiction.

MutableRAG fixes this with a **Read-Evaluate-Mutate loop**. Retrieved chunks are flagged as "labile." A background async task checks whether the new conversational context contradicts or updates the stored fact. If it does, the original chunk is physically overwritten — not duplicated — with the synthesized update. The `version` counter on each `MemoryChunk` tracks how many times it has been reconsolidated.

```python
# After processing: "I use React for frontend work."
# Then later: "We're migrating to Svelte next quarter."

# On retrieval, MutableRAG detects the contradiction and rewrites in background:
memories = await engine.retrieve("frontend framework", reconsolidate=True)

# The stored chunk is now:
# "User is migrating from React to Svelte for frontend work." (version=2)
# The old "I use React" chunk no longer exists.
```

---

### 3. `AffectiveRouter` — Amygdala Filter

> *Inspired by Bower's (1981) mood-congruent memory theory and the amygdala's role in tagging memories with emotional salience.*

Embedding models retrieve on semantic similarity alone. A panicked "URGENT: server is down" retrieves the same as a calm "what's our server stack?" — the emotional context is invisible.

`AffectiveRouter` classifies every interaction on three axes — **valence** (−1 to +1), **arousal** (0 = calm to 1 = urgent), **complexity** (0 = simple to 1 = complex) — and attaches this `CognitiveState` as metadata. During retrieval, the scoring formula blends semantic similarity with affective state match:

```
final_score = (cosine_similarity × 0.7) + (state_match × 0.3)
```

When a user is panicking about a bug, mnemos surfaces how previous crises were resolved — not just semantically similar code snippets.

```python
from mnemos.types import CognitiveState

# A chunk encoded during a high-urgency incident:
# chunk.cognitive_state = CognitiveState(valence=-0.8, arousal=0.95, complexity=0.7)

# During a calm planning session, this chunk scores lower.
# During another incident, it rises to the top — because state matches.
```

---

### 4. `SleepDaemon` — Hippocampal-Neocortical Consolidation

> *Inspired by the two-stage memory model: the hippocampus stores fast, raw episodic traces; slow-wave sleep replays them to the neocortex, extracts semantic knowledge, and prunes the originals.*

Every interaction enters the episodic buffer regardless of the surprisal gate — the "hippocampus." When idle (configurable interval), `SleepDaemon` replays the buffer through an LLM consolidation pass, extracting permanent facts and user preferences. These are written as semantic `MemoryChunk` objects to long-term storage. The raw episodic buffer is then pruned.

Optionally, the daemon identifies repeated reasoning patterns across sessions and generates Python tool code to automate them — declarative memory crystallizing into procedural reflex.

```python
# After a session of interactions:
result = await engine.consolidate()

print(result.facts_extracted)
# [
#   "User is a Python developer specializing in ML",
#   "User deploys models on AWS SageMaker",
#   "User's team uses GitHub Actions for CI/CD",
#   "User prefers Neovim and dark mode",
# ]

print(f"Pruned {result.chunks_pruned} raw episodes → {len(result.facts_extracted)} permanent facts")
```

---

### 5. `SpreadingActivation` — Energy-Based Graph RAG

> *Inspired by Collins & Loftus (1975) spreading activation theory: concepts in semantic memory are nodes in a network, activation propagates along edges and decays with distance.*

Vector search is a point-in-space lookup. It retrieves exact mathematical matches but misses the associative "train of thought." Hearing "Docker bug" should pre-activate "Ubuntu," "last Tuesday's deployment," and "nginx config" — not because they match the query string, but because they are connected in the memory graph.

`SpreadingActivation` injects activation energy (1.0) at the best-match node. Energy flows along graph edges, decaying 20% per hop. Every node above the activation threshold is included in the retrieval results, creating a moving spotlight of associative context.

```
"Docker bug" ──(1.0)──▶ node: "Docker networking issue from last week"
                              │(0.8)
                              ▼
                         node: "Ubuntu 22.04 server config"
                              │(0.64)
                              ▼
                         node: "nginx reverse proxy setup"  ← threshold: 0.3 ✓
                              │(0.51)
                              ▼
                         node: "old nginx config"           ← threshold: 0.3 ✓
```

```python
from mnemos import SpreadingActivation
from mnemos.config import SpreadingConfig

sa = SpreadingActivation(
    embedder=embedder,
    config=SpreadingConfig(
        initial_energy=1.0,
        decay_rate=0.2,        # 20% loss per hop
        activation_threshold=0.3,
        max_hops=3,
    )
)
```

---

## Quick Start

```bash
pip install mnemos-memory
```

> **Note:** The PyPI package is `mnemos-memory` but the import name is just `import mnemos`.

Zero external dependencies required. The default configuration uses `MockLLMProvider` and `SimpleEmbeddingProvider` for instant experimentation.

```python
import asyncio
from mnemos import MnemosEngine, Interaction

async def main():
    engine = MnemosEngine()  # MockLLM + SimpleEmbedding + InMemoryStore

    await engine.process(Interaction(role="user", content="I use Python for ML."))
    await engine.process(Interaction(role="user", content="I deploy on AWS SageMaker."))
    await engine.process(Interaction(role="user", content="Sure, sounds good."))  # filtered

    memories = await engine.retrieve("cloud infrastructure", top_k=3)
    for m in memories:
        print(m.content)

asyncio.run(main())
```

**With Ollama (recommended for production):**

```bash
pip install 'mnemos-memory[ollama]'
```

```python
from mnemos import MnemosEngine, MnemosConfig
from mnemos.utils import OllamaProvider, SQLiteStore

engine = MnemosEngine(
    config=MnemosConfig(),
    llm=OllamaProvider(model="llama3"),
    store=SQLiteStore(db_path="memory.db"),
)
```

**With OpenAI:**

```bash
pip install 'mnemos-memory[openai]'
```

```python
from mnemos.utils import OpenAIProvider

engine = MnemosEngine(
    llm=OpenAIProvider(api_key="sk-...", model="gpt-4o-mini"),
    store=SQLiteStore(db_path="memory.db"),
)
```

---

## MCP Integration

mnemos ships a full [Model Context Protocol](https://modelcontextprotocol.io) server. Any MCP-compatible agent — Claude Code, Cursor, Windsurf, Cline — can discover and call mnemos tools natively.

```bash
pip install 'mnemos-memory[mcp]'
```

### Claude Code Plugin Install (Like ClaudeMem)

This repository now includes a native Claude Code plugin package (`.claude-plugin/plugin.json`) that auto-wires Mnemos MCP.

```text
/plugin marketplace add anthony-maio/mnemos
/plugin install mnemos-memory@mnemos-marketplace
```

On first run, the plugin bootstraps a local virtual environment under `.claude-plugin/.venv`, installs Mnemos with MCP extras, and launches the MCP server over stdio.

Default plugin behavior:
- persistent SQLite memory store at `.claude-plugin/mnemos.db`
- automatic provider selection:
  - `openclaw` if `MNEMOS_OPENCLAW_API_KEY` exists
  - otherwise `openai` if `MNEMOS_OPENAI_API_KEY` exists
  - otherwise `ollama` if `MNEMOS_OLLAMA_URL` exists
  - otherwise `mock`
 - embedding provider inferred from the selected LLM provider unless explicitly overridden
 - if `MNEMOS_STORE_TYPE=qdrant`, the plugin bootstrap installs the `qdrant` extra automatically

The server exposes eight tools:

| Tool | Description |
|------|-------------|
| `mnemos_store` | Process a memory through the full pipeline (surprisal gate → affective tagging → graph), with optional `scope` + `scope_id` |
| `mnemos_retrieve` | Retrieve with spreading activation + emotional re-ranking + reconsolidation, with scoped filtering (`current_scope`, `scope_id`, `allowed_scopes`) |
| `mnemos_consolidate` | Trigger sleep consolidation: episodic buffer → semantic long-term memory |
| `mnemos_forget` | Delete a specific memory by ID |
| `mnemos_stats` | System-wide statistics across all modules |
| `mnemos_health` | Profile readiness and dependency diagnostics |
| `mnemos_inspect` | Full details on a specific memory chunk |
| `mnemos_list` | List all stored memories |

Readiness check:

```bash
mnemos-cli doctor
```

Threshold-aware doctor check (recommend qdrant only when limits are hit):

```bash
mnemos-cli doctor --qdrant-chunk-threshold 5000 --latency-p95-threshold-ms 250 --observed-p95-ms 180
```

One-command profile generation:

```bash
# Starter (default) profile
mnemos-cli profile starter --format dotenv --write .mnemos.profile.env

# Local performance profile (embedded qdrant)
mnemos-cli profile local-performance --format dotenv --write .mnemos.profile.env
```

Scoped memory examples (cross-project aware):

```bash
# Store project-scoped memory
mnemos-cli store "Use uv for Python tooling in this repo" --scope project --scope-id repo-alpha

# Store global preference memory
mnemos-cli store "Prefer concise summaries" --scope global

# Retrieve from current project plus global memory
mnemos-cli retrieve "tooling preferences" --current-scope project --scope-id repo-alpha --allowed-scopes project,global

# Audit current project memories
mnemos-cli list --scope project --scope-id repo-alpha --limit 20
mnemos-cli search "terraform" --scope project --scope-id repo-alpha
mnemos-cli export --scope project --scope-id repo-alpha --format jsonl --output .mnemos-export.jsonl

# Dry-run purge old project memories, then confirm
mnemos-cli purge --scope project --scope-id repo-alpha --older-than-days 30 --dry-run
mnemos-cli purge --scope project --scope-id repo-alpha --older-than-days 30 --yes
```

Profile + compatibility docs:
- [docs/profiles/starter-sqlite.md](docs/profiles/starter-sqlite.md)
- [docs/profiles/local-performance-embedded-qdrant.md](docs/profiles/local-performance-embedded-qdrant.md)
- [docs/profiles/scale-external-qdrant.md](docs/profiles/scale-external-qdrant.md)
- [docs/cursor-antigravity.md](docs/cursor-antigravity.md)
- [docs/mcp-transport-contract.md](docs/mcp-transport-contract.md)
- [docs/client-compatibility-matrix.md](docs/client-compatibility-matrix.md)

### Claude Code / Claude Desktop

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mnemos": {
      "command": "mnemos-mcp",
      "env": {
        "MNEMOS_LLM_PROVIDER": "ollama",
        "MNEMOS_LLM_MODEL": "llama3",
        "MNEMOS_STORE_TYPE": "sqlite",
        "MNEMOS_SQLITE_PATH": "~/.mnemos/memory.db"
      }
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "mnemos": {
      "command": "mnemos-mcp",
      "env": {
        "MNEMOS_LLM_PROVIDER": "mock",
        "MNEMOS_STORE_TYPE": "sqlite",
        "MNEMOS_SQLITE_PATH": ".mnemos/memory.db"
      }
    }
  }
}
```

### Windsurf

Add to `~/.windsurf/mcp.json`:

```json
{
  "mcpServers": {
    "mnemos": {
      "command": "mnemos-mcp",
      "env": {
        "MNEMOS_LLM_PROVIDER": "mock",
        "MNEMOS_STORE_TYPE": "sqlite",
        "MNEMOS_SQLITE_PATH": "~/.mnemos/memory.db"
      }
    }
  }
}
```

Once configured, you can tell your agent:

```
"Remember that I use Neovim and prefer dark mode."
"What do you know about my infrastructure setup?"
"Consolidate what you've learned from our conversation."
"Forget the memory about the old server IP."
```

The agent's memory now filters the mundane, tags emotional context, updates stale facts on recall, and compresses session logs into lasting knowledge — automatically.

For deterministic Claude Code auto-memory, use the shipped hook config at [docs/claude-code-hooks.json](docs/claude-code-hooks.json). It auto-ingests user prompts and high-signal tool failures via `mnemos-cli autostore-hook`, then consolidates on `PreCompact`/`Stop`.

Memory writes now pass through a shared safety firewall across ingestion, reconsolidation, and sleep consolidation. Configure with:
- `MNEMOS_MEMORY_SAFETY_ENABLED` (`true` by default)
- `MNEMOS_MEMORY_SECRET_ACTION` (`block` default)
- `MNEMOS_MEMORY_PII_ACTION` (`redact` default)

Governance controls for retention and growth limits:
- `MNEMOS_MEMORY_CAPTURE_MODE` (`all` | `manual_only` | `hooks_only`)
- `MNEMOS_MEMORY_RETENTION_TTL_DAYS` (`0` disables TTL pruning)
- `MNEMOS_MEMORY_MAX_CHUNKS_PER_SCOPE` (`0` disables per-scope cap)

---

## Architecture Diagram

```
                         ┌──────────────────────────────────────┐
                         │           MnemosEngine                │
                         └──────────────────────────────────────┘

  ENCODE PATH (process)
  ──────────────────────────────────────────────────────────────────────────
  Interaction
      │
      ├─────────────────────────────────────────────────────────▶ SleepDaemon
      │                                                          (episodic buffer)
      ▼
  SurprisalGate ──── low surprisal ──▶ [DISCARD]
      │
      │ high surprisal
      ▼
  AffectiveRouter
  (classify valence / arousal / complexity → tag CognitiveState)
      │
      ▼
  MemoryStore (SQLite / InMemory)
      │
      ▼
  SpreadingActivation.add_node()
  (auto-connect to semantic neighbors in graph)


  RETRIEVE PATH (retrieve)
  ──────────────────────────────────────────────────────────────────────────
  Query
      │
      ├──▶ AffectiveRouter.classify_state(query)
      │
      ├──▶ SpreadingActivation.retrieve()
      │    (inject energy → propagate along edges → threshold filter)
      │
      ├──▶ AffectiveRouter.retrieve()
      │    score = similarity × 0.7 + state_match × 0.3
      │
      ├──▶ Merge + re-rank (activation boost for graph-connected nodes)
      │
      └──▶ MutableRAG.flag_labile() → async reconsolidation background task


  MAINTENANCE PATH (consolidate)
  ──────────────────────────────────────────────────────────────────────────
  Idle trigger
      │
      ▼
  SleepDaemon
      │
      ├──▶ LLM: extract permanent facts from episodic buffer
      ├──▶ Write semantic MemoryChunks to long-term store
      ├──▶ Add new nodes to SpreadingActivation graph
      ├──▶ Prune raw episodic buffer
      └──▶ (optional) Proceduralize repeated patterns → generate Python tools
```

---

## Configuration

All configuration is expressed as Pydantic models and can be serialized to/from JSON/YAML.

```python
from mnemos import MnemosEngine, MnemosConfig
from mnemos.config import (
    SurprisalConfig, MutableRAGConfig,
    AffectiveConfig, SleepConfig, SpreadingConfig
)

config = MnemosConfig(
    surprisal=SurprisalConfig(threshold=0.25),
    sleep=SleepConfig(consolidation_interval_seconds=1800),
    spreading=SpreadingConfig(decay_rate=0.15, max_hops=4),
    debug=True,
)
engine = MnemosEngine(config=config)
```

### Key Options

| Module | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| `SurprisalConfig` | `threshold` | `0.3` | Cosine distance threshold for encoding. Higher = stricter gate, fewer memories stored. |
| `SurprisalConfig` | `history_window` | `10` | Recent turns used for intent prediction. |
| `MutableRAGConfig` | `enabled` | `True` | Toggle async reconsolidation on retrieval. |
| `MutableRAGConfig` | `reconsolidation_cooldown_seconds` | `60` | Minimum gap between reconsolidations of the same chunk. |
| `AffectiveConfig` | `weight_similarity` | `0.7` | Semantic similarity weight in retrieval scoring. |
| `AffectiveConfig` | `weight_state` | `0.3` | Affective state match weight in retrieval scoring. |
| `SleepConfig` | `consolidation_interval_seconds` | `3600` | Minimum idle time before consolidation triggers. |
| `SleepConfig` | `min_episodes_before_consolidation` | `10` | Minimum episodic buffer depth before consolidation. |
| `SleepConfig` | `enable_proceduralization` | `False` | Generate Python tools from repeated reasoning patterns. |
| `SpreadingConfig` | `initial_energy` | `1.0` | Activation energy injected at seed node. |
| `SpreadingConfig` | `decay_rate` | `0.2` | Energy lost per graph hop (20%). |
| `SpreadingConfig` | `activation_threshold` | `0.3` | Minimum energy for a node to be included in results. |
| `SpreadingConfig` | `max_hops` | `3` | Maximum graph traversal depth. |
| `SpreadingConfig` | `auto_connect_threshold` | `0.6` | Minimum cosine similarity for auto-connecting new nodes. |

### MCP Server Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMOS_LLM_PROVIDER` | `mock` | `mock`, `ollama`, `openai`, or `openclaw` |
| `MNEMOS_LLM_MODEL` | `llama3` | Model name for the LLM provider |
| `MNEMOS_OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `MNEMOS_OPENAI_API_KEY` | — | Required when using `openai` provider |
| `MNEMOS_OPENAI_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `MNEMOS_OPENCLAW_API_KEY` | — | OpenClaw API key (or fallback to `MNEMOS_OPENAI_API_KEY`) |
| `MNEMOS_OPENCLAW_URL` | — | OpenClaw API base URL (or fallback to `MNEMOS_OPENAI_URL`) |
| `MNEMOS_EMBEDDING_PROVIDER` | inferred from `MNEMOS_LLM_PROVIDER`, else `simple` | `simple`, `ollama`, `openai`, or `openclaw` |
| `MNEMOS_EMBEDDING_MODEL` | provider-dependent | Embedding model name (e.g. `nomic-embed-text`) |
| `MNEMOS_EMBEDDING_DIM` | `384` | Embedding dimension for `simple` provider |
| `MNEMOS_STORE_TYPE` | `memory` | `memory`, `sqlite`, or `qdrant` |
| `MNEMOS_SQLITE_PATH` | `mnemos_memory.db` | SQLite database path |
| `MNEMOS_QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `MNEMOS_QDRANT_API_KEY` | — | Optional Qdrant API key |
| `MNEMOS_QDRANT_PATH` | — | Local embedded Qdrant path (overrides URL) |
| `MNEMOS_QDRANT_COLLECTION` | `mnemos_memory` | Qdrant collection name |
| `MNEMOS_QDRANT_VECTOR_SIZE` | — | Optional fixed vector size for pre-created collections |
| `MNEMOS_SURPRISAL_THRESHOLD` | `0.3` | Surprisal gate sensitivity |
| `MNEMOS_DEBUG` | `false` | Enable verbose debug logging |

Backward-compatible aliases are supported for migration:
- `MNEMOS_STORAGE` -> `MNEMOS_STORE_TYPE`
- `MNEMOS_DB_PATH` -> `MNEMOS_SQLITE_PATH`

If `MNEMOS_EMBEDDING_PROVIDER` is unset, Mnemos now infers it from `MNEMOS_LLM_PROVIDER` for `ollama`, `openai`, and `openclaw`. Set it explicitly to `simple` if you want the lightweight fallback.

---

## Comparison with Existing Tools

Every other memory library for LLMs is, at its core, a wrapper around a vector database. mnemos is the only open-source implementation of the neuroscience mechanisms that make biological memory actually work.

| Feature | mnemos | Mem0 | Zep | LangMem | MemGPT |
|---------|:------:|:----:|:---:|:-------:|:------:|
| Surprisal-gated encoding (predictive coding) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Mutable memories (reconsolidation, no stale data) | ✅ | Partial | Partial | ❌ | ❌ |
| Affective/emotional state-dependent retrieval | ✅ | ❌ | ❌ | ❌ | ❌ |
| Sleep consolidation (episodic → semantic compression) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Spreading activation graph RAG | ✅ | ❌ | ❌ | ❌ | ❌ |
| Append-only vector storage | ❌ | ✅ | ✅ | ✅ | ✅ |
| MCP server (Claude Code, Cursor, Windsurf) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Zero-dependency quick start | ✅ | ❌ | ❌ | ❌ | ❌ |
| Fully composable modules | ✅ | ❌ | ❌ | Partial | ❌ |
| Pydantic-validated config | ✅ | ❌ | ❌ | ❌ | ❌ |

The distinction isn't just architectural. It's causal: standard tools retrieve stale facts, accumulate contradictions, miss emotional context, and grow without bound. mnemos doesn't.

---

## Tech Stack

**Core (zero optional dependencies for quick start):**
- `numpy >= 1.24` — embedding arithmetic and cosine similarity
- `pydantic >= 2.0` — validated config and domain types
- `httpx >= 0.24` — async HTTP for LLM providers

**Optional (install what you need):**
- `ollama` — local LLM inference via Ollama (`pip install 'mnemos-memory[ollama]'`)
- `openai` — OpenAI or any OpenAI-compatible API (`pip install 'mnemos-memory[openai]'`)
- `qdrant` — Qdrant vector database backend (`pip install 'mnemos-memory[qdrant]'`)
- `mcp` — MCP server for Claude Code, Cursor, Windsurf (`pip install 'mnemos-memory[mcp]'`)
- `neo4j` — Neo4j (planned) graph backend for SpreadingActivation at scale (`pip install 'mnemos-memory[neo4j]'`)

**Install everything:**
```bash
pip install 'mnemos-memory[all]'
```

**Storage backends built-in:**
- `InMemoryStore` — zero setup, for development and testing
- `SQLiteStore` — persistent, zero external services, suitable for personal deployments
- `QdrantStore` — vector database backend for scalable retrieval

## Retrieval Benchmark Harness

Run reproducible retrieval benchmarks with `Recall@k`, `MRR`, and `p95` latency:

```bash
mnemos-benchmark --stores memory,sqlite,qdrant --retrievers baseline,engine --top-k 5
```

You can provide a custom dataset (`.json` or `.jsonl`) with `id`, `content`, and `queries` (plus optional scoped fields like `scope`, `scope_id`, query-level `allowed_scopes`):

```bash
mnemos-benchmark --stores qdrant --retrievers baseline,engine --dataset ./benchmarks/retrieval.jsonl --top-k 10
```

Replacement-claim gate run:

```bash
mnemos-benchmark --stores memory --retrievers baseline,engine --dataset-pack claim-driving --top-k 1 --enforce-production-gate
```

Gate output fields live under `gates.production_replacement.*`.
Use `--baseline-scope-aware` if you want baseline retrieval to apply scope filters in scoped-memory datasets.

## Production Checklist

Use [docs/production-readiness-checklist.md](docs/production-readiness-checklist.md) before advertising a deployment as production-ready.

---

## Contributing

Contributions are welcome. The codebase follows a clean module boundary — each of the five memory modules lives in `mnemos/modules/` and implements the same `MemoryModule` interface.

```bash
git clone https://github.com/anthony-maio/mnemos
cd mnemos
pip install -e '.[dev]'
pytest
```

**Good first contributions:**
- Neo4j backend for `SpreadingActivation` (the storage interface is ready, the backend isn't)
- Weaviate storage backend
- Proceduralization quality improvements in `SleepDaemon`
- Benchmarks comparing retrieval quality against standard RAG baselines

Please open an issue before starting large changes. Keep PRs focused — one feature or fix per PR.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

If you use mnemos in research, please cite:

```bibtex
@software{maio2026mnemos,
  author       = {Maio, Anthony},
  title        = {mnemos: Biomimetic Memory Architectures for Large Language Models},
  year         = {2026},
  version      = {0.1.0},
  url          = {https://github.com/anthony-maio/mnemos},
  note         = {Implements surprisal-triggered encoding, memory reconsolidation,
                  affective routing, hippocampal-neocortical consolidation, and
                  spreading activation for LLM memory systems.}
}
```

---

*mnemos* (μνῆμος) — from Ancient Greek, meaning "mindful" or "remembering." The name of the daemon in Greek mythology associated with memory.

Built by [Anthony Maio](mailto:anthony.maio@gmail.com).
