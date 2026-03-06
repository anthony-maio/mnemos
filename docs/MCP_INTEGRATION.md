# Mnemos MCP Integration Guide

## What is MCP?

The Model Context Protocol (MCP) lets agentic platforms — Claude Code, Cursor, Windsurf, Cline, and others — discover and call external tools in a standardized way. Mnemos ships an MCP server that exposes its full biomimetic memory system as tools any agent can use.

## Quick Setup

### 1. Install with MCP support

```bash
pip install 'mnemos-memory[mcp]'
```

### 2. Configure your agent

### Plugin-first install for Claude Code

If you want ClaudeMem-style plugin installation, use the built-in plugin packaging in this repo:

```text
/plugin marketplace add anthony-maio/mnemos
/plugin install mnemos-memory@mnemos-marketplace
```

The plugin manifest auto-registers the Mnemos MCP server and runs a bootstrap wrapper that:
- creates a local `.claude-plugin/.venv`
- installs Mnemos with MCP extras
- launches `mnemos.mcp_server` over stdio
- defaults to persistent SQLite storage
 - installs the `qdrant` extra automatically if `MNEMOS_STORE_TYPE=qdrant`

#### Claude Code / Claude Desktop

Add to `~/.claude/claude_desktop_config.json` (or your project's `.claude` config):

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

#### Cursor

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

#### Windsurf

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

#### Any MCP-compatible agent (stdio transport)

```bash
# Just run the server — the agent connects via stdin/stdout
mnemos-mcp

# Or via Python module
python -m mnemos.mcp_server
```

### 3. Done — your agent now has biomimetic memory

The agent will automatically discover these tools:

| Tool | What it does |
|------|-------------|
| `mnemos_store` | Store a memory through the surprisal gate + affective tagging pipeline |
| `mnemos_retrieve` | Retrieve memories with spreading activation + emotional re-ranking |
| `mnemos_consolidate` | Trigger sleep consolidation (episodic → semantic compression) |
| `mnemos_forget` | Delete a specific memory |
| `mnemos_stats` | System-wide statistics from all modules |
| `mnemos_health` | Profile readiness and dependency diagnostics |
| `mnemos_inspect` | Full details on a specific memory |
| `mnemos_list` | List all stored memories |

Compatibility + contract docs:
- [mcp-transport-contract.md](mcp-transport-contract.md)
- [client-compatibility-matrix.md](client-compatibility-matrix.md)
- [profiles/starter-sqlite.md](profiles/starter-sqlite.md)
- [profiles/local-performance-embedded-qdrant.md](profiles/local-performance-embedded-qdrant.md)
- [profiles/scale-external-qdrant.md](profiles/scale-external-qdrant.md)

---

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMOS_LLM_PROVIDER` | `mock` | LLM backend: `mock`, `ollama`, `openai`, or `openclaw` |
| `MNEMOS_LLM_MODEL` | `llama3` | Model name for the LLM provider |
| `MNEMOS_OLLAMA_URL` | `http://localhost:11434` | Ollama API URL |
| `MNEMOS_OPENAI_API_KEY` | — | Required if provider is `openai` |
| `MNEMOS_OPENAI_URL` | `https://api.openai.com/v1` | OpenAI-compatible URL |
| `MNEMOS_OPENCLAW_API_KEY` | — | OpenClaw API key (or fallback to `MNEMOS_OPENAI_API_KEY`) |
| `MNEMOS_OPENCLAW_URL` | — | OpenClaw API URL (or fallback to `MNEMOS_OPENAI_URL`) |
| `MNEMOS_EMBEDDING_PROVIDER` | inferred from `MNEMOS_LLM_PROVIDER`, else `simple` | Embedding backend: `simple`, `ollama`, `openai`, or `openclaw` |
| `MNEMOS_EMBEDDING_MODEL` | provider-dependent | Embedding model name |
| `MNEMOS_STORE_TYPE` | `memory` | Storage backend: `memory`, `sqlite`, or `qdrant` |
| `MNEMOS_SQLITE_PATH` | `mnemos_memory.db` | SQLite database path |
| `MNEMOS_QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `MNEMOS_QDRANT_API_KEY` | — | Optional Qdrant API key |
| `MNEMOS_QDRANT_PATH` | — | Local embedded Qdrant path (overrides URL) |
| `MNEMOS_QDRANT_COLLECTION` | `mnemos_memory` | Qdrant collection name |
| `MNEMOS_QDRANT_VECTOR_SIZE` | — | Optional fixed vector size |
| `MNEMOS_SURPRISAL_THRESHOLD` | `0.3` | Surprisal gate sensitivity (0-1) |
| `MNEMOS_EMBEDDING_DIM` | `384` | Embedding vector dimension |
| `MNEMOS_DEBUG` | `false` | Enable debug logging |

Backward-compatible aliases:
- `MNEMOS_STORAGE` works as an alias for `MNEMOS_STORE_TYPE`
- `MNEMOS_DB_PATH` works as an alias for `MNEMOS_SQLITE_PATH`

If `MNEMOS_EMBEDDING_PROVIDER` is not set, Mnemos infers it from `MNEMOS_LLM_PROVIDER` for `ollama`, `openai`, and `openclaw`.

### Provider recommendations

- **Development/testing**: Use `mock` — zero dependencies, deterministic
- **Local deployment**: Use `ollama` with a small model like `llama3:8b`
- **Production**: Use `openai`, `openclaw`, or any OpenAI-compatible API (vLLM, Together, etc.)

---

## Example Usage in Claude Code

Once configured, you can say things like:

```
"Remember that the user prefers dark mode and uses Neovim as their editor."
→ Agent calls mnemos_store

"What do you know about my coding preferences?"
→ Agent calls mnemos_retrieve

"Consolidate what you've learned from our conversation."
→ Agent calls mnemos_consolidate

"Forget the memory about the old server IP."
→ Agent calls mnemos_forget
```

The agent's memory now works like a brain: it filters the mundane, tags emotions, updates stale facts on recall, and compresses episodic experience into lasting knowledge.

---

## Architecture

The MCP server wraps the full `MnemosEngine` pipeline:

```
Encode (mnemos_store):
  Input → SurprisalGate → AffectiveRouter → SpreadingActivation → Store

Retrieve (mnemos_retrieve):
  Query → SpreadingActivation → AffectiveRouter → MutableRAG → Results

Consolidate (mnemos_consolidate):
  Idle → SleepDaemon → Facts extracted → Episodes pruned
```

Each module is inspired by a specific neuroscience mechanism:
- **SurprisalGate**: Predictive coding (only encode prediction errors)
- **MutableRAG**: Memory reconsolidation (facts evolve on recall)
- **AffectiveRouter**: Amygdala-mediated state-dependent retrieval
- **SleepDaemon**: Hippocampal-neocortical transfer during sleep
- **SpreadingActivation**: Collins & Loftus associative networks

---

## Claude Code: Deep Integration

Beyond the MCP server, mnemos provides a CLI and hooks configuration for automatic memory operations.

### Register the MCP server

```bash
# Available to all projects (user scope)
claude mcp add --scope user mnemos -- mnemos-mcp

# Or project-specific
claude mcp add mnemos -- mnemos-mcp
```

### CLI for hooks and automation

The `mnemos-cli` command provides shell-friendly operations:

```bash
# Store a memory
mnemos-cli store "The user prefers Python 3.12 and uses pytest"

# Retrieve memories
mnemos-cli retrieve "testing preferences" --top-k 5

# Trigger consolidation
mnemos-cli consolidate

# View statistics
mnemos-cli stats

# Readiness diagnostics
mnemos-cli doctor
```

The CLI defaults to SQLite storage (persistent across sessions), unlike the MCP server which defaults to in-memory.

### Hooks for automatic memory

Copy the example hooks from `docs/claude-code-hooks.json` into your Claude Code settings. The included hooks:

- **PreCompact**: Before context compression, consolidate episodic memory so important context is saved to long-term storage before it's lost.
- **Stop**: After Claude finishes responding, trigger consolidation to compress session learnings.

### Environment variables

Both `mnemos-mcp` and `mnemos-cli` read the same `MNEMOS_*` environment variables. Set them in your shell profile for consistent behavior:

```bash
export MNEMOS_LLM_PROVIDER=ollama
export MNEMOS_LLM_MODEL=llama3
export MNEMOS_STORE_TYPE=sqlite
export MNEMOS_SQLITE_PATH=~/.mnemos/memory.db
```
