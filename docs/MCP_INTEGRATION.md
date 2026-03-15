# Mnemos MCP Integration Guide

## What is MCP?

The Model Context Protocol (MCP) lets agentic platforms — Claude Code, Codex, Cursor, Windsurf, Cline, and others — discover and call external tools in a standardized way. Mnemos ships an MCP server that exposes its full memory system as tools any agent can use.

## Quick Setup

### 1. Install with MCP support

```bash
pip install 'mnemos-memory[mcp]'
```

### 2. Launch the control plane

```bash
mnemos ui
```

The control plane is now the primary onboarding path. It lets you:

- save a canonical global `mnemos.toml`
- import existing preview-user env/config setups
- preview and apply Claude Code, Cursor, and Codex host configs
- write the Codex repo `AGENTS.md` memory block as part of Codex host setup
- run health and smoke checks before daily use

### 3. Configure your agent

The generated host configs should point Mnemos at the canonical config file with `MNEMOS_CONFIG_PATH`, rather than embedding provider secrets into every host file.

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

#### Claude Code / Claude Desktop

Add to `~/.claude/claude_desktop_config.json` (or your project's `.claude` config):

```json
{
  "mcpServers": {
    "mnemos": {
      "command": "mnemos-mcp",
      "env": {
        "MNEMOS_CONFIG_PATH": "~/.config/Mnemos/mnemos.toml"
      }
    }
  }
}
```

If this repo's Claude plugin wrapper is available, the control plane prefers that path automatically.

#### Cursor

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "mnemos": {
      "command": "mnemos-mcp",
      "env": {
        "MNEMOS_CONFIG_PATH": "~/.config/Mnemos/mnemos.toml"
      }
    }
  }
}
```

For project-level soft-auto behavior, add a Cursor rule as well:

```bash
mnemos-cli antigravity cursor --target cursor-rule --write .cursor/rules/mnemos-memory.mdc
```

The control plane preview/apply path now writes both the MCP config and the Cursor rule file.

#### Codex

Codex uses the same MCP server surface, with repo-level guidance in `AGENTS.md` to make memory use consistent.

- Starting config: [mcp-configs/codex.json](mcp-configs/codex.json)
- Setup guide: [codex.md](codex.md)
- Optional local skill: [skills/mnemos-codex/SKILL.md](../skills/mnemos-codex/SKILL.md)
- Generate the repo `AGENTS.md` block with `mnemos-cli antigravity codex --target codex-agents`
- Generate an optional Codex Automation prompt with `mnemos-cli antigravity codex --target codex-automation`

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

### 3. Done — your agent now has scoped agent memory

The agent will automatically discover these tools:

| Tool | What it does |
|------|-------------|
| `mnemos_store` | Store a memory through the surprisal gate + affective tagging pipeline (supports `scope` + `scope_id`) |
| `mnemos_retrieve` | Retrieve memories with spreading activation + emotional re-ranking (supports `current_scope`, `scope_id`, `allowed_scopes`) |
| `mnemos_consolidate` | Trigger sleep consolidation (episodic → semantic compression) |
| `mnemos_forget` | Delete a specific memory |
| `mnemos_stats` | System-wide statistics from all modules |
| `mnemos_health` | Profile readiness and dependency diagnostics |
| `mnemos_inspect` | Full details on a specific memory, including provenance, revision history, and graph context |
| `mnemos_list` | List all stored memories |

Compatibility + contract docs:
- [mcp-transport-contract.md](mcp-transport-contract.md)
- [client-compatibility-matrix.md](client-compatibility-matrix.md)
- [codex.md](codex.md)
- [cursor-antigravity.md](cursor-antigravity.md)

---

## Configuration

The canonical runtime config is now `mnemos.toml`, typically stored at the platform config directory:

- Windows: `%APPDATA%/Mnemos/mnemos.toml`
- macOS: `~/Library/Application Support/Mnemos/mnemos.toml`
- Linux: `$XDG_CONFIG_HOME/Mnemos/mnemos.toml` or `~/.config/Mnemos/mnemos.toml`

Optional per-project overrides live at `.mnemos/mnemos.toml`.

Precedence:

1. CLI flags
2. environment variables
3. project `.mnemos/mnemos.toml`
4. global `mnemos.toml`
5. built-in defaults

Environment variables still work and override config files. They are now the advanced/manual path:

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMOS_CONFIG_PATH` | platform config dir | Canonical global config path used by host integrations |
| `MNEMOS_LLM_PROVIDER` | `mock` | LLM backend: `mock`, `ollama`, `openai`, `openclaw`, or `openrouter` |
| `MNEMOS_LLM_MODEL` | `llama3` | Model name for the LLM provider |
| `MNEMOS_OLLAMA_URL` | `http://localhost:11434` | Ollama API URL |
| `MNEMOS_OPENAI_API_KEY` | — | Required if provider is `openai` |
| `MNEMOS_OPENAI_URL` | `https://api.openai.com/v1` | OpenAI-compatible URL |
| `MNEMOS_OPENCLAW_API_KEY` | — | OpenClaw API key (or fallback to `MNEMOS_OPENAI_API_KEY`) |
| `MNEMOS_OPENCLAW_URL` | — | OpenClaw API URL (or fallback to `MNEMOS_OPENAI_URL`) |
| `MNEMOS_OPENROUTER_API_KEY` | — | Required if provider is `openrouter` |
| `MNEMOS_OPENROUTER_URL` | `https://openrouter.ai/api/v1` | OpenRouter API URL |
| `MNEMOS_EMBEDDING_PROVIDER` | inferred from `MNEMOS_LLM_PROVIDER`, else `simple` | Embedding backend: `simple`, `ollama`, `openai`, `openclaw`, or `openrouter` |
| `MNEMOS_EMBEDDING_MODEL` | provider-dependent | Embedding model name |
| `MNEMOS_STORE_TYPE` | `memory` | Storage backend: `memory` or `sqlite` |
| `MNEMOS_SQLITE_PATH` | `mnemos_memory.db` | SQLite database path |
| `MNEMOS_SURPRISAL_THRESHOLD` | `0.3` | Surprisal gate sensitivity (0-1) |
| `MNEMOS_MEMORY_SAFETY_ENABLED` | `true` | Enable shared memory write safety firewall |
| `MNEMOS_MEMORY_SECRET_ACTION` | `block` | Secret handling mode: `allow`, `redact`, or `block` |
| `MNEMOS_MEMORY_PII_ACTION` | `redact` | PII handling mode: `allow`, `redact`, or `block` |
| `MNEMOS_MEMORY_CAPTURE_MODE` | `all` | Ingestion mode: `all`, `manual_only`, or `hooks_only` |
| `MNEMOS_MEMORY_RETENTION_TTL_DAYS` | `0` | Prune memories older than N days (`0` disables TTL pruning) |
| `MNEMOS_MEMORY_MAX_CHUNKS_PER_SCOPE` | `0` | Max chunks per `(scope, scope_id)` partition (`0` disables cap) |
| `MNEMOS_EMBEDDING_DIM` | `384` | Embedding vector dimension |
| `MNEMOS_DEBUG` | `false` | Enable debug logging |

Backward-compatible aliases:
- `MNEMOS_STORAGE` works as an alias for `MNEMOS_STORE_TYPE`
- `MNEMOS_DB_PATH` works as an alias for `MNEMOS_SQLITE_PATH`

If `MNEMOS_EMBEDDING_PROVIDER` is not set, Mnemos infers it from `MNEMOS_LLM_PROVIDER` for `ollama`, `openai`, `openclaw`, and `openrouter`.

### Provider recommendations

- **Development/testing**: Use `mock` — zero dependencies, deterministic
- **Local deployment**: Use `ollama` with a small model like `llama3:8b`
- **Production**: Use `openai`, `openclaw`, `openrouter`, or any OpenAI-compatible API

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

## Inspectability

Mnemos now exposes one shared inspectability surface across CLI, MCP, and the control plane:

- CLI: `mnemos inspect <chunk-id>`
- MCP: `mnemos_inspect`
- Control plane: open the Memory panel and select a chunk

The current inspector shows:

- scope and scope ID
- provenance such as storage source, ingest channel, and encoding reason
- revision history for reconsolidated memories
- graph neighbor count and top neighbors
- raw metadata for debugging

Rollback is intentionally not part of this first inspectability slice yet.

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

# Store project/workspace/global-scoped memories
mnemos-cli store "Use pnpm in this repository" --scope project --scope-id repo-alpha
mnemos-cli store "In this workspace, default to black + mypy" --scope workspace --scope-id consulting-workspace
mnemos-cli store "Prefer concise status updates" --scope global

# Retrieve scoped memories (project + global)
mnemos-cli retrieve "coding preferences" --current-scope project --scope-id repo-alpha --allowed-scopes project,global

# Trigger consolidation
mnemos-cli consolidate

# View statistics
mnemos-cli stats

# Audit memory
mnemos-cli list --scope project --scope-id repo-alpha --limit 20
mnemos-cli search "terraform" --scope project --scope-id repo-alpha
mnemos-cli export --scope project --scope-id repo-alpha --format jsonl --output .mnemos-export.jsonl
mnemos-cli purge --scope project --scope-id repo-alpha --older-than-days 30 --dry-run
mnemos-cli purge --scope project --scope-id repo-alpha --older-than-days 30 --yes

# Readiness diagnostics
mnemos-cli doctor

# Threshold-aware readiness diagnostics
mnemos-cli doctor --chunk-threshold 5000 --latency-p95-threshold-ms 250 --observed-p95-ms 180

# One-command profile generation
mnemos-cli profile default --format dotenv --write .mnemos.profile.env

# Dry-run migration into SQLite
mnemos-cli migrate-store --source-store qdrant --target-store sqlite --dry-run

# Execute migration into SQLite
mnemos-cli migrate-store --source-store qdrant --target-store sqlite

# Generate a Cursor rule for soft-auto memory use
mnemos-cli antigravity cursor --target cursor-rule --write .cursor/rules/mnemos-memory.mdc
```

The CLI defaults to SQLite storage (persistent across sessions), unlike the MCP server which defaults to in-memory.

### Hooks for automatic memory

Copy the example hooks from `docs/claude-code-hooks.json` into your Claude Code settings. The included hooks:

- **UserPromptSubmit**: Automatically ingest high-signal user prompts into Mnemos (`mnemos-cli autostore-hook UserPromptSubmit`).
- **PostToolUse**: Automatically ingest high-signal tool failures into Mnemos (`mnemos-cli autostore-hook PostToolUse`).
- **PreCompact**: Before context compression, consolidate episodic memory so important context is saved to long-term storage before it's lost.
- **Stop**: After Claude finishes responding, trigger consolidation to compress session learnings.

The auto-store hook path is deterministic and conservative:
- skips low-signal prompts
- skips sensitive content (tokens/secrets/key material patterns)
- only stores tool outputs when failure/error patterns are present

This Claude hook path has been validated on the canonical SQLite-backed `MNEMOS_CONFIG_PATH` setup as of March 15, 2026. Codex and Cursor can point at the same shared config, but they still rely on explicit Mnemos tool usage or host instruction policies today rather than shipped hook capture.

Treat hard auto-capture as host-dependent outside Claude Code. Codex, Cursor, OpenClaw, and generic MCP hosts can use soft-auto instruction packs today, but they should not be marketed as having built-in hook parity unless that host actually exposes verified lifecycle hooks.

### Environment variables

Both `mnemos-mcp` and `mnemos-cli` read the same `MNEMOS_*` environment variables. Set them in your shell profile for consistent behavior:

```bash
export MNEMOS_LLM_PROVIDER=ollama
export MNEMOS_LLM_MODEL=llama3
export MNEMOS_STORE_TYPE=sqlite
export MNEMOS_SQLITE_PATH=~/.mnemos/memory.db
```
