# Mnemos for Codex

Codex support in Mnemos is MCP-first. There is no separate Codex plugin format for v1.

## Status

- Current compatibility tier: Tier 2
- Install surface: MCP server + repo-level `AGENTS.md`
- Promotion to Tier 1 requires verified end-to-end daily-use validation, not just config checks

## 1. Install Mnemos with MCP support

```bash
pip install "mnemos-memory[mcp]"
```

Then launch the control plane:

```bash
mnemos ui
```

Use it to:

- choose `Dev` or `Pro` onboarding mode
- configure your provider and storage
- preview/apply the Codex MCP config
- run the built-in smoke check

For reliable scoped memory, prefer a real embedding provider and SQLite starter storage:

```bash
$env:MNEMOS_LLM_PROVIDER="openclaw"
$env:MNEMOS_EMBEDDING_PROVIDER="openclaw"
$env:MNEMOS_STORE_TYPE="sqlite"
$env:MNEMOS_SQLITE_PATH=".mnemos/memory.db"
```

If you already have a shared always-on Neo4j deployment, point the same global config at it instead:

```toml
[storage]
type = "neo4j"
neo4j_uri = "bolt://192.168.86.41:7687"
neo4j_database = "neo4j"
neo4j_label = "MnemosMemoryChunk"

[providers.neo4j]
username = "neo4j"
password = "..."
```

## 2. Register the MCP server for Codex

Use [mcp-configs/codex.json](mcp-configs/codex.json) as the starting point. The checked-in template stays runnable from a source checkout with `python -m mnemos.mcp_server`, while the control plane writes the installed-package form automatically.

```json
{
  "mcpServers": {
    "mnemos": {
      "command": "python",
      "args": ["-m", "mnemos.mcp_server"],
      "env": {
        "MNEMOS_CONFIG_PATH": "~/.config/Mnemos/mnemos.toml"
      }
    }
  }
}
```

## 3. Add an `AGENTS.md` memory policy

This repo now ships a ready-to-use example at [AGENTS.md](../AGENTS.md).

Generate a Codex-specific policy block:

```bash
mnemos-cli antigravity codex --target codex-agents
```

Add the generated policy to your repo-level `AGENTS.md` if you want to customize it. The minimum workflow is:

1. Start each task with `mnemos_retrieve`.
2. Use `current_scope=project` and `scope_id` equal to the repo name.
3. Store only durable facts with `mnemos_store`.
4. Finish substantial work with `mnemos_consolidate`.
5. Use `mnemos_inspect` when you need to verify why a memory exists or how it changed.

Ready-to-copy snippet:

```md
## Mnemos Memory

Use Mnemos through MCP automatically on every substantial coding task in this repository.

1. At the start of every substantial user task, call `mnemos_retrieve` with a focused query tied to the task.
2. Use `current_scope=project`, set `scope_id` to the current repository name, and include `allowed_scopes=project,global` unless broader scope is needed.
3. During execution, call `mnemos_store` only for durable facts:
   - stable user or maintainer preferences
   - project architecture decisions and rationale
   - environment and tooling facts that will matter again
   - recurring bug patterns and their fixes
4. Before finishing substantial work, call `mnemos_consolidate`.
5. If a retrieved memory looks suspicious, call `mnemos_inspect` before storing a correction.
6. Never store secrets, credentials, tokens, or transient chatter.
7. If Mnemos MCP tools are unavailable in the current host, continue normally without blocking work.
```

## 3b. Optional Codex Automations

Codex Automations are useful for scheduled Mnemos maintenance, not chat-session capture. Generate a prompt you can paste into a Codex Automation with:

```bash
mnemos-cli antigravity codex --target codex-automation
```

Use it for recurring hygiene checks such as `mnemos-cli doctor` and `mnemos-cli stats`. Do not treat Codex Automations as a substitute for host lifecycle hooks.

## 4. Recommended v1 operating mode

- Default store: SQLite
- Upgrade to Qdrant only when benchmarked latency or dataset size justifies it
- Neo4j is a valid advanced/shared deployment path when you already operate it; migrate existing memories with `mnemos-cli migrate-store --source-store qdrant --target-store neo4j`
- Treat `SimpleEmbeddingProvider` as development-only
- Audit for legacy unscoped chunks with `mnemos-cli doctor` before claiming v1 isolation

## Validation Loop

Run at least one real Codex session through:

1. `mnemos_retrieve`
2. normal coding work
3. `mnemos_store`
4. `mnemos_consolidate`

Until that flow is verified in daily use, Codex remains documented as Tier 2 in [client-compatibility-matrix.md](client-compatibility-matrix.md).

## Neo4j-backed status

Codex now validates cleanly against a shared `MNEMOS_CONFIG_PATH` that points at Neo4j. That means the MCP server can use the same always-on backend as Claude Code or Cursor. It does **not** mean Codex has automatic capture hooks yet; the current validated loop is still soft-auto instruction plus MCP:

1. `mnemos_retrieve`
2. normal coding work
3. `mnemos_store`
4. `mnemos_consolidate`

## Inspectability in Codex

If a retrieved memory looks wrong, inspect it before storing more state:

1. call `mnemos_inspect`
2. verify scope and provenance
3. review revision history if the fact was reconsolidated
4. only then decide whether to store a correcting fact or forget the chunk
