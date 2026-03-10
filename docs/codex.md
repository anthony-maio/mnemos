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
mnemos-cli antigravity codex
```

Add the generated policy to your repo-level `AGENTS.md` if you want to customize it. The minimum workflow is:

1. Start each task with `mnemos_retrieve`.
2. Use `current_scope=project` and `scope_id` equal to the repo name.
3. Store only durable facts with `mnemos_store`.
4. Finish substantial work with `mnemos_consolidate`.

Ready-to-copy snippet:

```md
## Mnemos Memory

Use Mnemos through MCP on every substantial coding task.

1. At task start, call `mnemos_retrieve` with a focused query.
2. Use `current_scope=project`, `scope_id=<repo-name>`, and `allowed_scopes=project,global`.
3. Call `mnemos_store` only for durable facts:
   - project architecture decisions
   - stable user or team preferences
   - tooling and environment facts
   - recurring bug patterns and fixes
4. Before finishing substantial work, call `mnemos_consolidate`.
5. Never store secrets, credentials, tokens, or transient chatter.
```

## 4. Recommended v1 operating mode

- Default store: SQLite
- Upgrade to Qdrant only when benchmarked latency or dataset size justifies it
- Treat `SimpleEmbeddingProvider` as development-only
- Audit for legacy unscoped chunks with `mnemos-cli doctor` before claiming v1 isolation

## Validation Loop

Run at least one real Codex session through:

1. `mnemos_retrieve`
2. normal coding work
3. `mnemos_store`
4. `mnemos_consolidate`

Until that flow is verified in daily use, Codex remains documented as Tier 2 in [client-compatibility-matrix.md](client-compatibility-matrix.md).
