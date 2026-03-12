# MCP Client Compatibility Matrix

Last updated: **March 12, 2026**

## Tier Definitions

- **Tier 1 (release-blocking)**: real end-to-end validation in a supported host workflow.
- **Tier 2 (documented)**: config/docs validated, but not yet promoted to release-blocking E2E support.

## Matrix

| Client | Tier | Status | Tested Config | Notes |
|---|---|---|---|---|
| Claude Code | Tier 1 | Supported | plugin + direct `mnemos-mcp` | Primary target; SQLite starter remains the default, and hook-based auto-capture has now been validated on a Neo4j-backed shared config. |
| Claude Desktop | Tier 1 | Supported | `docs/mcp-configs/claude-desktop.json` | Tested with `python -m mnemos.mcp_server` and local-safe mock/simple defaults. |
| Generic MCP stdio host | Tier 1 | Supported | `docs/mcp-configs/generic-stdio.json` | Tested with `python -m mnemos.mcp_server`; swap provider env for production. |
| Codex | Tier 2 | Documented | `docs/mcp-configs/codex.json` + `docs/codex.md` | Supported through MCP + `AGENTS.md`; shared `MNEMOS_CONFIG_PATH` resolution to Neo4j is validated, but daily-use E2E promotion is still pending. |
| Cursor | Tier 2 | Best effort + Antigravity pack | `docs/mcp-configs/cursor.json` + `docs/cursor-antigravity.md` | Repo-level `.cursor/mcp.json` shared-config resolution to Neo4j is covered in tests; no shipped auto-capture path yet. |
| Windsurf | Tier 2 | Best effort | `docs/mcp-configs/windsurf.json` | Smoke config validated in CI (non-blocking). |
| Cline | Tier 2 | Best effort | `docs/mcp-configs/cline.json` | Smoke config validated in CI (non-blocking). |

## Known Caveats

- Tier 2 hosts may differ in MCP UX and timeout defaults.
- Tier 1 checked-in config files are intentionally minimal and runnable from source or an installed package; replace `mock`/`simple` providers with `ollama`, `openai`, or `openclaw` for real usage.
- Claude Code is the only host with shipped deterministic auto-capture hooks today.
- Codex currently depends on the repo's `AGENTS.md` to consistently trigger `mnemos_retrieve`, `mnemos_store`, and `mnemos_consolidate`.
- Cursor still depends on project instructions / Antigravity policy text for automatic Mnemos tool usage.
- For highest retrieval quality, avoid `MNEMOS_EMBEDDING_PROVIDER=simple` in production profiles.
- Qdrant remote profiles require `qdrant-client` extra and reachable network endpoint.
