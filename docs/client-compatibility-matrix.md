# MCP Client Compatibility Matrix

Last updated: **March 8, 2026**

## Tier Definitions

- **Tier 1 (release-blocking)**: real end-to-end validation in a supported host workflow.
- **Tier 2 (documented)**: config/docs validated, but not yet promoted to release-blocking E2E support.

## Matrix

| Client | Tier | Status | Tested Config | Notes |
|---|---|---|---|---|
| Claude Code | Tier 1 | Supported | plugin + direct `mnemos-mcp` | Primary target; plugin defaults to SQLite starter profile. |
| Claude Desktop | Tier 1 | Supported | `docs/mcp-configs/claude-desktop.json` | Tested with `python -m mnemos.mcp_server` and local-safe mock/simple defaults. |
| Generic MCP stdio host | Tier 1 | Supported | `docs/mcp-configs/generic-stdio.json` | Tested with `python -m mnemos.mcp_server`; swap provider env for production. |
| Codex | Tier 2 | Documented | `docs/mcp-configs/codex.json` + `docs/codex.md` | Supported through MCP + `AGENTS.md`; promote to Tier 1 after verified end-to-end daily-use validation. |
| Cursor | Tier 2 | Best effort + Antigravity pack | `docs/mcp-configs/cursor.json` + `docs/cursor-antigravity.md` | Smoke config validated in CI (non-blocking). |
| Windsurf | Tier 2 | Best effort | `docs/mcp-configs/windsurf.json` | Smoke config validated in CI (non-blocking). |
| Cline | Tier 2 | Best effort | `docs/mcp-configs/cline.json` | Smoke config validated in CI (non-blocking). |

## Known Caveats

- Tier 2 hosts may differ in MCP UX and timeout defaults.
- Tier 1 checked-in config files are intentionally minimal and runnable from source or an installed package; replace `mock`/`simple` providers with `ollama`, `openai`, or `openclaw` for real usage.
- Codex currently depends on the repo's `AGENTS.md` to consistently trigger `mnemos_retrieve`, `mnemos_store`, and `mnemos_consolidate`.
- For highest retrieval quality, avoid `MNEMOS_EMBEDDING_PROVIDER=simple` in production profiles.
- Qdrant remote profiles require `qdrant-client` extra and reachable network endpoint.
