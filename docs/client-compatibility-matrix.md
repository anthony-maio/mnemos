# MCP Client Compatibility Matrix

Last updated: **March 12, 2026**

## Tier Definitions

- **Tier 1 (release-blocking)**: real end-to-end validation in a supported host workflow.
- **Tier 2 (documented)**: config/docs validated, but not yet promoted to release-blocking E2E support.

## Matrix

| Client | Tier | Status | Tested Config | Notes |
|---|---|---|---|---|
| Claude Code | Tier 1 | Supported | plugin + direct `mnemos-mcp` | Primary target; the default local SQLite path is the shipped setup, and hook-based auto-capture is validated there. |
| Claude Desktop | Tier 1 | Supported | `docs/mcp-configs/claude-desktop.json` | Tested with `python -m mnemos.mcp_server` and local-safe mock/simple defaults. |
| Generic MCP stdio host | Tier 1 | Supported | `docs/mcp-configs/generic-stdio.json` | Tested with `python -m mnemos.mcp_server`; swap provider env for production. |
| Codex | Tier 2 | Documented | `docs/mcp-configs/codex.json` + `docs/codex.md` | Supported through MCP + a stronger `AGENTS.md` pack; the shared SQLite-backed `MNEMOS_CONFIG_PATH` path is validated, and optional Codex Automations can run maintenance checks, but daily-use E2E promotion is still pending. |
| Cursor | Tier 2 | Best effort + Antigravity pack | `docs/mcp-configs/cursor.json` + `docs/cursor-antigravity.md` | Repo-level `.cursor/mcp.json` plus `.cursor/rules/mnemos-memory.mdc` soft-auto setup is now covered in tests; no shipped host-hook auto-capture path yet. |
| Windsurf | Tier 2 | Best effort | `docs/mcp-configs/windsurf.json` | Smoke config validated in CI (non-blocking). |
| Cline | Tier 2 | Best effort | `docs/mcp-configs/cline.json` | Smoke config validated in CI (non-blocking). |

## Known Caveats

- Tier 2 hosts may differ in MCP UX and timeout defaults.
- Tier 1 checked-in config files are intentionally minimal and runnable from source or an installed package; replace `mock`/`simple` providers with `ollama`, `openai`, or `openclaw` for real usage.
- Claude Code is the only host with shipped deterministic auto-capture hooks today.
- Codex currently depends on the repo's `AGENTS.md` to consistently trigger `mnemos_retrieve`, `mnemos_store`, and `mnemos_consolidate`; optional Codex Automations are maintenance helpers, not chat hooks.
- Cursor still depends on project instructions / `.cursor/rules` for automatic Mnemos tool usage.
- For highest retrieval quality, avoid `MNEMOS_EMBEDDING_PROVIDER=simple` in production profiles.
