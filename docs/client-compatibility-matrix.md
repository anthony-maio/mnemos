# MCP Client Compatibility Matrix

Last updated: **November 8, 2026**

The 0.6.1 patch candidate is unpublished. Current installed validation uses
Windows Python 3.11.13, a mock LLM and simple embeddings. The compatibility tests
launch the configured MCP processes through a generic Python MCP client; using
a Claude configuration file does not run Claude Code or Claude Desktop.
The tiers below remain release requirements, not evidence that all have passed.

## Tier Definitions

- **Tier 1 (release-blocking)**: real end-to-end validation in a supported host workflow.
- **Tier 2 (documented)**: config/docs validated, but not yet promoted to release-blocking E2E support.

## Matrix

| Client | Tier | Status | Tested Config | Notes |
|---|---|---|---|---|
| Claude Code | Tier 1 | Automated plugin transport passes; native acceptance pending | plugin wrapper + direct MCP process | Current candidate passes a stdio roundtrip through the external plugin wrapper. Interactive host behavior and hook-based capture are not established by that test. |
| Claude Desktop | Tier 1 | Automated config transport passes; native acceptance pending | `docs/mcp-configs/claude-desktop.json` | The configured process passes a generic-client roundtrip. Registration and use inside Claude Desktop remain unverified for this candidate. |
| Generic MCP stdio host | Tier 1 | Automated roundtrip passes | `docs/mcp-configs/generic-stdio.json` | Installed package tested with the Python MCP client and mock/simple providers; this does not establish real-provider readiness. |
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
