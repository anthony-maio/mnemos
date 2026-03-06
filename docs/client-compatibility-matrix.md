# MCP Client Compatibility Matrix

Last updated: **March 6, 2026**

## Tier Definitions

- **Tier 1 (contractual)**: release-blocking compatibility target.
- **Tier 2 (best effort)**: smoke-tested and documented; non-blocking for release.

## Matrix

| Client | Tier | Status | Tested Config | Notes |
|---|---|---|---|---|
| Claude Code | Tier 1 | Supported | plugin + direct `mnemos-mcp` | Primary target; plugin defaults to SQLite starter profile. |
| Claude Desktop | Tier 1 | Supported | `docs/mcp-configs/claude-desktop.json` | Uses standard MCP `stdio` registration. |
| Generic MCP stdio host | Tier 1 | Supported | `docs/mcp-configs/generic-stdio.json` | Tool/resource contract is host-agnostic. |
| Cursor | Tier 2 | Best effort | `docs/mcp-configs/cursor.json` | Smoke config validated in CI (non-blocking). |
| Windsurf | Tier 2 | Best effort | `docs/mcp-configs/windsurf.json` | Smoke config validated in CI (non-blocking). |
| Cline | Tier 2 | Best effort | `docs/mcp-configs/cline.json` | Smoke config validated in CI (non-blocking). |

## Known Caveats

- Tier 2 hosts may differ in MCP UX and timeout defaults.
- For highest retrieval quality, avoid `MNEMOS_EMBEDDING_PROVIDER=simple` in production profiles.
- Qdrant remote profiles require `qdrant-client` extra and reachable network endpoint.
