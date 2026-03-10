# AGENTS.md

## Mnemos Memory

Use Mnemos through MCP on every substantial coding task in this repository.

1. At task start, call `mnemos_retrieve` with a focused query.
2. Use `current_scope=project`, `scope_id=mnemos`, and `allowed_scopes=project,global`.
3. Call `mnemos_store` only for durable facts:
   - architecture decisions and rationale
   - stable tooling and environment facts
   - recurring bug patterns and their fixes
   - durable maintainer preferences that apply to this repo
4. Before finishing substantial work, call `mnemos_consolidate`.
5. Never store secrets, credentials, tokens, or transient chatter.

If Mnemos MCP tools are unavailable in the current host, continue normally without blocking work.
