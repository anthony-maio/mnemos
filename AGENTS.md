# AGENTS.md

## Mnemos Memory

Use Mnemos through MCP automatically on every substantial coding task in this repository.

1. At the start of every substantial user task, call `mnemos_retrieve` with a focused query tied to the task.
2. Use `current_scope=project`, set `scope_id` to the current repository/workspace name, and include `allowed_scopes=project,global` unless the user asks for broader scope.
3. During execution, call `mnemos_store` only for durable facts:
   - stable user or maintainer preferences
   - project architecture decisions and rationale
   - environment and tooling facts that will matter again
   - recurring bug patterns and their fixes
4. Before finishing substantial work, call `mnemos_consolidate`.
5. If a retrieved memory looks suspicious, call `mnemos_inspect` before storing a correction.
6. Never store secrets, credentials, tokens, or one-off transient chatter.
7. If Mnemos MCP tools are unavailable, continue normally without blocking work.
