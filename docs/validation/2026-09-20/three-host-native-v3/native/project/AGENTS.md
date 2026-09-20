## Mnemos Memory

Use Mnemos through MCP automatically on every substantial coding task in this repository.

1. At the start of every substantial user task, call `mnemos_retrieve` with a focused query tied to the task.
2. Use `current_scope=project` and `allowed_scopes=project`. Let the client resolve the enabled project's registered identity; never use a folder basename as `scope_id`. Include global or workspace only when the task needs those scopes.
3. During execution, call `mnemos_store` only for durable facts:
   - stable user or maintainer preferences
   - project architecture decisions and rationale
   - environment and tooling facts that will matter again
   - recurring bug patterns and their fixes
4. Before finishing substantial work, call `mnemos_consolidate`.
5. If a retrieved memory looks suspicious, call `mnemos_inspect`, then `mnemos_correct` with its fact ID and current expected_version. Use `mnemos_forget` for an explicit deletion.
6. Never store secrets, credentials, tokens, or one-off transient chatter.
7. If Mnemos MCP tools are unavailable, continue normally without blocking work.
