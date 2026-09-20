## Mnemos Memory

Use Mnemos memory tools automatically on every substantial coding task in this repository.

1. At the start of every substantial user task, call `mnemos_retrieve` with a focused query tied to the task.
2. Keep retrieval in the active provider's project scope. For MCP tools whose schema exposes `current_scope` and `allowed_scopes`, set both to `project`; for provider-owned tools without those fields, omit them and let the host inject scope. Let the client resolve the registered project identity; never use a folder basename as `scope_id`. Include global or workspace only when the active schema supports those scopes and the task needs them.
3. During execution, use the active provider's durable-write tool, such as `mnemos_store` or `mnemos_remember`, only when the host exposes that tool and its schema supports the call, and only for durable facts:
   - stable user or maintainer preferences
   - project architecture decisions and rationale
   - environment and tooling facts that will matter again
   - recurring bug patterns and their fixes
4. Before finishing substantial work, call the active provider's consolidation tool only when the host exposes it and its schema supports the call, such as `mnemos_consolidate`; otherwise do not invent a tool call.
5. If a retrieved memory looks suspicious, call `mnemos_inspect`, then `mnemos_correct` with its fact ID and current expected_version. Use `mnemos_forget` for an explicit deletion.
6. Never store secrets, credentials, tokens, or one-off transient chatter.
7. If Mnemos tools are unavailable, continue normally without blocking work.
