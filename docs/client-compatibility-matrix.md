# Client compatibility and observed evidence

Updated September 17, 2026 for the unreleased functional-local-memory worktree. Implementation, protocol tests, and native conversation acceptance are separate claims.

| Client | Implemented integration | Observed evidence | Native conversation acceptance |
| --- | --- | --- | --- |
| Claude Code | Command hooks, transcript deltas, context injection, MCP tools | Native 2.1.191 seeded recall plus user and assistant capture observed with installed d17dd24; both events survive restart | Transcript-unavailable fallback verified; cross-project event-ID collision remains open. Full lifecycle matrix pending |
| Codex | Command lifecycle hooks, supported context output, MCP tools | Native CLI 0.146.0 dispatched a Windows marker through hooks.json; isolated wheel adapter and Windows shell each captured distinct prompts, supplied context, and survived restart | Native Mnemos automatic capture and recall unproven; marker cancellation inconclusive |
| Hermes Agent | Optional MemoryProvider and explicit Windows/WSL bridge | Installed Hermes 0.21.3 API calls and actual WSL -> Windows authenticated transport | Pending |
| Generic MCP stdio | MCP server as service client | Subprocess capture, job status, inspect, retrieve, correct, forget | Host-specific acceptance pending |
| Claude Desktop | MCP configuration | Shared protocol coverage; no current desktop conversation evidence | Pending |
| Cursor | MCP plus project rule | Configuration and policy generation tests | Pending |
| Windsurf / Cline | MCP examples | Documentation/configuration only | Pending |

The cross-host fixture uses Codex, Claude Code, and Hermes scope contexts over a real temporary HTTP service and SQLite store. It checks restart, correction preserving identity, forget/replay suppression, and a foreign-project canary. Its provider fixtures are synthetic. It does not launch three native conversations.

Exact evidence boundaries are in [local-memory-hosts.md](validation/local-memory-hosts.md) and the [native host smoke record](validation/2026-09-16/native-host-smoke.md). The [Hermes bridge record](validation/2026-09-16/hermes-bridge-smoke.md) documents installed API and transport checks. The two-week pilot is NOT RUN.

All clients need an enabled project, registered profile, and the same runtime descriptor. A repository basename is not a scope identity. Mock cognition is rejected by the normal configured service; use a real provider and input-specific readiness probe. See [functional local memory](functional-local-memory.md), [Codex setup](codex.md), and [Hermes setup](integrations/hermes.md).
