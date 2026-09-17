# Client compatibility and observed evidence

Updated September 17, 2026 for the unreleased functional-local-memory worktree. Implementation, protocol tests, and native conversation acceptance are separate claims.

| Client | Implemented integration | Observed evidence | Native conversation acceptance |
| --- | --- | --- | --- |
| Claude Code | Command hooks, transcript deltas, context injection, MCP tools | Native 2.1.191 returned a random canary and captured both roles across restart with installed source20f. Source554 subsequently passed the SQLite replay repair in a fresh wheel | Fallback capture observed with degraded identity coverage. Persistent native attempt timed out; full lifecycle and complete transcript acceptance remain open |
| Codex CLI | Command lifecycle hooks, supported context output, MCP tools | Native 0.146.0 used source20f's generated TOML commands, returned a random canary absent from its prompt, and captured both roles across restart | Bounded hook transport observed with fixture vectors and unavailable cognition. Full native correction/forget lifecycle remains open |
| Codex Desktop | Same command hooks and MCP fallback | Shared adapter coverage; no current desktop conversation evidence | Pending |
| Hermes Agent | Optional MemoryProvider and explicit Windows/WSL bridge | Installed 0.21.3 API calls and actual WSL -> Windows authenticated transport. Configured-registry repair passed disposable WSL plugin discovery; its full run failed during service startup | Model turn has not completed; cross-host native recall remains open |
| Generic MCP stdio | MCP server as service client | Subprocess capture, job status, inspect, retrieve, correct, forget | Host-specific acceptance pending |
| Claude Desktop | MCP configuration | Shared protocol coverage; no current desktop conversation evidence | Pending |
| Cursor | MCP plus project rule | Configuration and policy generation tests | Pending |
| Windsurf / Cline | MCP examples | Documentation/configuration only | Pending |

The cross-host fixture uses Codex, Claude Code, and Hermes scope contexts over a real temporary HTTP service and SQLite store. It checks restart, correction preserving identity, forget/replay suppression, and a foreign-project canary. Its provider fixtures are synthetic. It does not launch three native conversations.

Exact evidence boundaries are in [local-memory-hosts.md](validation/local-memory-hosts.md) and the [native host smoke record](validation/2026-09-16/native-host-smoke.md). The [Hermes bridge record](validation/2026-09-16/hermes-bridge-smoke.md) documents installed API and transport checks. The two-week pilot is NOT RUN.

The current source fixes the earlier Claude cross-project event-ID collision
and Stop replay defect. The [installed scope checks](validation/2026-09-16/claude-scope-verification.md)
and [replay checks](validation/2026-09-16/claude-replay-verification/README.md)
record those repairs. They do not replace native host lifecycle acceptance.
The [Claude identity contract](validation/2026-09-16/claude-host-identity-contract.md)
records the remaining ambiguity when a fallback has no explicit transcript association.

All clients need an enabled project, registered profile, and the same runtime descriptor. A repository basename is not a scope identity. Mock cognition is rejected by the normal configured service; use a real provider and input-specific readiness probe. See [functional local memory](functional-local-memory.md), [Codex setup](codex.md), and [Hermes setup](integrations/hermes.md).
