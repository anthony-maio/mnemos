# Client compatibility and observed evidence

Updated September 17, 2026 for the unreleased functional-local-memory worktree. Implementation, protocol tests, and native conversation acceptance are separate claims.

| Client | Implemented integration | Observed evidence | Native conversation acceptance |
| --- | --- | --- | --- |
| Claude Code | Command hooks, transcript deltas, context injection, MCP tools; provisional missing-ID capture excluded from extraction | Installed source6600 completed native 2.1.191 health and retrieve calls through the actual plugin against the intended project store, plus three lifecycle hooks, seeded recall, restart, and reconstructed Stop replay | Bounded native read and provisional identity contracts passed. Original exact-two-raw-events gate remains failed. Complete extraction/correction/forget lifecycle remains open |
| Codex CLI | Command lifecycle hooks, supported context output, MCP tools | Native 0.146.0 used source20f's generated TOML commands, returned a random canary absent from its prompt, and captured both roles across restart | Bounded hook transport observed with fixture vectors and unavailable cognition. Full native correction/forget lifecycle remains open |
| Codex Desktop | Same command hooks and MCP fallback | Shared adapter coverage; no current desktop conversation evidence | Pending |
| Hermes Agent | Optional MemoryProvider and explicit Windows/WSL bridge | Installed 0.21.3 API calls and actual WSL -> Windows authenticated transport. Sourcec805 passed full verification and discovery through the actual wheel entry point in WSL | Model turn has not completed; cross-host native recall remains open |
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

The [latest host evidence](validation/2026-09-16/related-forget-verification/README.md)
also preserves the cancelled sourcec805 native Codex lifecycle attempt and
matched annotated/unannotated recorder probes. Sourceb179 now declares
accurate MCP operation effects. Installed sourcea38 subsequently completed one
native Codex health call after the probe pinned MCP command resolution to the
verified wheel. No approval or hook-trust override was used. The response
reported the seeded SQLite store; no hook callbacks were observed. This proves
MCP health dispatch, not the complete capture or correction/forget lifecycle.
[Command binding and native health evidence](validation/2026-09-16/codex-health-installation-binding/README.md).

Source644's fresh wheel then passed native health with the generated exact
interpreter command and inherited PATH, verifying the product launcher repair
without the earlier probe's PATH patch.
[Current package check](validation/2026-09-16/extraction-contract-package-verification/README.md).

The later d94 installed wheel passed another isolated MCP/API lifecycle
preflight. Its native Codex run reached `mnemos_store`, then received
`user cancelled MCP tool call` without a successful service result. It did not
reach the remaining native lifecycle steps. No approval or hook-trust bypass
was used, and cleanup removed the private auth copy and runtime descriptor.
[Preserved native attempt](validation/2026-09-16/native-codex-lifecycle-v4/README.md).

The installed `56182d9` wheel then reached a single native `mnemos_retrieve`
call in a fresh fixture. Codex returned `user cancelled MCP tool call` with
no result. This independently leaves retrieval acceptance open, even though
the package and 87 installed contract cases passed. The
[read probe](validation/2026-09-17/native-codex-retrieve-v2/README.md) preserves
its exact command, stream, installation binding, and cleanup evidence.

The [native provisional capture record](validation/2026-09-16/provisional-native-claude-v8/README.md)
preserves the original duplicate gate alongside the separate identity result.

Claude 2.1.191 also passed a separate
[plugin metadata check](validation/2026-09-16/claude-plugin-loader-v1/README.md)
against frozen source9aba. Its native metadata commands recognized the session
plugin and all five hook names. Hook execution and MCP connection were not
run; list and details disagreed about MCP inventory. This does not close native
plugin-session acceptance.

The subsequent [native plugin checkpoint](validation/2026-09-16/native-plugin-checkpoint-v1/README.md)
uses frozen `6600dc2` and the verified installed wheel. Its v10 turn completed
in 10.567 seconds and reported $0.0291 cost. The actual plugin's three lifecycle
hooks ran, the seeded canary was recalled, and native initialization reported
Mnemos connected with fourteen tools. The service retained the captures after
restart; a reconstructed Stop payload replayed the retained native transcript
through the plugin without adding an event. No MCP tool call was executed.

Two UUID-backed transcript events and one provisional user event were retained.
The provisional event remained deferred and ineligible for fact extraction;
its identity contract passed. The original exact-two-raw-event gate and the
runner's aggregate association check across all user events remained failed.
Matching identical text alone is not a safe way to merge repeated prompts.
The later [hook-time observation](validation/2026-09-17/native-claude-hook-identity-v3/README.md)
confirmed that the native prompt payload had no prompt ID and its transcript
did not yet exist during UserPromptSubmit. The transcript was available by
Stop. This observed host timing prevents an earlier transcript lookup from
resolving that provisional event's identity.
The later [native MCP read attempt](validation/2026-09-17/native-claude-mcp-v11/README.md)
completed its native turn and passed capture identity, restart, and replay
checks, but initialization left Mnemos pending with no tools or tool calls.
The seeded answer was available through hook context and does not prove MCP
retrieval. This negative result leaves native MCP read acceptance open.
The subsequent [v12 read check](validation/2026-09-17/native-claude-mcp-v12/README.md)
used default permission mode with the same two-tool allowlist. Actual health
and retrieve calls and their correlated successful results were observed;
health matched the fixture database, profile, and project, and retrieve returned
the project-scoped canary. All new read criteria passed, including identity,
restart, and replay. This does not prove native extraction or mutation behavior.
The installed `56182d9` wheel's [v15 lifecycle attempt](validation/2026-09-17/native-claude-lifecycle-v15/README.md)
again started with an empty native tool list and Mnemos pending. It executed
no lifecycle tool calls; phase 2 was not run. A separate installed-package
MCP exchange exposed all 14 tools in 1.853 seconds. That control verifies the
catalog in isolation, but does not explain the native startup failure or
establish correction and forget acceptance.
The [v16 startup-wait control](validation/2026-09-17/native-claude-lifecycle-v16/README.md)
set `MCP_CONNECTION_NONBLOCKING=0` only for its disposable Claude process.
Mnemos connected with all 14 tools, and all six checks of the actual native
correct/consolidate/retrieve sequence passed. A subsequent direct retrieval
after maintenance failed before restart or forget. Capture diagnostics also
remained degraded. This records a successful native correction sequence;
complete lifecycle and general startup reliability remain open.
The v9 attempt preserves a budget-limit failure with inherited account-level
connectors; v10 disabled those connectors only in its disposable process.

Generated Codex, Cursor, and fallback Claude MCP configuration now binds the
Python interpreter that generated it, matching the hook installation. This
prevents a bare `mnemos-mcp` command from selecting an older installation on
the host's PATH. Regenerate the configuration if that Python environment moves.
Existing plugin wrappers retain their explicit interpreter binding.

All clients need an enabled project, registered profile, and the same runtime descriptor. A repository basename is not a scope identity. Mock cognition is rejected by the normal configured service; use a real provider and input-specific readiness probe. See [functional local memory](functional-local-memory.md), [Codex setup](codex.md), and [Hermes setup](integrations/hermes.md).
