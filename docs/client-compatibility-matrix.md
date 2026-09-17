# Client compatibility and observed evidence

Updated September 17, 2026 for the unreleased functional-local-memory worktree. Implementation, protocol tests, and native conversation acceptance are separate claims.

| Client | Implemented integration | Observed evidence | Native conversation acceptance |
| --- | --- | --- | --- |
| Claude Code | Command hooks, transcript deltas, context injection, MCP tools; provisional missing-ID capture excluded from extraction | Installed sourcea38 completed a persistent native 2.1.191 turn and restart/replay. Two normal transcript-associated events and one deferred capture-only fallback were observed; maintenance created no facts/evidence from the fallback. Instrumented hook settings were supplied explicitly | Provisional identity contract passed. Original exact-two-raw-events gate remains failed; complete lifecycle and plugin discovery remain open |
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

The [native provisional capture record](validation/2026-09-16/provisional-native-claude-v8/README.md)
preserves the original duplicate gate alongside the separate identity result.

Claude 2.1.191 also passed a separate
[plugin metadata check](validation/2026-09-16/claude-plugin-loader-v1/README.md)
against frozen source9aba. Its native metadata commands recognized the session
plugin and all five hook names. Hook execution and MCP connection were not
run; list and details disagreed about MCP inventory. This does not close native
plugin-session acceptance.

Generated Codex, Cursor, and fallback Claude MCP configuration now binds the
Python interpreter that generated it, matching the hook installation. This
prevents a bare `mnemos-mcp` command from selecting an older installation on
the host's PATH. Regenerate the configuration if that Python environment moves.
Existing plugin wrappers retain their explicit interpreter binding.

All clients need an enabled project, registered profile, and the same runtime descriptor. A repository basename is not a scope identity. Mock cognition is rejected by the normal configured service; use a real provider and input-specific readiness probe. See [functional local memory](functional-local-memory.md), [Codex setup](codex.md), and [Hermes setup](integrations/hermes.md).
