# Client compatibility and observed evidence

Updated September 17, 2026 for the unreleased functional-local-memory worktree. Implementation, protocol tests, and native conversation acceptance are separate claims.

| Client | Implemented integration | Observed evidence | Native conversation acceptance |
| --- | --- | --- | --- |
| Claude Code | Command hooks, transcript deltas, context injection, MCP tools; provisional missing-ID capture excluded from extraction | Installed source dedfcfd completed native correction, consolidation, retrieval, and forgetting through the actual 2.1.191 plugin, with restart and reconstructed Stop replay checks | All 15 bounded fact-lifecycle criteria passed in v18 with the new startup default. Degraded capture keeps the overall result partial. Extraction quality and complete capture acceptance remain open |
| Codex CLI | Command lifecycle hooks, supported context output, MCP tools | Native 0.146.0 retrieved the seeded project fact through an approved MCP call against installed source30eaf42, excluding the foreign project fact. Earlier source20f separately demonstrated hook recall and both-role capture across restart | Bounded MCP retrieval passes with fixture vectors. The latest process timed out during shutdown and retained only user capture. Complete capture, correction/forget, and cross-host lifecycle acceptance remain open |
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

A later [approval-path investigation](validation/2026-09-17/native-codex-approval-diagnosis-v1.md)
identified noninteractive approval handling as a plausible cause. A similar
upstream report concerns another CLI version, so it does not prove the cause
here. A subsequent [interactive check](validation/2026-09-17/native-codex-interactive-v1/README.md)
accepted the normal per-call approval, then returned an MCP argument error:
Codex supplied a scope list while the tool accepted only a string. No retrieval
reached the service. Source `d58e082` now accepts either form and rejects empty
or invalid scope lists; [stdio regression checks](validation/2026-09-17/mcp-scope-arguments-v1/README.md)
cover the original failure and foreign-project isolation. The subsequent
[interactive v2 check](validation/2026-09-17/native-codex-interactive-v2/README.md)
used the repaired installed `30eaf42` wheel and normal per-call approval.
The actual MCP result returned the seeded project fact with its stable ID and
version, excluding the foreign project fact. A matching service dispatch
distinguishes this call from two hook-driven recalls. The native process
later timed out during shutdown; its original failed report is preserved.
User capture was committed, but assistant capture was absent. This closes
bounded native MCP retrieval only, with deterministic fixture vectors and no
model-quality claim. It does not isolate the cause of earlier cancellations.

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
The [exact-query diagnosis](validation/2026-09-17/claude-v16-retrieval-diagnosis-v2/README.md)
found the corrected record persisted at version 2. Constant fixture embeddings
left all seven records tied; ID ordering and the unchanged context budget
excluded the target. This explains the diagnostic retrieval miss without
promoting the unrun native restart and forget phases.
The subsequent [v17 lifecycle check](validation/2026-09-17/native-claude-lifecycle-v17/README.md)
used installed `8d865a5` and deterministic lexical fixture vectors. Native
correction, consolidation, retrieval, and forgetting succeeded. The correction
survived restart; the forgotten target stayed absent after a second restart
and reconstructed Stop replay, while the unrelated fact remained retrievable.
All 15 fact-lifecycle criteria passed. Hook capture retained degraded diagnostics,
so the overall result remains partial. This verifies the bounded native fact
lifecycle, not extraction quality or complete capture acceptance.
The v9 attempt preserves a budget-limit failure with inherited account-level
connectors; v10 disabled those connectors only in its disposable process.

Generated Codex, Cursor, and fallback Claude MCP configuration now binds the
Python interpreter that generated it, matching the hook installation. This
prevents a bare `mnemos-mcp` command from selecting an older installation on
the host's PATH. Regenerate the configuration if that Python environment moves.
Existing plugin wrappers retain their explicit interpreter binding.
Generated Codex and Claude MCP entries also bind their actual host identity,
and the Claude plugin wrapper sets `MNEMOS_HOST=claude-code` in its child
environment. This prevents new MCP evidence from falling back to `manual`
or inheriting another client's identity. Existing installations need the
updated launcher or regenerated configuration; historical evidence is not
rewritten by this change.

The Claude plugin and generated Claude entries now default to `alwaysLoad: true`
for Mnemos; an existing explicit false in generated configuration is retained.
The [isolated startup control](validation/2026-09-17/native-claude-alwaysload-v2/README.md)
used the `8d865a5` plugin with only this manifest field overlaid and global
nonblocking startup enabled. Claude 2.1.191 exposed all 14 tools at initialization
and completed one health call against the intended database, profile, and
project. This supports the server-specific startup setting; it is separate from
the v17 lifecycle result and does not remove capture limitations.

The [v18 lifecycle run](validation/2026-09-17/native-claude-lifecycle-v18/README.md)
used the complete `dedfcfd` source and installed wheel with no manifest overlay,
the unchanged v17 runner, and global nonblocking startup enabled. Both native
phases exposed the required tools at initialization. All 15 fact-lifecycle
criteria passed; degraded capture kept the overall result partial. A service
response logged `ConnectionAbortedError` during the run; no fact criterion
failed, and the connection-abort cause remains unproven. This isolated run
does not establish general startup reliability or semantic retrieval quality.

All clients need an enabled project, registered profile, and the same runtime descriptor. A repository basename is not a scope identity. Mock cognition is rejected by the normal configured service; use a real provider and input-specific readiness probe. See [functional local memory](functional-local-memory.md), [Codex setup](codex.md), and [Hermes setup](integrations/hermes.md).
