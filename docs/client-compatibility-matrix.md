# Client compatibility and observed evidence

Updated September 19, 2026 for the unreleased functional-local-memory worktree. The versions below identify observed runs, not a promise about every installed or future host version. Adapter tests, native conversations, model quality, and daily-use readiness are separate claims.

The latest [source/package checkpoint](validation/2026-09-19/correction-retry-checkpoint-v1/README.md) is `ea88dd4`, with 1,201 source passes, 513 installed passes, four/two respective skips, and seven replay assertions. It repairs recovery of a completed correction's durable acknowledgement without overwriting a later edit. The earlier [Hermes API checkpoint](validation/2026-09-19/hermes-lifecycle-checkpoints-v1/README.md) at `5e4b057` exercises all six scoped tools through the actual Windows/WSL bridge, growing transcript replay, provider recreation, and extraction-ineligible fallback receipts. That check uses fixture providers, with no model conversation or service outage. The [mixed-host replay probe](validation/2026-09-18/native-crosshost-recovery-v4/README.md) verifies the `fc72cc2` installed wheel and clean plugin snapshot. Earlier lifecycle results retain their original source bindings; a newer package does not silently rerun those observations.

| Client | Implemented integration | Observed native evidence | Remaining acceptance |
| --- | --- | --- | --- |
| Claude Code 2.1.191 | Command hooks, transcript deltas, context injection, MCP tools, and extraction-ineligible provisional capture when native identity is missing | [Shared lifecycle v9](validation/2026-09-18/native-codex-claude-continuity-v9/README.md) on `5af3d15` retrieved and corrected a Codex-taught fact, then reviewed and deleted the target plus five selected copies. [Claude lifecycle v18](validation/2026-09-17/native-claude-lifecycle-v18/README.md) on `dedfcfd` passed 15 bounded fact-lifecycle criteria. Mixed-host v4 on `fc72cc2` verified native queue replay and restart. | Degraded identity diagnostics, complete capture, native real-model extraction, and general startup reliability remain open. |
| Codex CLI 0.146.0 | Generated lifecycle hooks, context output, and MCP tools | Shared lifecycle v9 taught the fact, retrieved Claude's corrected version after restart, and completed both post-forget queries without repeating the removed values. Mixed-host v4 verified same-event replay from both native hosts and unchanged records through restart. | Earlier Stop timeouts remain recorded. Complete capture across lifecycle failures, tail latency, real-model quality, and pilot reliability remain open. |
| Codex Desktop | Shared command-hook adapter and MCP fallback | No isolated native desktop conversation has been validated. | NOT RUN; CLI evidence does not establish desktop behavior. |
| Hermes Agent 0.21.3 in WSL | Optional MemoryProvider, lifecycle callbacks, entry-point discovery, and explicit Windows/WSL bridge | The [isolated patched CLI v4 run](validation/2026-09-19/native-hermes-local-v4/README.md) passes all 16 single-turn scoped recall and raw-capture checks. Four committed callbacks produce exactly two persisted conversation events. The earlier duplicate-callback failure is preserved. | The installed CLI omits provider cwd. Built-in remove/replace lack exact Mnemos targets. Capture across failures, native outage/restart recovery, and cross-host correction/forget remain open. |
| Generic MCP stdio | MCP server acts as a client of the local service | Subprocess tests cover capture, job status, inspect, retrieve, correct, and forget. | Protocol evidence only; each host needs its own acceptance run. |
| Claude Desktop | MCP configuration | Shared protocol coverage. | Native conversation NOT RUN. |
| Cursor | MCP plus project rules | Configuration and policy-generation tests. | Native conversation NOT RUN. |
| Windsurf / Cline | MCP examples | Documentation/configuration only. | Native conversation NOT RUN. |

The [Hermes rewind diagnostic](validation/2026-09-19/hermes-rewind-diagnostic-v2/README.md)
also reproduces a same-session identity collision: an identical regenerated
turn after `rewound=True` reuses the removed turn's capture IDs and timestamps.
The outbox remains at four records rather than six logical observations. This
is an installed MemoryManager callback fixture, not interactive CLI or service
recovery acceptance. Identifying only the regenerated suffix remains open.

The shared v9 lifecycle result passes 15 bounded criteria, including scoped controls, correction retaining fact identity, reviewed deletion, direct replay suppression, and final recall. All three Codex processes exited zero, but reverse and final Stop hooks timed out. The fixture used lexical vectors, unavailable cognition, and disabled modules. It establishes those observed operations, not semantic quality, arbitrary paraphrase erasure, or complete host acceptance.

The [Hermes API recovery probe](validation/2026-09-19/hermes-api-recovery-v1/README.md)
on `5e4b057` passes a separate outage/restart contract. Three fresh WSL host
processes demonstrate durable buffering, a required checkpoint backed by the
real outbox, ordinary-request replay after explicit service restoration, and
forget suppression after another restart. Exact IDs/timestamps and both scope
controls survive. Providers are fixtures; complete native model lifecycle,
automatic daemon restoration, and hard-kill acceptance remain open.

The newer mixed-host v4 result passes a different, bounded recovery protocol. Four native turns exited zero. Three queued events received overlapping deliveries from Codex and Claude, each resolving to one durable event and job. Eight normal messages and two non-extractable provisional records survived restart; the outbox emptied and both scope controls survived. One request expired at the unchanged two-second pairing deadline. Overlap was instrumented, service restoration was explicit, and the providers were fixtures. This does not establish automatic daemon recovery or real-provider readiness.

Claude's prompt hook did not supply a native occurrence ID in the observed host version, and the transcript was not yet available during the [hook-time identity probe](validation/2026-09-17/native-claude-hook-identity-v3/README.md). Mnemos retains that prompt provisionally, then captures UUID-backed transcript records separately. Provisional records cannot produce facts. Matching text alone does not establish identity, so `identity_reference_unavailable` and `ambiguous_transcript_association` remain visible rather than being hidden by a passing fact-lifecycle result.

Generated Codex, Cursor, and fallback Claude MCP entries bind the Python interpreter that generated them. Generated Codex/Claude entries and the Claude plugin also bind the actual host identity. Regenerate configuration if its Python environment moves. The Claude plugin defaults to `alwaysLoad: true`; generated configuration preserves an explicit existing false. The [startup control](validation/2026-09-17/native-claude-alwaysload-v2/README.md) and subsequent lifecycle v18 record support that setting without claiming general startup reliability.

All clients need an enabled project, a registered profile, and the same runtime descriptor. A repository basename is not a scope identity. The normal configured service rejects mock cognition; local readiness requires an input-specific provider check. See the [operating guide](functional-local-memory.md), [Codex setup](codex.md), [Hermes setup](integrations/hermes.md), and [host-validation procedure](validation/local-memory-hosts.md).

Historical negative evidence remains preserved:

- [Shared lifecycle v6](validation/2026-09-17/native-codex-claude-continuity-v6/README.md) failed when final Codex recall reused a surviving captured copy and skipped its second query. V9's reviewed deletion is a bounded later result, not a rewrite of that failure.
- [Mixed replay v3](validation/2026-09-18/native-crosshost-recovery-v3/README.md) failed its original first-arrival gate. V4 tracks active waiters; the prior failure and preliminary budget/exact-reply failures remain intact.
- [Claude lifecycle v15](validation/2026-09-17/native-claude-lifecycle-v15/README.md) initialized with Mnemos pending and made no lifecycle calls. V18 later passed fact-lifecycle checks but retained degraded capture diagnostics and an unexplained connection-abort observation.
- [Native host smoke](validation/2026-09-16/native-host-smoke.md) and [Hermes bridge smoke](validation/2026-09-16/hermes-bridge-smoke.md) preserve the earlier host, API, and transport boundaries.

The [implementation status](validation/2026-09-16/implementation-status.md) and [claim ledger](validation/2026-09-16/claim-evidence-ledger.md) retain the broader history and current gates. Three [baseline SSD performance runs](validation/2026-09-19/performance-recovery-v1/README.md) now pass the recall/capture latency targets; this does not establish native-host deadlines or full-module latency. Real-model ablations, Hermes/Desktop acceptance, complete capture/recovery, and the two-week pilot remain open. Nothing here promotes the worktree to a published release.
