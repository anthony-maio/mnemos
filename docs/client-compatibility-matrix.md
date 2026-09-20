# Client compatibility and observed evidence

The [three-host v2 attempt](validation/2026-09-20/three-host-native-v2/README.md) stopped at its first Codex phase.
The synthetic fact was stored and its job completed, but native calls still
used an `exec` wrapper despite the requested direct-tool setting. Separate
provider and MCP execution IDs did not meet the frozen correlation requirement.
The attempt remains failed; no Claude or Hermes model phase followed. Cleanup
passed. Its separate eight-tool Hermes preflight passed, and 141 offline
benchmark cases passed. Three-host continuity remains open. The earlier Hermes
v10 result below remains one complete lifecycle pass and two failures.

The [native Hermes v10 cohort](validation/2026-09-20/hermes-native-cohort-v10/README.md)
completed one of three full lifecycles on `f2f7414`. Attempt 1 omitted
restarted retrieval. Attempt 2 committed deletion but failed the review
requirement: two selected additional facts had not been inspected. Attempt 3
passed teaching, restarted recall, correction, corrected recall, reviewed
deletion, source replay suppression, and restarted final recall. All 40 model
responses had verified 65,536-token context and positive GPU residency; all
attempts established cleanup. No stale-preview rejection occurred, so this
does not establish native recovery from that error or a causal improvement.

This is the first complete bounded Hermes lifecycle in these fixed cohorts,
on an isolated patched host with all five cognition modules disabled. The
two failures remain part of the result. Three-host continuity, complete
capture, T18 quality, Desktop, and the operational pilot remain open.
Earlier checkpoint sections below retain their historical source bindings.

The [stale-preview rejection repair](validation/2026-09-20/stale-preview-checkpoint-v1/README.md)
at `f2f7414` passes 1,547 source tests with 4 skips,
all six source/package checks, 650 isolated installed cases with
4 matching skips, and seven replay assertions. The native v9 failure
used the preceding `e419c2a` wheel and remains failed. Native v10 subsequently
ran the repaired package with the one-pass, two-failure result above.

In v9 attempt 3, the first apply correctly copied all ten candidate IDs,
their required sources, and the preview digest, but changed the preview
limit from 100 to 20. The limit is part of the digest. The old installed
wheel reproduces an opaque HTTP 500 for that mismatch. The repair preserves
the rejection and rollback while returning a typed, noncommitted HTTP 409
that tells the caller to obtain a fresh preview and reuse its digest and
limit. It does not fix arguments, select sources, retry deletion, or drain
pending captures. The tested stale-version path has the same contract;
other validation and unexpected internal errors retain their prior handling.

The [context-verified Hermes v9 cohort](validation/2026-09-20/hermes-native-cohort-v9/README.md)
finished zero of three full lifecycles on `e419c2a`. All 39 model responses
had verified 65,536-token serving context and positive GPU residency; all
three attempts established cleanup. Attempt 1 omitted restarted retrieval.
Attempt 2 passed through corrected recall, then inspected related candidates
without applying deletion. Attempt 3 reached three deletion calls, each
returning degraded HTTP 500 with no committed receipt. The stale-preview rejection was later reproduced and repaired as described
above; the original native result remains failed. No source replay or final recall ran.

Both forget previews reached subsequent recorded HTTP requests intact,
with ten candidates and ten required source IDs. The
[endpoint diagnostic](validation/2026-09-20/ollama-context-boundary-v1/README.md)
found that the old alias served 16,384 tokens through the OpenAI endpoint
while native readiness loaded 65,536. Pinning the context in a new alias
repairs that configuration defect. This does not prove the cause of earlier
failures, and earlier requested settings are not effective-context proof.
The isolated Hermes source stayed pinned; the independently advanced live
interpreter installation and its dependencies were separately bound.
Native lifecycle, three-host continuity, complete capture, T18 quality,
Desktop, and the operational pilot remain open.

Subsequent diagnostics narrow that uncertainty. The [saved previews](validation/2026-09-20/hermes-v8-preview-persistence-v1/README.md)
in v8 attempts 2 and 3 match the native tool results intact. A separate
[single instrumented attempt](validation/2026-09-20/hermes-request-observation-v1/README.md)
recorded two outgoing teach requests with the complete source-selection
schema, thinking disabled, and the requested context setting. That field
did not prove the serving context. It failed at
teaching after omitting the job-status call and never reached forgetting.
This is limited request-boundary evidence, not a rerun or repair of v8.

The [fixed Hermes v8 cohort](validation/2026-09-20/hermes-native-cohort-v8/README.md) ran the verified `e419c2a`
retrieval-view package and finished zero of three complete lifecycles. All
readiness checks and cleanup passed. Attempt 1 saved the fact, then announced
restarted retrieval without calling it. Attempts 2 and 3 passed teaching,
restarted recall, correction, and restarted corrected recall, then failed at
forgetting. Missing source selections were rejected without commitment;
attempt 3 also submitted a malformed digest and an invalid preview limit.
Neither forget phase reached a successful deletion, and no attempt reached
source replay or final recall. The host exits were inside the outer deadline,
not outer process timeouts. Earlier cohorts and T18 results remain unchanged.

The retrieval-view change therefore has package evidence but no successful
native lifecycle in this cohort. Full request bodies were not captured, so
schema forwarding and host processing of large tool results need direct
verification before assigning a cause to the model's mistakes. Native
lifecycle, three-host continuity, complete capture, quality acceptance,
Desktop, and the operational pilot remain open.

The [fixed Hermes v7 cohort](validation/2026-09-20/hermes-native-cohort-v7/README.md) on verified `1a11058`
finished zero of three complete lifecycles. All three model-readiness checks
passed with positive reported GPU residency, and all attempts established
cleanup. Attempt 1 announced a correction without inspecting or correcting;
attempt 2 passed correction and restarted recall, then announced an inspection
without inspecting, previewing, or deleting; attempt 3 saved the fact and
reported its pending receipt but omitted the required job-status check.
No attempt reached a deletion receipt, source replay, or final recall.
This does not establish whether the clearer coverage receipt improves model
selection. The eight tools, prompts, capture, deadlines, and erasure gates
retain the v6 criteria. Earlier cohorts and T18 results remain unchanged.

The correction trace also returned older unverified conversation containing
instructions not to change memory. That is an observed input, not proof of
causation. A separate prospective diagnostic must distinguish context effects
from general next-action omissions. Native mutation and forgetting, three-host
continuity, complete capture, model-quality acceptance, Desktop, and the pilot
remain open.

The latest verified [deletion coverage checkpoint](validation/2026-09-19/forget-coverage-checkpoint-v1/README.md)
at `1a11058` passes 1,439 source tests with 4 skips, all six
source/package checks, 608 isolated installed cases with 4 matching
skips, and seven replay assertions. Related-forget receipts now report
target-only or reviewed selection and retained exact, advisory, and eligible
source counts within the validated preview at commit. Direct forget has no
preview counts. Both explicitly leave semantic erasure unproven. Tool guidance
asks the model to report that coverage. Native v7 ran this package and
failed all three lifecycles before any deletion receipt was reached.
Earlier native and model results retain their original runtime bindings.

Current provider package: [source-selection rejection repair](validation/2026-09-19/related-forget-error-checkpoint-v1/README.md)
on `e454a85` passes 1,427 source tests (4 skips), all six source/package
checks, 604 installed cases (4 matching skips), and seven replay assertions.
Omitting source events required by selected related facts now returns a
specific HTTP 409 rejection through the authenticated client and Hermes,
with no mutation, invalid-request retry, or incidental outbox replay. The
existing generic failures remain for other validation paths and unexpected
errors. No native model has run this repaired package. The v6 failures below
remain bound to their earlier `db932be` runtime and unchanged outcomes.

Previous native observation: the [fixed Hermes v6 cohort](validation/2026-09-19/hermes-native-cohort-v6/README.md)
on verified `db932be` finished zero of three full lifecycles. All three
passed teaching, restarted recall, correction, and restarted corrected
recall, then failed at forgetting. Attempts 1 and 2 used the new preview/apply
tools but selected only the target, leaving conversational copies. Attempt 1
also skipped inspections; attempt 2 passed reviewed target-deletion checks
and failed only the retained-label scan. Attempt 3 selected additional facts
without their required source IDs; both apply calls returned degraded HTTP
500. No source replay or final recall ran in this cohort. All outcomes and
cleanup are archived. Three-host continuity, complete capture, real-model
quality acceptance, Desktop, and the pilot remain open.

Earlier native observation: the [fixed Hermes v5 cohort](validation/2026-09-19/hermes-native-cohort-v5/README.md)
on verified `fd83fb2` finished zero of three full lifecycles. All three
attempts passed teaching, restarted recall, correction, and restarted
corrected recall. Attempts 1 and 2 deleted the target and suppressed original
source replay, but older conversational copies remained in the final model
request. Attempt 3 announced a deletion without calling the tool. No
provider-error gate failed. The earlier v3 and v4 outcomes remain preserved.
Three-host continuity, complete capture, real-model quality acceptance,
Desktop, and the pilot remain open.

Earlier native lifecycle observation: the [Hermes v2 attempt](validation/2026-09-19/hermes-native-lifecycle-local-v2/README.md)
stopped at teach after the model used the wrong returned job ID. The
[verified error-handling repair](validation/2026-09-19/job-not-found-checkpoint-v1/README.md)
changes unknown or inaccessible job lookups from HTTP 500 to a generic 404.
It does not correct model arguments or establish native lifecycle acceptance.
The complete three-host native driver remains NOT RUN.

Updated September 19, 2026 for the unreleased functional-local-memory worktree. The versions below identify observed runs, not a promise about every installed or future host version. Adapter tests, native conversations, model quality, and daily-use readiness are separate claims.

The earlier [source/package checkpoint](validation/2026-09-19/hermes-native-identity-checkpoint-v1/README.md) is `3730fd9`, with 1,247 source passes, 530 installed passes, four/two respective skips, and seven replay assertions. It preserves native Hermes message identity through rewind and rejects changed-payload retries under the same native ID. It includes the earlier correction acknowledgement repair. The earlier [Hermes API checkpoint](validation/2026-09-19/hermes-lifecycle-checkpoints-v1/README.md) at `5e4b057` exercises all six scoped tools through the actual Windows/WSL bridge, growing transcript replay, provider recreation, and extraction-ineligible fallback receipts. That check uses fixture providers, with no model conversation or service outage. The [mixed-host replay probe](validation/2026-09-18/native-crosshost-recovery-v4/README.md) verifies the `fc72cc2` installed wheel and clean plugin snapshot. Earlier lifecycle results retain their original source bindings; a newer package does not silently rerun those observations.

The first native model-driven Hermes lifecycle attempt is preserved as a
[failed terminal archive](validation/2026-09-19/hermes-native-lifecycle-local-v1/README.md).
Its teach phase stopped because the requested asserted content omitted the
final period and an auxiliary title-generation request had null options. The
later patched-host checks are separate evidence and do not convert this
installed native failure into full lifecycle acceptance.

| Client | Implemented integration | Observed native evidence | Remaining acceptance |
| --- | --- | --- | --- |
| Claude Code 2.1.191 | Command hooks, transcript deltas, context injection, MCP tools, and extraction-ineligible provisional capture when native identity is missing | [Shared lifecycle v9](validation/2026-09-18/native-codex-claude-continuity-v9/README.md) on `5af3d15` retrieved and corrected a Codex-taught fact, then reviewed and deleted the target plus five selected copies. [Claude lifecycle v18](validation/2026-09-17/native-claude-lifecycle-v18/README.md) on `dedfcfd` passed 15 bounded fact-lifecycle criteria. Mixed-host v4 on `fc72cc2` verified native queue replay and restart. | Degraded identity diagnostics, complete capture, native real-model extraction, and general startup reliability remain open. |
| Codex CLI 0.146.0 | Generated lifecycle hooks, context output, and MCP tools | Shared lifecycle v9 taught the fact, retrieved Claude's corrected version after restart, and completed both post-forget queries without repeating the removed values. Mixed-host v4 verified same-event replay from both native hosts and unchanged records through restart. | Earlier Stop timeouts remain recorded. Complete capture across lifecycle failures, tail latency, real-model quality, and pilot reliability remain open. |
| Codex Desktop | Shared command-hook adapter and MCP fallback | No isolated native desktop conversation has been validated. | NOT RUN; CLI evidence does not establish desktop behavior. |
| Hermes Agent in WSL, isolated source `14a3463` with cwd-only patch | Optional MemoryProvider, lifecycle callbacks, entry-point discovery, and explicit Windows/WSL bridge | [Native v10](validation/2026-09-20/hermes-native-cohort-v10/README.md) completes one of three full lifecycles on `f2f7414`, including correction, reviewed deletion, replay suppression, and final recall. All 40 effective-context checks and three cleanup checks pass. | Two attempts fail at omitted recall and incomplete candidate inspection. Five cognition modules were disabled. Live compatibility, built-in remove/replace reconciliation, capture across failures, native outage recovery, and three-host continuity remain open. |
| Generic MCP stdio | MCP server acts as a client of the local service | Subprocess tests cover capture, job status, inspect, retrieve, correct, and forget. | Protocol evidence only; each host needs its own acceptance run. |
| Claude Desktop | MCP configuration | Shared protocol coverage. | Native conversation NOT RUN. |
| Cursor | MCP plus project rules | Configuration and policy-generation tests. | Native conversation NOT RUN. |
| Windsurf / Cline | MCP examples | Documentation/configuration only. | Native conversation NOT RUN. |

The [Hermes rewind diagnostic](validation/2026-09-19/hermes-rewind-diagnostic-v2/README.md)
also reproduces a same-session identity collision: an identical regenerated
turn after `rewound=True` reuses the removed turn's capture IDs and timestamps.
The outbox remains at four records rather than six logical observations. This
is an installed MemoryManager callback fixture, not interactive CLI or service
recovery acceptance. The [actual database successor](validation/2026-09-19/hermes-native-rows-contract-v3/README.md) now preserves the surviving prefix and captures the regenerated suffix as new occurrences when native row IDs are available. ID-less fallback, loaders omitting IDs, compaction clones, and existing-data identity migration remain unresolved.

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
