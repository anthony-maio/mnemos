# Client compatibility and observed evidence

## November 7, 2026 latest acceptance limits

The frozen `f5a944ea` wheel passes its bounded native Codex v12 explicit-memory
lifecycle with hooks disabled. Full configuration acceptance and actual pilot
use remain open. No later test below transfers acceptance to another artifact.

The [installed retrieval/support comparison](validation/2026-11-07/retrieval-support-development-v1.md)
uses actual returned candidates across 272 observations and 68 known questions.
Precision is 94.74%, expected-fact retention is 90%, correct abstention is 52/56
and complete p95 is 8,641.48 ms. Precision, retention and latency fail. The
comparison is closed, the treatment stays experimental and no fresh acceptance
has been started from this result.

The [installed pilot command timing](validation/2026-11-07/installed-pilot-command-timing-v1.md)
uses the same frozen wheel and 10,000-fact snapshot for both launchers. Complete
recall p95 is 1,098.84 ms for Python and 1,508.41 ms for the native experiment,
above 500 ms. Stop p95 is 1,359.04/360.88 ms and prompt p95 is 2,992.09/2,036.34 ms.
Neither launcher provides an early durable acknowledgment. Internal persistence
is not a host acknowledgment or a replacement for the 100 ms capture target.
The 160 matching output pairs and 208 recovered captures establish integrity
for this run, not native-host timing or pilot reliability. All prior results
retain their dates, artifacts and original pass/fail labels.

The optional explicit-MCP learning proposal passed its separately authorized
[nine-phase native validation](validation/2026-11-07/explicit-learning-pilot-native-v2.md)
on the same frozen wheel. It remains inactive; activation is awaiting a separate
decision. Validation does not count toward the original two-week pilot or
semantic-quality gates. Both use counters remain zero.

The subsequent [discriminative support screen](validation/2026-11-07/discriminative-support-feasibility-v1.md)
also failed: it retained 0/40 supported answers and took 933.20 ms support-only
p95. It is closed and outside defaults. Its component timing does not replace
complete retrieval timing. Canonical source now also includes the locally tested
IPv6 dispatch repair at `49667577`; the frozen pilot wheel does not include it.

## November 7, 2026 evidence-index checkpoint

Product `f5a944ea` passes [3,674 full-suite tests with four platform skips](validation/2026-11-07/full-regression-fact-evidence-index-v1.md).
All 75 experimental native-launcher subprocess tests run and pass. The
[isolated installed artifact](validation/2026-11-07/fact-evidence-index-install-v1.md)
also passes real-BGE CLI/MCP remember, inspection, service restart, same-ID recall
and project isolation. Its evidence index preserves equal-time ordering across
restart. The package remains an unpublished development candidate.

The [complete installed comparison](validation/2026-11-07/fact-evidence-index-performance-v1.md)
uses the same native launcher in both arms. All 160 output/context pairs match,
all 208 captures survive restart, and every job completes. Indexed recall p95
passes 500 ms at 104.92 ms. Complete Stop/prompt p95 is 321.73/365.53 ms; both
fail that experiment's whole-command 100 ms limit. The original requirement is
100 ms for durable capture; direct acknowledgment timing remains unproven. The
[prospective timing distinction](plans/2026-11-07-candidate-convergence.md) does
not change the recorded failures. Earlier results retain their separate bindings.

The latest [native Codex lifecycle](validation/2026-11-07/native-codex-v12.md)
passes all six phases on this same `f5a944ea` wheel, with 19 correlated calls,
reviewed forgetting and suppressed original-source replay. All six inventories
and installed guidance delivery verify. Hooks were disabled; no trust bypass
was used. The earlier Codex v11 and Claude v5 results retain their own bindings.
This pass does not establish automatic capture, normal hook trust, the complete
pilot configuration or the experimental native launcher.

The [refreshed private pilot preparation](validation/2026-11-07/persistent-pilot-v4.md)
uses the same wheel. Its 96 installed package files, 57 dependency versions,
four configuration probes and seven disposable application/rollback checks
pass. Live settings remain unchanged. The pilot is inactive at zero days;
quality, durable acknowledgment, normal hook trust and actual use remain open.

## November 7, 2026 pilot integration checkpoint

The [Claude pilot integration check](validation/2026-11-07/native-pilot-claude.md)
loaded the drafted project MCP and local hook settings using the installed
`6d0e5ba6` wheel. One print-mode session restarted the disposable store,
retrieved and inspected the seed, and captured five raw records with verified
physical project/profile/session identity. Cleanup passed and the real pilot
stayed inactive. The prompt and Stop hooks still reported degraded identity:
one provisional prompt record is separate from its UUID-backed transcript
record. Its capture-only job and four extraction jobs remain pending; the seed
is the sole fact. This bounded integration pass does not establish complete
capture identity, interactive approval, Codex pilot hooks or daily use.

The subsequent [Codex pilot integration check](validation/2026-11-07/native-pilot-codex.md)
passed on the same installed wheel. Two correlated native calls retrieved and
inspected the controller seed. UserPromptSubmit and Stop persisted one user
prompt and one assistant reply with normal identities and a shared native turn.
Their completed jobs produced two unasserted raw chunks, not accepted facts.
Cleanup passed. The run used an invocation-only trust override after strict
command and artifact validation; ordinary interactive trust and all-hook
coverage remain unverified. Both pilot integrations remain synthetic checks.

The [configuration rollback rehearsal](validation/2026-11-07/pilot-config-rollback.md)
also passed exact application/restore checks for all seven drafts and fourteen
drift refusals on disposable copies. Nine live paths stayed unchanged. These
results do not apply the draft, pass recovery/quality gates or start daily use.

The [Codex pilot startup-outage check](validation/2026-11-07/native-pilot-codex-outage.md)
then passed on the same installed wheel with normal auto-start enabled. A held
runtime lock blocked startup during one native turn; both messages entered the
outbox. Releasing the lock and making an ordinary health request drained them,
and maintenance completed two unasserted raw chunks. Exact source replay and
a second restart preserved the completed rows and explicit control. Cleanup
passed. This covers one simulated startup fault; provider failure, mid-write
crash, ordinary trust, quality and actual daily use remain open.

A separate [installed embedding recovery check](validation/2026-11-07/pilot-embedding-recovery.md)
passed one injected typed provider failure on `6d0e5ba6`. Two writes stayed
pending during the fault, existing recall reported lexical-only degradation,
and ordinary installed service restart restored real semantic processing after
the unchanged backoff. Exact replay and another restart preserved rows. This
API check adds no native-host evidence and does not establish natural ONNX
failure classification, extraction quality or daily use.

## November 6, 2026 checkpoint

The [native Codex v7 check](validation/2026-11-06/native-codex-v7-checkpoint.md)
passed all six explicit-memory phases on installed product `6d0e5ba6` with
runner `6c12ff94`: teach and inspect, restarted recall, same-ID correction,
corrected recall, reviewed forgetting, and final recall after restart and
original-source replay. All 19 MCP calls completed with verified native call
correlation. Source evidence, retained controls, scope isolation and cleanup
passed. This is one synthetic scenario and one run, with experimental modules
disabled. The earlier [v6 pass and failed attempts](validation/2026-11-06/native-codex-checkpoint.md)
retain their original artifact bindings and outcomes.

The same product passed isolated installed CLI/MCP checks. The later
[full source suite](validation/2026-11-06/full-regression-codex-v7.md)
at `6c12ff94` passed 2,845 tests with four platform-specific skips and the
canonical website explicitly selected.

[Claude v5](validation/2026-11-06/native-claude-v5-checkpoint.md) passed all six
phases on installed product `6d0e5ba6`, with runner `15229fb7`, Claude Code
2.1.191 and `claude-sonnet-4-6`. All 29 calls passed native correlation and
result checks. Each startup advertised exactly the permitted tools. Complete
reviewed deletion, ten exact original-source replay suppressions, final
restarted recall, retained controls and cleanup passed. Both credential sources
were unchanged. The earlier failed attempts remain preserved.

This is one synthetic run on the named wheel. Its forget phase used $0.4962732
of the $0.50 phase allowance; the pass does not establish a comfortable cost
margin or broader reliability. Current-candidate Hermes acceptance, automatic
extraction, native tokenizer compatibility and actual pilot use remain open.
Codex v7 and Claude v5 now share the same wheel binding. Neither pass transfers
automatically to later wheels or the separate pilot launcher configuration.

## Earlier checkpoints

The [Codex outage run](validation/2026-09-20/native-codex-outage-v1/README.md)
passes one stopped-service raw-capture recovery on verified product `9722d92`.
Two exact native messages remained buffered, then persisted after restart with
two pending jobs and zero facts. Receipt accounting, physical identities,
replay, second restart, and cleanup passed. Generated MCP and hook settings
explicitly pin disposable runtime/storage and disabled autostart. This adds
bounded Codex outage evidence; extraction and wider reliability remain open.

The [three-host v7 run](validation/2026-09-20/three-host-native-v7/README.md) passed all 11 native phases on verified
product `e6887de`: Codex and Claude teaching, Hermes recall and same-ID
correction, corrected recall from all three hosts, Claude reviewed deletion,
and final recall from all three hosts after restart and source replay.
All 12 original-source replays were suppressed. Independent read-only SQLite
inspection confirmed the target absent, 18 suppression rows, and unchanged
review, unrelated, and foreign-scope controls. Cleanup passed.

The [Claude service-outage v2 run](validation/2026-09-20/native-claude-outage-v2/README.md)
adds one bounded transport observation on verified product `a64afb7`. Claude
produced two exact native messages while the disposable service was stopped.
Three queued entries, including one provisional prompt capture, drained after
restart; three pending jobs, zero facts, replay, a second restart, physical
identities, and cleanup all passed. This is raw capture and outbox recovery for
one disposable run. It does not establish provider-outage or in-flight crash
recovery, extraction quality, or general reliability. The earlier v1
default-store failure remains part of the evidence.

Default clients now honor `MNEMOS_AUTO_START` when no explicit constructor
argument is supplied. Native disposable phases bind runtime and SQLite paths
to their fresh output root, so the outage observation does not rely on a user
profile's default store. Explicit `auto_start=False` remains the test and
managed-service-stop escape hatch.

The three-host run is one bounded lifecycle with real Nomic embeddings and all
five cognition modules disabled. It does not close complete capture, Codex Desktop,
broader native outage recovery, full-module quality, general reliability, or the
two-week pilot. Hermes v14's one pass and two failures remain separate.
The runner passed 149 offline checks, including prerequisite-integrity and
documentation cases; its import preflight passed before native execution.

Earlier run narratives below retain the evidence and open gates at their
recorded checkpoints; use the newest result and current task table for status.

The [selection-rejection package](validation/2026-09-20/selection-rejection-checkpoint-v1/README.md)
at `5155ebd` passes 1,638 source tests with four skips, all six source/package
checks, 655 isolated installed cases with four matching skips, and seven
replay assertions. Out-of-preview selected fact/source IDs receive a fixed,
noncommitting rejection while scope, digest, version, and deletion rules stay
unchanged. No native model has run this package. The failed v11 cohort below
belongs to preceding source `697b672`; its outcomes remain unchanged.

The [native Hermes v11 cohort](validation/2026-09-20/hermes-native-cohort-v11/README.md)
finished zero of three complete lifecycles on `697b672`. Every attempt passed
teaching, restarted recall, correction, and corrected recall, then failed at
forgetting. Attempt 1 deleted the target and two selected copies but retained
two user captures with the labels. Attempt 2 supplied out-of-preview source
IDs and received HTTP 500. Attempt 3 obtained a preview without applying it.
All 50 model responses had verified 65,536-token context and positive GPU
residency; cleanup and independently inspected control state passed for all
three attempts. No source replay or final recall ran.

The Hermes-only setup does not install shared Codex AGENTS guidance, so this
cohort cannot measure that repair's effect. The planned three-host v6 run
remains NOT RUN because this package has no passing native prerequisite.
Earlier v10 remains one pass and two failures on its own package. Complete
capture, full-module quality, Desktop, and the operational pilot remain open.

The [three-host v5 attempt](validation/2026-09-20/three-host-native-v5/README.md) passed
both teaching phases, then failed at Hermes recall on unsupported scope
arguments. Query-only retries retrieved both facts but do not clear the
failure. Both service starts reported a running worker; native correction
was not reached. Generated project guidance explicitly requests the rejected
arguments and appears in all three Hermes provider requests. That instruction
conflict is repaired in source `697b672`; native confirmation remains open.
Cleanup passed; full continuity remains open.

The [three-host v4 attempt](validation/2026-09-20/three-host-native-v4/README.md) passed
Codex teaching, Claude teaching, and Hermes retrieval of both facts after a
service restart. Hermes correction then failed: an unsupported scope argument
was rejected, required inspection was omitted, and the committed correction
job remained pending. The final fact stayed at version 1. Later phases did
not run. Cleanup passed. The runner processes jobs after a successful native
phase; validation stopped before that maintenance step. Pending status in
this run does not establish a production worker defect.

The [v3 model-substitution failure](validation/2026-09-20/three-host-native-v3/README.md) and
[v2 correlation failure](validation/2026-09-20/three-host-native-v2/README.md) remain failed.
Full three-host continuity remains open. Hermes v10 separately retains one
complete bounded lifecycle pass and two failures.

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
The complete three-host native driver had not run at that checkpoint; v7 above now passes the bounded lifecycle.

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
| Claude Code 2.1.191 | Command hooks, transcript deltas, context injection, MCP tools, extraction-ineligible provisional capture when native identity is missing, and one bounded stopped-service outbox recovery observation | [Three-host v7](validation/2026-09-20/three-host-native-v7/README.md) passes this host's bounded shared lifecycle on `e6887de`, with all five modules disabled. [Claude outage v2](validation/2026-09-20/native-claude-outage-v2/README.md) on `a64afb7` preserves two native messages and three queued entries through service stop, restart, replay, and a second restart. Earlier evidence: [Shared lifecycle v9](validation/2026-09-18/native-codex-claude-continuity-v9/README.md) on `5af3d15` retrieved and corrected a Codex-taught fact, then reviewed and deleted the target plus five selected copies. [Claude lifecycle v18](validation/2026-09-17/native-claude-lifecycle-v18/README.md) on `dedfcfd` passed 15 bounded fact-lifecycle criteria. Mixed-host v4 on `fc72cc2` verified native queue replay and restart. | Complete capture, native real-model extraction, provider or in-flight crash recovery, and general startup reliability remain open. |
| Codex CLI 0.146.0 | Generated lifecycle hooks, context output, and MCP tools | [Codex outage v1](validation/2026-09-20/native-codex-outage-v1/README.md) on `9722d92` passes one stopped-service raw-capture recovery with two pending jobs and zero facts. [Three-host v7](validation/2026-09-20/three-host-native-v7/README.md) passes this host's bounded shared lifecycle on `e6887de`, with all five modules disabled. Earlier evidence: Shared lifecycle v9 taught the fact, retrieved Claude's corrected version after restart, and completed both post-forget queries without repeating the removed values. Mixed-host v4 verified same-event replay from both native hosts and unchanged records through restart. | Earlier Stop timeouts remain recorded. Complete capture across lifecycle failures, tail latency, real-model quality, and pilot reliability remain open. |
| Codex Desktop | Shared command-hook adapter and MCP fallback | No isolated native desktop conversation has been validated. | NOT RUN; CLI evidence does not establish desktop behavior. |
| Hermes Agent in WSL, isolated source `14a3463` with cwd-only patch | Optional MemoryProvider, lifecycle callbacks, entry-point discovery, and explicit Windows/WSL bridge | [Native v14](validation/2026-09-20/hermes-native-cohort-v14/README.md) completes one of three full lifecycles on `e6887de`, including correction, reviewed-copy deletion, replay suppression, restart, and final recall. All 35 serving-context checks and three cleanup checks pass. [Three-host v7](validation/2026-09-20/three-host-native-v7/README.md) separately passes the shared bounded lifecycle on the same product. | Two v14 attempts fail at omitted required actions. Five cognition modules were disabled. Live compatibility, built-in remove/replace reconciliation, complete capture, native outage recovery, and general reliability remain open. |
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
