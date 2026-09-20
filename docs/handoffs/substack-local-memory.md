# Mnemos local memory: research and Substack writing handoff

Status: working handoff, September 20, 2026. Engineering and validation remain in progress. Reconcile the final checkpoint and open results before treating this as a finished account.

## Assignment

Write approximately 2,500 words for Anthony Maio's Substack about the work of turning Mnemos's memory promises into a dependable local system. Explain what we found, what changed, what the experiments actually establish, and where the evidence stops. The reader should understand why remembering a fact becomes difficult when several agents can capture it, correct it, forget it, or lose contact with the service.

Apply [writing-in-anthonys-voice](C:/Users/Ant/.agents/skills/writing-in-anthonys-voice/SKILL.md), the skill Anthony explicitly selected. Read its embedded examples before drafting. Use connected explanatory prose, concrete examples, straight quotes, and standard US keyboard characters. Do not plant typos, manufacture personal reactions, or invent experiences. This is an article, so use the skill's article guidance rather than its quick social-post style. If transferring to another machine, include the skill file with this handoff.

Deliver an article draft, three possible titles, and a separate compact source/claim note. Keep the article close to 2,500 words, excluding source notes. Do not publish it.

## Read this first

The article's central question is whether several agents can share a memory
that remains useful after a correction, a restart, or a request to forget.
The work produced tested infrastructure and exposed unresolved behavior in
real model runs. Do not frame it as a completed launch or proof of the paper's
claimed advantages.

Start with the [original audit and plan](../plans/2026-09-16-functional-local-memory.md),
the suggested narrative and capture tradeoff below, the
[current claim ledger](../validation/2026-09-16/claim-evidence-ledger.md), and
[Anthony's voice skill](C:/Users/Ant/.agents/skills/writing-in-anthonys-voice/SKILL.md).
Choose three or four repairs that carry the explanation. Use their linked
records to check the details. The long checkpoint history below is a research
reference; the reader does not need a tour of every run or test count.

The strongest concrete examples are stable identities after replay, a
correction whose acknowledgement is lost, captured copies that survive a
requested deletion, and shared instructions that conflict with a host's tool
schema. Explain what each example establishes and what it leaves unresolved.
Keep fixture observations, native conversations, and proposed operating
choices distinct. Do not invent Anthony's reactions or daily-use experience.

Return the draft, titles, and claim notes as separate files. In the notes,
map each quantitative or completion claim to its exact checkpoint and source;
flag local evidence that needs a public citation before publication. Preserve
negative results even if they complicate the ending.

## Latest implementation and native evidence

The [native Hermes v12 cohort](../validation/2026-09-20/hermes-native-cohort-v12/README.md)
finished zero of three complete lifecycles on `5155ebd`. One attempt omitted
corrected retrieval, one stopped after preview, and one received the repaired
invalid-selection rejection before applying an incomplete preview twice and
falling back to target-only deletion. All 44 model responses had verified
65,536-token context and GPU residency. Cleanup and independent scope-control
inspection passed. No attempt reached replay or final recall.

The [review-rejection repair](../validation/2026-09-20/review-rejection-development-v1/README.md)
also corrects a missed client path: invalid selections could trigger replay of
valid pending captures after rejection. The earlier test used an invalid
capture and did not establish the claimed no-replay behavior. A shared
rejection map now governs decoding and the early return; incomplete previews,
duplicate selections, and dependency conflicts receive safe rejections.
Deletion and scope requirements are unchanged. Full package validation of
this successor remains pending; v12 tested the preceding installed package.

The checkout handoff now includes findings after portable packet v24. That
packet remains a dated working snapshot; refresh it after the next package
checkpoint before using these new findings in the article.

Packet v24 adds the source-association development measurements to the verified
package record and preceding native failures. The engineering goal remains open.

The [source-association repeat measurement](../validation/2026-09-20/source-association-measurement-v2/README.md)
completed 45 observations across five cases, three arms, and three deterministic
repetitions. After a measurement-helper repair, all non-lifecycle controls
passed, but no arm produced a target path or target top-five result. Nine
lifecycle controls failed because preserving a shared source conflicts with
the existing deletion behavior. The corrected fact survived at version 2 and
the forgotten fact stayed absent after replay and restart. These findings do
not establish retrieval utility. The [original failed measurement](../validation/2026-09-20/source-association-measurement-v1/README.md),
[helper regression](../validation/2026-09-20/source-association-visit-budget-v1/README.md),
and [frozen-vector preparation](../validation/2026-09-20/source-association-vectors-v1/README.md)
are retained. This gives the writer a concrete example of checking the
measurement itself without turning an unsuccessful experiment into a success.

The [selection-rejection package](../validation/2026-09-20/selection-rejection-checkpoint-v1/README.md)
at `5155ebd` passes 1,638 source tests with four skips, all six source/package
checks, 655 isolated installed cases with four matching skips, and seven
replay assertions. Out-of-preview selected fact/source IDs receive a fixed,
noncommitting rejection while scope, digest, version, and deletion rules stay
unchanged. Native v12 has now tested this package with the failures above.
The failed v11 cohort below belongs to preceding source `697b672`; its outcomes
remain unchanged.

The [native Hermes v11 cohort](../validation/2026-09-20/hermes-native-cohort-v11/README.md)
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

The [shared-guidance package checkpoint](../validation/2026-09-20/shared-guidance-checkpoint-v1/README.md)
records 1,615 source tests passed with four skips, six successful source/package
checks, 652 installed cases passed with four matching skips, and seven replay
assertions. It repairs the conflict seen in the v5 trace between shared
instructions and Hermes tool schemas. This is a verified instruction/dispatch
repair; its effect on native completion has not yet been demonstrated. The new
three-attempt prerequisite cohort failed all three lifecycles under unchanged requirements.

The [offline correction-worker repair](../validation/2026-09-20/three-host-worker-checkpoint-v1/README.md) fixes a benchmark setup defect. The default production service already starts its scheduler; the custom runner did not. The repaired runner passes a correction/restart regression and 167 offline cases. Native acceptance remains open. Packet v23 includes this record.

The portable v24 packet is a working snapshot through the source-association repeat measurement, verified selection-rejection package, and preceding native Hermes v11 failures. It includes the earlier partial cross-host success and the later v5 failure. Refresh the handoff and packet at final acceptance; this is enough to begin research, not a finished engineering account.

The [three-host v5 attempt](../validation/2026-09-20/three-host-native-v5/README.md) passed
both teaching phases, then failed at Hermes recall on unsupported scope
arguments. Query-only retries retrieved both facts but do not clear the
failure. Both service starts reported a running worker; native correction
was not reached. Generated project guidance explicitly requests the rejected
arguments and appears in all three Hermes provider requests. That instruction
conflict is repaired in source `697b672`; native confirmation remains open.
Cleanup passed; full continuity remains open.

The [three-host v4 attempt](../validation/2026-09-20/three-host-native-v4/README.md) passed
Codex teaching, Claude teaching, and Hermes retrieval of both facts after a
service restart. Hermes correction then failed: an unsupported scope argument
was rejected, required inspection was omitted, and the committed correction
job remained pending. The final fact stayed at version 1. Later phases did
not run. Cleanup passed. The runner processes jobs after a successful native
phase; validation stopped before that maintenance step. Pending status in
this run does not establish a production worker defect.

The [v3 model-substitution failure](../validation/2026-09-20/three-host-native-v3/README.md) and
[v2 correlation failure](../validation/2026-09-20/three-host-native-v2/README.md) remain failed.
Full three-host continuity remains open. Hermes v10 separately retains one
complete bounded lifecycle pass and two failures.

The [native Hermes v10 cohort](../validation/2026-09-20/hermes-native-cohort-v10/README.md)
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

The [stale-preview rejection repair](../validation/2026-09-20/stale-preview-checkpoint-v1/README.md)
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

The [context-verified Hermes v9 cohort](../validation/2026-09-20/hermes-native-cohort-v9/README.md)
finished zero of three full lifecycles on `e419c2a`. All 39 model responses
had verified 65,536-token serving context and positive GPU residency; all
three attempts established cleanup. Attempt 1 omitted restarted retrieval.
Attempt 2 passed through corrected recall, then inspected related candidates
without applying deletion. Attempt 3 reached three deletion calls, each
returning degraded HTTP 500 with no committed receipt. The stale-preview rejection was later reproduced and repaired as described
above; the original native result remains failed. No source replay or final recall ran.

Both forget previews reached subsequent recorded HTTP requests intact,
with ten candidates and ten required source IDs. The
[endpoint diagnostic](../validation/2026-09-20/ollama-context-boundary-v1/README.md)
found that the old alias served 16,384 tokens through the OpenAI endpoint
while native readiness loaded 65,536. Pinning the context in a new alias
repairs that configuration defect. This does not prove the cause of earlier
failures, and earlier requested settings are not effective-context proof.
The isolated Hermes source stayed pinned; the independently advanced live
interpreter installation and its dependencies were separately bound.
Native lifecycle, three-host continuity, complete capture, T18 quality,
Desktop, and the operational pilot remain open.

Subsequent diagnostics narrow that uncertainty. The [saved previews](../validation/2026-09-20/hermes-v8-preview-persistence-v1/README.md)
in v8 attempts 2 and 3 match the native tool results intact. A separate
[single instrumented attempt](../validation/2026-09-20/hermes-request-observation-v1/README.md)
recorded two outgoing teach requests with the complete source-selection
schema, thinking disabled, and the requested context setting. That field
did not prove the serving context. It failed at
teaching after omitting the job-status call and never reached forgetting.
This is limited request-boundary evidence, not a rerun or repair of v8.

The [fixed Hermes v8 cohort](../validation/2026-09-20/hermes-native-cohort-v8/README.md) ran the verified `e419c2a`
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

The verified [retrieval-view checkpoint](../validation/2026-09-20/retrieval-views-checkpoint-v1/README.md)
at `e419c2a` passes 1,500 source tests with 4 skips,
all six source/package checks, 648 isolated installed cases
with 4 matching skips, and seven replay assertions. Ordinary recall
now excludes whole captured conversations before retrieval limits. Explicit
evidence review returns those copies separately, with provenance and unverified
labels. Capture storage and related-forget inventory remain intact. This is a
verified API change. Native v8 subsequently ran this package and failed all
three lifecycles; v7 retains its original runtime and outcome. Native lifecycle,
three-host continuity, complete capture, quality acceptance, Desktop, and the
pilot remain open.

The updated CLI and MCP subprocess checks verify that the child imports the
same package as the parent. Older parent-process origin checks alone did not
establish that boundary for those subprocesses. Preserve the earlier test
results while keeping that limit on their installed-package interpretation.

The [fixed Hermes v7 cohort](../validation/2026-09-20/hermes-native-cohort-v7/README.md) on verified `1a11058`
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

The [captured-evidence contract review](../validation/2026-09-20/captured-evidence-contract-review.md)
traces why whole conversations entered ordinary memory context with sleep
extraction disabled. They were labeled unverified, but shared the returned
fact channel at the time of v7. The verified retrieval-view change now
preserves capture and exposes raw history through explicit evidence review.
It does not prove the cause of v7's omissions or relax stored-copy erasure
checks. Exact provider request bodies were not recorded in v7, so replaying
its public transcript cannot be called a faithful request reproduction.

The [fixed Hermes v6 cohort](../validation/2026-09-19/hermes-native-cohort-v6/README.md)
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

The [fixed Hermes v5 cohort](../validation/2026-09-19/hermes-native-cohort-v5/README.md)
on verified `fd83fb2` finished zero of three full lifecycles. All three
attempts passed teaching, restarted recall, correction, and restarted
corrected recall. Attempts 1 and 2 deleted the target and suppressed original
source replay, but older conversational copies remained in the final model
request. Attempt 3 announced a deletion without calling the tool. No
provider-error gate failed. The earlier v3 and v4 outcomes remain preserved.
Three-host continuity, complete capture, real-model quality acceptance,
Desktop, and the pilot remain open.

The [fixed Hermes v4 cohort](../validation/2026-09-19/hermes-native-cohort-v4/README.md)
ran all three planned attempts on verified runtime `8f8f63f`. All three teach
phases saved the exact fact and called job status. Full lifecycles still
finished zero for three. Attempt 1 passed correction, restarted recall,
forgetting, and known-source replay, then retrieved the deleted value from
newly captured assistant narration. Attempt 2 failed on locale-encoded bridge
output. Attempt 3 retrieved and inspected a fact, then announced a correction
without calling the correction tool. Preserve the difference between a model
claim and an observed mutation. The three-host driver was unrun at that earlier checkpoint; see the current v4 partial result and failure above.

The earlier [v3 cohort](../validation/2026-09-19/hermes-native-cohort-v3/README.md)
also failed all three attempts, at teach, because each model omitted the
job-status call. Its results remain separate. V4's later progress is useful
evidence, but is not a completed integration or a controlled estimate of the
guidance change's effect.

The packet now includes [dated website and paper snapshots](../validation/2026-09-19/writer-primary-sources-v1/README.md),
with original URLs, retrieval timestamps, response metadata, and hashes. These
preserve the current public sources, not a proven snapshot of the original
audit date. Cite the public URLs and distinguish their promises from later
local validation. The paper's broader competitor and neuroscience claims
still need independent primary-source research before repeating them.

## Starting point and source map

The initial question was the gap between [the Mnemos website](https://mnemos.making-minds.ai/) and [the paper](https://making-minds.ai/papers/mnemos.pdf), and what it would take to make the software functional local memory. Read those primary sources directly and record the versions or retrieval dates. Do not treat the current website as an immutable record of the original audit.

Local checkout: `D:/Development/mnemos/mnemos/.worktrees/functional-local-memory`, branch `wt/functional-local-memory`. Repository-relative links below resolve within that checkout. A writer without repository access will need the linked documents and relevant evidence artifacts exported alongside this handoff.

| Question | Starting evidence |
| --- | --- |
| What was promised and what was missing? | [Original audit and T01-T19 plan](../plans/2026-09-16-functional-local-memory.md), especially capability inventory and paper reconciliation |
| What is implemented and what remains open? | [Implementation status](../validation/2026-09-16/implementation-status.md), [claim/evidence ledger](../validation/2026-09-16/claim-evidence-ledger.md), and [progress](../validation/2026-09-16/progress.md) |
| How does the memory service work? | [Functional local memory](../functional-local-memory.md) and implementation at the cited commit |
| Which host claims have native evidence? | [Client compatibility matrix](../client-compatibility-matrix.md) and its linked native-host records |
| What did the comparisons show? | [Evaluation status](../validation/2026-09-16/evaluation-running.md), [comparison diagnostics](../validation/2026-09-19/comparison-diagnostics.md), and raw reports linked from them |
| What should change in the research claims? | [Paper errata](../paper-errata.md) and the claim/evidence ledger |

Status prose can lag newer artifacts. Resolve disagreements against the commit-bound test reports and raw evaluation outputs, and retain the discrepancy in research notes until corrected.

## Suggested narrative, about 2,500 words

1. **The practical question, 250 words.** Several coding assistants should share useful memory across sessions and projects. Introduce one concrete failure that explains why that is harder than saving text. Use a verified fixture example and label it as such if no observed daily-use example exists.
2. **The promise and the audit, 400 words.** Explain the five modules in plain language: deciding what is surprising enough to retain, updating memories from evidence, using affective state in retrieval, background consolidation, and associative graph retrieval. Separate biological inspiration from measured software behavior.
3. **What dependable memory required, 700 words.** Follow one fact through capture, durable acknowledgement, extraction, scoped retrieval, correction, restart, and forgetting. Use a few actual repairs to make the consequences tangible: scope isolation, stable event identities and retries, revisions and derived graph state, write checks, and suppression of forgotten content.
4. **What the experiments exposed, 650 words.** Explain why passing deterministic tests does not prove extraction quality or better recall. Include the real-model comparisons, negative results, and the difference between a faster client fixture and faster end-to-end memory. State incomplete experiments plainly.
5. **What can be claimed now, 350 words.** Give the final observed state of Codex, Claude Code, and Hermes integration, plus the pilot status. Explain practical operating choices only where the evidence supports them.
6. **Where this leaves the work, 150 words.** End at the actual engineering result. An unresolved research question is acceptable. Do not invent a successful release or a daily-use experience to produce a neat ending.

The hardware interruption is available as context: Anthony reported that the remote dual-3090 machine failed and moved the work to a local RTX 4080 with 16 GB. Treat the failure history as user-reported; do not infer an electrical cause. Colab was considered and tabled. Include this only if it helps explain the experimental constraints.

## Capture choices worth explaining

Anthony asked for advice about automatic conversation capture versus saving
only facts an agent explicitly selects. Present this as an engineering
tradeoff, not a settled personal preference or a proven deployment decision.
Automatic capture preserves source context and reduces omissions, but adds
noise, extraction cost, sensitive material, and opportunities to recapture
stale assistant claims. Explicit selection makes intent clearer and reduces
volume, but depends on the agent noticing what matters and actually calling
the tool. The native v4 and v5 results illustrate both problems directly.

The proposed operating approach is a hybrid: opt-in local capture as source
material, conservative promotion into durable facts, explicit remember and
correct actions when intent is clear, source-backed retrieval, and deletion
that covers known copies and replay. Captured conversation is not itself an
asserted fact. Describe this as the design direction; the open capture,
quality, and lifecycle gates prevent a claim that daily operation is already
dependable. Do not turn automatic capture into permission to retain every
project or all private conversation indefinitely.

## Current evaluation update

The [local model tool diagnostic](../validation/2026-09-19/model-forget-development-v1/README.md) on installed
`e454a85` finished with Qwen3.5:9b passing 2 of three cases and
GPT-OSS:20b passing 0 of three. Each scheduled case ran once;
failures and startup timeouts remain in the archive. Passing cases require
observed inspection and preview, selected copies and their required sources,
persisted erasure and suppression, and unchanged own-project and foreign
controls. The runner also rechecks state through a read-only connection
after service shutdown. Its 68 focused offline tests, formatting, and
103-file type check pass. These counts are separate from the earlier full
package checkpoint.

This is a small development comparison with synthetic seeded facts,
SimpleEmbeddingProvider, and all five cognition modules disabled. It does
not automatically capture the conversation and is not a native Hermes,
restart, replay, extraction-quality, or three-host result. The native v6
failures and T18 negatives remain unchanged. The next native experiment
needs a separately frozen host/model profile and the existing erasure gates.

The failure modes matter more than the small pass counts. Qwen timed out
before responding in its first case. GPT-OSS timed out in its first two;
in the third, it deleted only the target with empty additional selections,
retained all three known copies, and claimed every related copy was gone.
Qwen's two passing cases did erase the seeded copies, but its final replies
repeated the deleted values and IDs. Since this diagnostic disables capture,
those passes cannot establish safe native recapture behavior. Use this as a
concrete example of why a tool receipt, persisted state, and an agent's final
claim need separate evidence.

The [historical-assertion contract review](../plans/2026-09-19-historical-assertion-contract-review.md)
examines the missing-correction-target rows. The model returned valid
`no_durable_facts` for historical wording, leaving no accepted fact for the
target-based correction operation. This establishes an ambiguity between
the fixture's expectations and the extraction contract, not a proven runtime
bug. A proposed 12-case development matrix needs independent review before
changing that contract. It does not replace the old labels or count as a
new quality result.

The [local model readiness archive](../validation/2026-09-19/local-model-readiness-v1/README.md)
records another operational constraint: the old GPT-OSS tag was present but
could not be read by Ollama. Its replacement loaded on the RTX 4080, with
about 12.8 GB reported in VRAM at a 32,768-token context. Preserve the failed
pull-client timeout and the readiness helper's punctuation mismatch when
describing this sequence. Successful loading is not evidence of memory
quality or native-host reliability.

Native evidence has a further qualification: the new [offline trace checks](../validation/2026-09-19/three-host-runner-offline-v1/README.md)
read six historical Codex/Claude sessions. Claude exposes directly correlated
tool-use/results. Codex's code-mode cells and MCP completions use different
identities, so the parser retains operation observations without claiming a
direct model-call match. The prospective three-host test still needs this
correlation gap resolved. Passing the parser's offline tests is not a new
native lifecycle result.

The independent encoding comparison ended on September 19 at 22:24:29 UTC
with all three reports marked incomplete. Of 600 observations per arm, the
full pipeline has 11 partial rows, no-surprisal has eight, and length-noise
has 15. Preserve these 34 partial rows out of 1,800 when discussing results;
metrics that exclude them are conditional diagnostics. Retrieval began next
and finished with incomplete coverage. The [terminal encoding archive](../validation/2026-09-19/encoding-terminal-local-v1/README.md)
preserves the original reports and explains denominators and conditional
paired differences. The subsequent [terminal retrieval archive](../validation/2026-09-19/retrieval-terminal-local-v1/README.md)
contains 589 complete and 11 partial observations, 900 query rows, and 3,600
strategy results. Its three alternative strategies have zero paired quality
delta against full on eligible rows. Neither archive closes complete-run
quality acceptance.

## Latest verified repair

The earlier verified [deletion coverage checkpoint](../validation/2026-09-19/forget-coverage-checkpoint-v1/README.md)
at `1a11058` passes 1,439 source tests with 4 skips, all six
source/package checks, 608 isolated installed cases with 4 matching
skips, and seven replay assertions. Related-forget receipts now report
target-only or reviewed selection and retained exact, advisory, and eligible
source counts within the validated preview at commit. Direct forget has no
preview counts. Both explicitly leave semantic erasure unproven. Tool guidance
asks the model to report that coverage. Native v7 ran this package and
failed all three lifecycles before any deletion receipt was reached.
Earlier native and model results retain their original runtime bindings.

Wheel SHA-256: `4e3d16d00f7b9ab54fe1e5350f7cf0fbe6209dc881bfc6d7d9f974f1c3f697a5`.
The [four new regression cases](../../tests/test_forget_receipts.py) verify
coverage without broadening deletion, preserve unselected advisory facts and
foreign controls, and keep semantic erasure unproven even when no preview
records remain. The direct GPT-OSS failure above motivated this change;
whether a model uses the clearer receipt correctly remains untested because
none of the v7 attempts reached it.

## Previous source-selection rejection repair

The [source-selection rejection checkpoint](../validation/2026-09-19/related-forget-error-checkpoint-v1/README.md)
at `e454a85` passes 1,427 source tests with 4 skips, all six source/package
checks, 604 isolated installed cases with 4 matching skips, and seven
replay assertions. Wheel SHA-256:
`94011300cda669ddcabefe9e9077af0f2aa01b6ed5f23bc13c8e6b7f2a5fbf35`.

Native v6 attempt 3 exposed a useful distinction: storage correctly refused
to delete selected copies without their required source events, but reported
the refusal as HTTP 500. The repair gives that specific validation failure a
typed HTTP 409 response with a fixed explanation. The client and Hermes
preserve it as rejected and noncommitted, without retrying the invalid
mutation or draining pending captures. Other validation paths remain separate.
The [regressions](../../tests/test_related_forget_service_errors.py) cover
rollback, correct explicit selection, and safe transport behavior. No native
model has run this package, and clearer errors do not establish better model
selection or completed erasure. Keep the v6 results above unchanged.

## Package used by native v6

The [Hermes reviewed-copy checkpoint](../validation/2026-09-19/hermes-reviewed-forget-checkpoint-v1/README.md)
at `db932be` passes 1,383 source tests with 4 skips, all six source/package
checks, 595 isolated installed cases with 4 matching skips, and seven
replay assertions. Wheel SHA-256:
`7df060463b35853d97898ca839b500a66e233bef26b8e52a7de1db799681190a`.

It exposes the existing preview/apply workflow as two additional Hermes
tools. The provider supplies scope; the caller supplies exact reviewed IDs,
the unchanged preview digest, and its matching limit. Empty selections
remove only the target. Fixtures demonstrate selected deletion, rejection of
a preview made stale by later capture, foreign-scope exclusion, and retention
of unselected advisory matches. These are service and package observations,
not native-model success. V6 subsequently exercised the two new tools,
but all three attempts failed at forgetting. The earlier v5 failures remain
unchanged. Distinguish a valid target-only deletion from erasing retained
conversational copies when the user requests both.

Explain the completeness limit carefully: the returned selection and its
dependencies fit the preview bound, but semantic search can still be
nonexhaustive or truncated. A tool named "forget" cannot establish that every
paraphrase has disappeared. Do not turn this repair into a claim of universal
erasure or completed cross-host continuity.

## Package used by native v5

The [native failure repair checkpoint](../validation/2026-09-19/native-repairs-checkpoint-v1/README.md)
at `fd83fb2` passes 1,351 source tests with 4 skips, all six source/package
checks, 569 installed cases with 4 matching skips, and seven replay
assertions. Wheel SHA-256: `4958c5ef7a509fefecacc0666d87f578b46e94e1a33a896831468c0f971bf895`.
It fixes explicit UTF-8 bridge output, typed missing-fact errors, and delayed
assistant capture that quotes a forgotten fact ID. The end-to-end regression
fails against preserved `8f8f63f` source and passes after the repair, including
restart, separate scope controls, and explicit reteaching with a new ID.

The assistant-ID safeguard conservatively suppresses an entire automatic
assistant event when it mixes an erased ID with unrelated material. It covers
exact token IDs, including native UUIDs; it does not establish arbitrary
paraphrase erasure or deletion of every historical raw source. User captures
and explicit writes remain eligible. The three earlier native v4 failures are
unchanged, and the subsequent v5 cohort still finished zero of three. Its surviving
older copies and omitted deletion are distinct from v4's newly recaptured
narration and bridge encoding failures.

## Package used by native v4

The [affective parsing and write-guidance checkpoint](../validation/2026-09-19/affective-guidance-checkpoint-v1/README.md)
at `8f8f63f` passes 1,324 source tests with 4 skips, all six source/package
checks, 543 installed cases with 2 matching skips, and the installed replay
probe. Its wheel SHA-256 is
`53444f7f87141e7107fe9f4daf4a6095e04c9d69d007cd218634fd2fff50f99f`.
This is the package used by native v4. The parser accepts signed decimals and
scientific notation, rejects malformed or nonfinite triplets, and preserves
prior state on failure. Replaying 300 archived responses changed zero states,
so the original 33/300 classifier-fidelity result remains negative. The
[fidelity review](../validation/2026-09-19/affective-fidelity-review.md) also
explains that the gold vectors were authored synthetic labels, with no
independent human calibration in the archive.

## Earlier verified repairs

The [job lookup error checkpoint](../validation/2026-09-19/job-not-found-checkpoint-v1/README.md)
at `68693f8` passes 1,308 source tests with 4 skips, all six
source/package checks, 533 installed cases with 2 matching skips,
and 7 replay assertions. Missing and out-of-scope jobs return the same
`404/job_not_found` response. Unexpected internal lookup errors remain 500,
and direct callers retain the `KeyError` contract. This repair leaves the
failed native lifecycle intact: it does not substitute the right job ID for
the model or claim that its pending write finished processing.

That checkpoint's wheel SHA-256 is `2e121c1ae81717d1b9a32349bd27ea7fafc2e5562e6eadbd7d24ddecd16f5cb8`. The prior package below
remains the runtime used for both failed native lifecycle attempts; do not
attribute those runs to this newer wheel.

The preceding [Hermes native identity checkpoint](../validation/2026-09-19/hermes-native-identity-checkpoint-v1/README.md)
at `3730fd9` passes 1,247 source tests with four skips, all six source/package
checks, 530 isolated installed cases with two matching skips, and seven replay
assertions. Native persisted message IDs now distinguish a regenerated turn
from a duplicate delivery. Reusing a native ID with changed content or role
fails as an identity conflict, including after service restart. This package
retains the earlier correction retry repair.

The [actual Hermes database fixture](../validation/2026-09-19/hermes-native-rows-contract-v3/README.md)
preserves the first turn's IDs during rewind and creates new IDs for an
identical regenerated last turn. Checkpoints, normal resume, and provider
recreation leave exactly six queued occurrences. This uses actual host
persistence and callbacks, without a model, service, or interactive CLI.
ID-less transcripts, loaders that omit IDs, compaction that clones rows, and
migration from existing prefix identities remain separate limitations.

The preceding native-identity wheel has SHA-256
`3772398a8e11b92d2cffd285b0571a733a9fd63cbdf932c30519d933cf0a6dd2`.
This is a tested package checkpoint, not final project acceptance. A useful
article example is that identical words can be a replay of an old observation
or a genuinely new turn. The host already had durable IDs that distinguish
them; the adapter needed to preserve those IDs and reject conflicting payloads.

## Evidence boundaries already established

- Earlier verified package checkpoint: `5e4b057b16906c2ec7dd302ae80748ef1ac5eddc`, with 1,195 source passes and four skips, all six source/package checks, 507 isolated installed contract passes and two matching skips, plus seven replay assertions. Wheel SHA-256: `7eaba268fb91ffae7d59b222cf8addb49d8b4b4857a37552e5404422e4009625`. The [Hermes lifecycle checkpoint](../validation/2026-09-19/hermes-lifecycle-checkpoints-v1/README.md) verifies scoped explicit tools, transcript identities across provider recreation, and extraction-ineligible provisional capture. Its installed Hermes API v2 run uses fixture providers and the actual Windows/WSL bridge, with no model conversation or outage. The archive also preserves two gaps found after the earlier candidate's suite passed. Overall acceptance remains open.

- The preceding `de6cdcf1c05fdad1bb19ed8f52638eff5ed67ae4` [recovery checkpoint](../validation/2026-09-19/loopback-recovery-checkpoint-v1/README.md) had 1,160 source passes and four skips, 474 isolated installed contract passes and two matching skips, plus seven replay assertions. It preserves both preceding full-suite failures and the later passing package evidence.

- An earlier fully verified checkpoint was `8fc79f01ab68d9c211f9be4818f9e566e271cefd`: 1,143 source tests passed, four skipped; the isolated installed wheel passed 440 selected cases, two skipped, plus seven replay assertions. See the [checkpoint](../validation/2026-09-19/retrieval-parity-checkpoint-v1/README.md). These are historical counts, not a claim about the eventual final build.
- The later `d0a344056dc9f93e10098ed405fa226095457066` run had 1,150 passes, one recovery-test failure, and four skips. A reproduced client-construction delay consumed the readiness budget. The repair is included in the verified `de6cdcf` package above; preserve the failed run when explaining how the defect was found.
- Local real-model evaluation uses Qwen3.5:9b and nomic-embed-text through Ollama. Cite exact digests, settings, hardware and source commit from the relevant run manifest; do not combine different checkpoints into one result.
- The September 19 lifecycle queue ended incomplete in five of seven configurations. Baseline and the configuration without sleep consolidation completed. Encoding/retrieval follow-on comparisons did not start in that queue. An [independent v2 queue](../validation/2026-09-19/supplemental-launch-v2/README.md) subsequently produced terminal but incomplete encoding and retrieval archives under the same frozen bindings. Completion alone does not establish that a configuration passed its quality gates.
- The [terminal lifecycle archive](../validation/2026-09-19/lifecycle-incomplete-local-v2/README.md) preserves all seven reports and their raw outputs. Fifty of 4,200 observations were partial because extraction returned no durable facts and a later correction could not resolve its target. No provider timeout caused those partial rows. Do not describe this as merely a failed benchmark runner, or retry only the failed observations until they succeed.
- The [extraction diagnosis](../validation/2026-09-19/lifecycle-extraction-diagnosis.md) identifies a more specific issue: those model responses reject "The earlier project note said ..." as historical rather than current evidence, while the fixture expects acquisition of a superseded fact. This is evidence of a temporal/assertion ambiguity. It does not prove the model was right to discard the information, and it is not permission to rewrite the held-out labels. The distinction could make a useful article example if the final account preserves those qualifications.
- Graph utility and end-to-end affective benefit remain unproved. Structural tests, rankings produced from supplied reference states, and classifier fidelity measure different things. Report their denominators and limitations separately.
- Storage-placement and HTTP client experiments isolate particular costs. Their gains cannot be presented as the same gains in complete real-model retrieval. Preserve failed latency measurements alongside successors.
- Three [baseline SSD performance repetitions](../validation/2026-09-19/performance-recovery-v1/README.md) on `de6cdcf` passed both unchanged latency gates: warm recall p95 245.214-252.222 ms, durable capture p95 28.804-36.062 ms, all 150 captures committed. Each run used 33 timed queries with real Nomic embeddings, 10,000 synthetic stored vectors, and fixture capture embeddings, with all cognition modules disabled. The first seed embedding took 6,691.178 ms, recorded separately from warm recall. This is a useful result with a specific scope, not evidence of semantic accuracy, full-module latency, or a native-host conversation.
- Native adapter tests, observed native host conversations, Codex Desktop, complete automatic capture, and a two-week pilot are separate evidence categories. Do not promote one into another.
- The [September 19 native Hermes attempts](../validation/2026-09-19/native-hermes-local-v1/README.md) completed local model turns, but failed acceptance because the installed CLI omitted the provider's session directory. A correct answer obtained through an unintended process-directory fallback is not evidence of safe scope isolation. The adapter repair closes that fallback; later patched-host runs validate explicit propagation in an isolated checkout. Built-in remove/replace notifications also lack exact Mnemos targets and remain unresolved. The diagnostic queue is in-memory only.
- The later [native lifecycle attempt](../validation/2026-09-19/hermes-native-lifecycle-local-v1/README.md) stopped at teach. Its `mnemos_remember` response reported `status: committed` with `processing_status: pending`, but the model omitted the exact target's final period and an auxiliary title-generation request lacked the required context option. These are distinct failures from the earlier scope defect. The [v2 configuration preflight](../validation/2026-09-19/hermes-native-discovery-preflight-v3/README.md) confirms that the disposable host disables the optional title model without making a model or service request; the successor lifecycle has now also failed, as described below.
- The [native Hermes v2 lifecycle](../validation/2026-09-19/hermes-native-lifecycle-local-v2/README.md) fixed both setup problems: the exact-content write returned `status: committed` with its job still pending, and all three model requests carried the required context field. This established the requested setting, not the effective loaded context; the later endpoint diagnostic found those could differ. The model then altered the returned job ID when calling `mnemos_job_status`. That unknown ID produced HTTP 500 and a degraded result, so the evaluator stopped before maintenance and later phases. This does not establish a completed asserted fact or a passed lifecycle. Keep the model's argument-copy error and the service's error handling distinct; no argument was repaired on the model's behalf.
- An earlier [isolated patched-host run](../validation/2026-09-19/native-hermes-local-v3/README.md) passed explicit scope, recall, foreign-project exclusion and raw user/assistant persistence. It failed the all-captures gate because duplicate callbacks generated different timestamps for the same event ID. The outbox rejected that conflict correctly. Preserve the failure and any successor separately; a patched host is also a distinct compatibility claim from the installed version.

- The [native Hermes v4 successor](../validation/2026-09-19/native-hermes-local-v4/README.md) passes all 16 bounded checks. Four capture calls returned committed while exactly two raw conversation events persisted. The native turn took 24.592 seconds, with a separate 3.569-second cold/load preflight. It uses the proposed CLI patch in an isolated host checkout; the installed host is unchanged. Full native correction, forgetting, outage recovery, extraction quality, and daily-use readiness remain open.

The [Hermes rewind diagnostic](../validation/2026-09-19/hermes-rewind-diagnostic-v2/README.md)
provides another bounded negative result. The installed provider treats an
identical regenerated turn after a same-session rewind as the original
observation. Its outbox retains four records where six logical observations
are expected. This is a callback fixture, not an interactive `/undo` run or a
service crash test. The native-ID repair above addresses the supported
persisted-message path; the ID-less fallback still has this limitation.
Normal duplicate replay must continue deduplicating. The first diagnostic's false-positive zero-capture report is
preserved and explicitly rejected.

## Finalization checklist

The [correction retry checkpoint](../validation/2026-09-19/correction-retry-checkpoint-v1/README.md)
at `ea88dd4` passes 1,201 source tests and 513 installed cases, with four and
two respective skips, all six source/package checks, and seven replay
assertions. A correction that commits before its reply is lost can now recover
its original job after restart without overwriting a newer edit. This is a
concrete reliability result from fixture providers and actual HTTP/SQLite;
it does not change the frozen real-model experiment or its missing targets.

New evidence for reconciliation: the [Hermes API recovery probe](../validation/2026-09-19/hermes-api-recovery-v1/README.md)
passes outage buffering, required checkpoint durability, native-request replay
after explicit restoration, and forget suppression across fresh host processes
on `5e4b057`. It uses fixture providers. The [graph coverage diagnosis](../validation/2026-09-19/graph-coverage-diagnosis-v1/README.md)
finds every recorded target disconnected from the selected seed; changing a
traversal threshold cannot repair those missing paths. Both records belong in
the final packet refresh, with their limitations intact.

The follow-up [development degree comparison](../validation/2026-09-19/graph-degree-development-v2/README.md)
tests four, eight, and sixteen neighbors on saved development vectors. All 15
combinations finish; sixteen neighbors restore a usable path in one case,
but no target reaches the top five. This is a concrete example of a mechanism
working without delivering the intended answer. These are five variants of
one development template, not five independent real-world workflows.

Candidate examples for the article, with bounded evidence:

| Problem | Resulting behavior | Evidence and limit |
| --- | --- | --- |
| An identical last turn is regenerated after Hermes rewind | Native row IDs preserve the surviving prefix and identify two new occurrences | [Native identity tests](../../tests/test_hermes_native_identity.py) and actual host database fixture; ID-less fallback and compaction remain limited |
| A committed capture loses its HTTP acknowledgement | The outbox retains a stable event identity for durable replay | [Lost acknowledgement tests](../../tests/test_service_lost_ack.py); synthetic service/SQLite evidence |
| A correction finishes before its acknowledgement reaches the client | An unchanged retry recovers the original job without rewriting a newer fact; a fresh edit still needs the current version | [Correction retry tests](../../tests/test_memory_service.py) and the real HTTP lost-ack test; fixture providers, with restart, raw-source expiry, scope rejection, and forget suppression covered |
| Two storage backends break exact-score ties differently | Both order tied candidates by memory ID and return no candidates for empty/nonpositive requests | [Storage contract tests](../../tests/test_storage_retrieval_contract.py) and the retrieval-parity checkpoint; consistency does not establish better semantic recall |
| Client setup consumes a short health-check budget | Plaintext loopback calls skip unused certificate loading while preserving auth and transport restrictions | [Recovery tests](../../tests/test_client_stale_descriptor.py); repeated local transport measurements, not model inference |
| A competing launcher exits before the original owner recovers | Polling can use the remaining original startup deadline without another launch | Same recovery tests; this does not prove every native host can tolerate the entire startup interval |
| A Windows reader blocks atomic publication of a new runtime descriptor | Bounded sharing-error retries preserve the old descriptor until replacement succeeds | [Publication tests](../../tests/test_service_competing_recovery.py); actual Windows held-reader reproduction and persistent-failure checks |
| Repeated Hermes callbacks reuse an ID but generate different timestamps | A private ledger retains first-observation timestamps across provider recreation; transcript prefixes distinguish repeated messages from replay | [Transcript identity regressions](../../tests/test_hermes_transcript_identity.py) and the installed API v2 record; arbitrary native transcript truncation and branching remain open |
| A callback lacks a usable transcript | The service retains a separate provisional receipt without extracting facts from it | [Provisional capture regression](../../tests/test_hermes_provisional_capture.py) and installed API v2; receipt counts cannot establish independent observations or complete capture |
| Hermes needs to correct or forget a specific fact | Provider tools enforce session scope, exact IDs and versions, and write request identities. Reviewed-copy preview/apply extends the original six tools to eight | [Tool regressions](../../tests/test_hermes_tools.py) and [reviewed-copy integration tests](../../tests/test_hermes_related_forget.py); older installed API v2 covers the original six. The native v6 cohort exercised both tools, but all three attempts failed at forgetting. The newer v7 cohort failed at correction, forgetting, and teaching respectively, without reaching preview/apply |
| Another client corrects or forgets a fact after Hermes retrieves it | Each ordinary prefetch reads the current service result | Same Hermes tests include a real service/SQLite correction-and-forget check; this removes a stale query cache but does not establish every native host lifecycle |

Choose the examples that support the narrative. This table is an evidence map,
not a requirement to recount every implementation detail in the article.

- [ ] Replace provisional status with the final verified source commit, wheel identity, source/installed checks, and artifact links.
- [ ] Reconcile every evaluation configuration, including failures and incomplete runs, with the final reports.
- [x] Add a concise before/after repair table with a source or regression test for each selected article example.
- [ ] Record final host continuity, latency, capture coverage, and pilot status without collapsing the separate gates.
- [ ] Identify which quantitative claims can be used, with units, sample sizes, settings, uncertainty where available, and limitations.
- [ ] Verify original website/paper claims against primary sources and link the dated corrections.
- [x] Prepare a portable working packet with this handoff, the selected voice skill, key reports, regression examples, and a file manifest. Refresh that packet at final acceptance. The portable v22 packet adds the verified shared-guidance package, worker repair, native v5 failure, and all three failed v11 attempts. It also includes the three-host v2 and v3 failures and the v4 teaching/restarted-recall result followed by correction failure, selected native records, and terminal encoding/retrieval reports, the two earlier native lifecycle failures, all eight fixed-cohort runs (v3-v9 retain twenty-one failed attempts; v10 adds one pass and two failures under its own binding; these are not a pooled reliability estimate), the deletion coverage and source-selection repair evidence, the six-case development diagnostic and model readiness records, the affective fidelity review, the endpoint context diagnostic and verified-context native successor, and dated website/paper snapshots. Some older evidence links still require the full checkout. Local validation artifacts are research inputs; public citations require accessible sources.
- [ ] Keep unverified interpretations and questions outside the article's factual claims, then make the final voice pass using Anthony's skill.
