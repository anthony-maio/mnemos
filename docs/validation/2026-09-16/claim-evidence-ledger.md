# Mnemos claim and evidence ledger

The [Claude outage successor](../2026-09-20/native-claude-outage-v2/README.md)
observes raw capture recovery for one conversation on verified product `a64afb7`.
Two native messages map to three queued entries including a provisional
prompt. Restart drains the outbox; independent database reads, replay, and
a second restart preserve raw sources and pending jobs. Zero facts and no
local provider responses are expected because extraction is disabled.
Cleanup passes. This is not a general outage or extraction-recovery claim.
The [package](../2026-09-20/autostart-environment-checkpoint-v1/README.md)
passes 1,834 source and 703 installed tests, four skips each, all six checks,
and seven replay assertions. The full-module matrix and pilot remain open.

The [native Claude outage attempt](../2026-09-20/native-claude-outage-failure-v1/README.md)
failed: the client ignored its autostart setting, and incomplete experiment
storage isolation let an unexpected owner write three synthetic events and
three jobs to the default database. Zero scoped facts were observed. The owner
was stopped and a verified private backup preserved; no cleanup deletion was
performed. Do not claim this attempt preserved an outage or avoided the default
store. The separately verified successor above does not change this outcome.

The [frozen retrieval fixture audit](../2026-09-20/retrieval-fixture-audit-v1/README.md)
identifies 20 foreign-scope queries mislabeled answerable and 60 explicit
remember bodies that differ from their gold facts. Original outcomes remain
106 hits and 183 misses among 289 eligible answerable observations, with
11 partial observations excluded. No corrected accuracy is claimed. The
[v4 repair](../2026-09-20/contract-v4-preparation-v1/README.md) is frozen on `3d09658`
and its new seven-profile matrix is running under the unchanged gates. No
terminal v4 result or old-to-new performance improvement is claimed.

The [native text audit](../2026-09-20/native-capture-coverage-v1/README.md)
matches 31 recorded native text identities to 43 service arrivals, separating
eight repeated deliveries and four provisional prompts. Twenty-nine identities
have completed jobs; two have suppression acknowledgements. This is bounded
receipt coverage, not 43 independent memories or general capture reliability.

The [three-host v7 run](../2026-09-20/three-host-native-v7/README.md) passed all 11 native phases on verified
product `e6887de`: Codex and Claude teaching, Hermes recall and same-ID
correction, corrected recall from all three hosts, Claude reviewed deletion,
and final recall from all three hosts after restart and source replay.
All 12 original-source replays were suppressed. Independent read-only SQLite
inspection confirmed the target absent, 18 suppression rows, and unchanged
review, unrelated, and foreign-scope controls. Cleanup passed.

This is one bounded lifecycle with real Nomic embeddings and all five
cognition modules disabled. It does not close complete capture, Codex Desktop,
native outage recovery, full-module quality, general reliability, or the
two-week pilot. Hermes v14's one pass and two failures remain separate.
The runner passed 149 offline checks, including prerequisite-integrity and
documentation cases; its import preflight passed before native execution.

Earlier run narratives below retain the evidence and open gates at their
recorded checkpoints; use the newest result and current task table for status.

The [native Hermes v14 cohort](../2026-09-20/hermes-native-cohort-v14/README.md)
completed one of three full lifecycles on `e6887de`. Attempt 2 passed reviewed
deletion, replay, restart, and final recall, copying the returned preconditions
exactly. Attempt 1 stopped after preview without applying deletion; attempt 3
stored the fact but omitted the required job-status check. All 35 responses
had verified context and GPU residency; cleanup and independent scope-control
inspection passed for all three. This supports a fresh cross-host prerequisite,
not a reliability claim. All five modules were disabled for this transport
baseline; broader readiness and quality gates remain open.

The [copyable-preconditions package](../2026-09-20/forget-preconditions-checkpoint-v1/README.md)
at `e6887de` passes 1,721 source tests with four skips, all six source/package
checks, 675 isolated installed cases with four matching skips, and seven
replay assertions. Complete forget previews now return the four apply
preconditions under their exact request names, without choosing records or
changing deletion checks. MCP's misleading preview description is corrected.
Native v14 above now tests this package; v13 below tested `db21750`.

The [clean-checkout failure](../2026-09-20/frozen-dataset-portability-v1/README.md)
is retained: Git line-ending conversion caused 15 benchmark hash failures.
The successor preserves the original dataset bytes through Git attributes;
the dataset, its original hash, and strict guard are unchanged. The
[saved graph seed audit](../2026-09-20/source-association-seed-audit-v1/README.md)
independently verifies 30 representative rows. It identifies seed/candidate
and score bottlenecks, with no new target reachability beyond semantic edges
and no improvement in target top-five retrieval. Existing negatives remain.

The [role-aware graph measurement](../2026-09-20/source-association-role-measurement-v1/README.md)
completed all 90 observations across five authored cases, two embedding
contracts, three relation arms, and three deterministic repetitions. Task
prefixes made the scope-isolation target reachable; source associations
improved its rank from 31 to 20 compared with semantic edges. No arm placed
any target in the top five. All non-lifecycle controls passed; 18 lifecycle
controls still failed on shared-source preservation. Independent verification
checked all 390 artifacts and the score/rank calculations. The focused suite
passed 31 tests. This is development evidence, with production defaults and
held-out results unchanged. Full-module quality remains open.

The [native Hermes v13 cohort](../2026-09-20/hermes-native-cohort-v13/README.md)
finished zero of three complete lifecycles on verified product `db21750`.
All three attempts passed teaching, restarted recall, correction, and corrected
recall. Two failed after applying limit 20 to previews with limit 100; the third
selected source IDs outside the preview and then committed target-only deletion.
All 55 model responses had verified context and GPU residency; cleanup and
independently inspected scope controls passed. No attempt reached replay or
final recall. This does not establish reliable reviewed-copy erasure.

The [paired embedding-prefix diagnostic](../2026-09-20/embedding-prefix-development-v1/README.md)
completed fresh measurements for five development cases. Task prefixes improved
several seed ranks, including 33 to 1 in the scope-isolation case, but neither
arm put any target in the top five. All 378 query/document vector bindings and
rank calculations were independently checked against raw responses. No graph
traversal ran, and production defaults and previous held-out results are unchanged.
The subsequent role-aware graph measurement above preserves those roles.

The [independent historical-assertion review](../2026-09-20/historical-assertion-independent-review.md)
found valid extraction abstentions followed by absent correction targets. It
does not establish a parser or target-resolution defect. The historical-memory
policy question is pending; prior labels and negative results remain fixed.

The [native Hermes v12 cohort](../2026-09-20/hermes-native-cohort-v12/README.md)
finished zero of three complete lifecycles on `5155ebd`. One attempt omitted
corrected retrieval, one stopped after preview, and one received the repaired
invalid-selection rejection before applying an incomplete preview twice and
falling back to target-only deletion. All 44 model responses had verified
65,536-token context and GPU residency. Cleanup and independent scope-control
inspection passed. No attempt reached replay or final recall.

The [review-rejection repair](../2026-09-20/review-rejection-development-v1/README.md)
also corrects a missed client path: invalid selections could trigger replay of
valid pending captures after rejection. The earlier test used an invalid
capture and did not establish the claimed no-replay behavior. A shared
rejection map now governs decoding and the early return; incomplete previews,
duplicate selections, and dependency conflicts receive safe rejections.
Deletion and scope requirements are unchanged. The [successor package checkpoint](../2026-09-20/review-rejection-checkpoint-v1/README.md)
at `db21750` passes 1,692 source tests with 4 skips, all six
source/package checks, 669 isolated installed cases with 4 matching
skips, and seven replay assertions. Native v13 has now tested this successor
with the failures above; v12 tested the preceding installed package.

The [selection-rejection package](../2026-09-20/selection-rejection-checkpoint-v1/README.md)
at `5155ebd` passes 1,638 source tests with four skips, all six source/package
checks, 655 isolated installed cases with four matching skips, and seven
replay assertions. Out-of-preview selected fact/source IDs receive a fixed,
noncommitting rejection while scope, digest, version, and deletion rules stay
unchanged. Native v12 has now tested this package with the failures above.
The failed v11 cohort below belongs to preceding source `697b672`; its outcomes
remain unchanged.

The [native Hermes v11 cohort](../2026-09-20/hermes-native-cohort-v11/README.md)
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

The [three-host v5 attempt](../2026-09-20/three-host-native-v5/README.md) passed
both teaching phases, then failed at Hermes recall on unsupported scope
arguments. Query-only retries retrieved both facts but do not clear the
failure. Both service starts reported a running worker; native correction
was not reached. Generated project guidance explicitly requests the rejected
arguments and appears in all three Hermes provider requests. That instruction
conflict is repaired in source `697b672`; native confirmation remains open.
Cleanup passed; full continuity remains open.

The [three-host v4 attempt](../2026-09-20/three-host-native-v4/README.md) passed
Codex teaching, Claude teaching, and Hermes retrieval of both facts after a
service restart. Hermes correction then failed: an unsupported scope argument
was rejected, required inspection was omitted, and the committed correction
job remained pending. The final fact stayed at version 1. Later phases did
not run. Cleanup passed. The runner processes jobs after a successful native
phase; validation stopped before that maintenance step. Pending status in
this run does not establish a production worker defect.

The [v3 model-substitution failure](../2026-09-20/three-host-native-v3/README.md) and
[v2 correlation failure](../2026-09-20/three-host-native-v2/README.md) remain failed.
Full three-host continuity remains open. Hermes v10 separately retains one
complete bounded lifecycle pass and two failures.

The [native Hermes v10 cohort](../2026-09-20/hermes-native-cohort-v10/README.md)
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

The [stale-preview rejection repair](../2026-09-20/stale-preview-checkpoint-v1/README.md)
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

The [context-verified Hermes v9 cohort](../2026-09-20/hermes-native-cohort-v9/README.md)
finished zero of three full lifecycles on `e419c2a`. All 39 model responses
had verified 65,536-token serving context and positive GPU residency; all
three attempts established cleanup. Attempt 1 omitted restarted retrieval.
Attempt 2 passed through corrected recall, then inspected related candidates
without applying deletion. Attempt 3 reached three deletion calls, each
returning degraded HTTP 500 with no committed receipt. The stale-preview rejection was later reproduced and repaired as described
above; the original native result remains failed. No source replay or final recall ran.

Both forget previews reached subsequent recorded HTTP requests intact,
with ten candidates and ten required source IDs. The
[endpoint diagnostic](../2026-09-20/ollama-context-boundary-v1/README.md)
found that the old alias served 16,384 tokens through the OpenAI endpoint
while native readiness loaded 65,536. Pinning the context in a new alias
repairs that configuration defect. This does not prove the cause of earlier
failures, and earlier requested settings are not effective-context proof.
The isolated Hermes source stayed pinned; the independently advanced live
interpreter installation and its dependencies were separately bound.
Native lifecycle, three-host continuity, complete capture, T18 quality,
Desktop, and the operational pilot remain open.

Subsequent diagnostics narrow that uncertainty. The [saved previews](../2026-09-20/hermes-v8-preview-persistence-v1/README.md)
in v8 attempts 2 and 3 match the native tool results intact. A separate
[single instrumented attempt](../2026-09-20/hermes-request-observation-v1/README.md)
recorded two outgoing teach requests with the complete source-selection
schema, thinking disabled, and the requested context setting. That field
did not prove the serving context. It failed at
teaching after omitting the job-status call and never reached forgetting.
This is limited request-boundary evidence, not a rerun or repair of v8.

The [fixed Hermes v8 cohort](../2026-09-20/hermes-native-cohort-v8/README.md) ran the verified `e419c2a`
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

The verified [retrieval-view checkpoint](../2026-09-20/retrieval-views-checkpoint-v1/README.md)
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

The [fixed Hermes v7 cohort](../2026-09-20/hermes-native-cohort-v7/README.md) on verified `1a11058`
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

The earlier verified [deletion coverage checkpoint](../2026-09-19/forget-coverage-checkpoint-v1/README.md)
at `1a11058` passes 1,439 source tests with 4 skips, all six
source/package checks, 608 isolated installed cases with 4 matching
skips, and seven replay assertions. Related-forget receipts now report
target-only or reviewed selection and retained exact, advisory, and eligible
source counts within the validated preview at commit. Direct forget has no
preview counts. Both explicitly leave semantic erasure unproven. Tool guidance
asks the model to report that coverage. Native v7 ran this package and
failed all three lifecycles before any deletion receipt was reached.
Earlier native and model results retain their original runtime bindings.

Previous package checkpoint: [source-selection rejection repair](../2026-09-19/related-forget-error-checkpoint-v1/README.md)
on `e454a85` passes 1,427 source tests (4 skips), all six source/package
checks, 604 installed cases (4 matching skips), and seven replay assertions.
Omitting source events required by selected related facts now returns a
specific HTTP 409 rejection through the authenticated client and Hermes,
with no mutation, invalid-request retry, or incidental outbox replay. The
existing generic failures remain for other validation paths and unexpected
errors. No native model has run this repaired package. The v6 failures below
remain bound to their earlier `db932be` runtime and unchanged outcomes.

The [fixed Hermes v6 cohort](../2026-09-19/hermes-native-cohort-v6/README.md)
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

The [fixed Hermes v5 cohort](../2026-09-19/hermes-native-cohort-v5/README.md)
on verified `fd83fb2` finished zero of three full lifecycles. All three
attempts passed teaching, restarted recall, correction, and restarted
corrected recall. Attempts 1 and 2 deleted the target and suppressed original
source replay, but older conversational copies remained in the final model
request. Attempt 3 announced a deletion without calling the tool. No
provider-error gate failed. The earlier v3 and v4 outcomes remain preserved.
Three-host continuity, complete capture, real-model quality acceptance,
Desktop, and the pilot remain open.

## Earlier native-host observation: v2

The [fresh native Hermes v2 attempt](../2026-09-19/hermes-native-lifecycle-local-v2/README.md) stopped at teach. Exact-content
and provider-context setup were corrected, but the model changed the returned
job ID in its status request. The unknown job produced HTTP 500 and a degraded
result. No later lifecycle phase ran; the three-host driver remains NOT RUN.
The original and successor failures are retained separately. Runtime `3730fd9`
and its passing package checks remain unchanged.

Assessment date: September 19, 2026. This covers the unreleased
functional-local-memory worktree, not the installed public package or website.
The capability IDs come from the original implementation plan.

Earlier package evidence is the [Hermes native identity checkpoint](../2026-09-19/hermes-native-identity-checkpoint-v1/README.md)
at `3730fd9`: 1,247 source passes with four skips, six source/package checks,
530 installed passes with two matching skips, and seven replay assertions.
The [actual host persistence fixture](../2026-09-19/hermes-native-rows-contract-v3/README.md)
establishes stable prefix identity and distinct regenerated suffix identity
when Hermes supplies native row IDs. ID-less fallback, some resume loaders,
compaction clones, and existing identity migration remain limited. The
[new wheel discovery preflight](../2026-09-19/hermes-native-discovery-preflight-v2/README.md)
also passes without model or service calls. None of these closes native model
lifecycle or pilot acceptance.

Earlier package evidence is the [Hermes lifecycle checkpoint](../2026-09-19/hermes-lifecycle-checkpoints-v1/README.md)
at `5e4b057`: 1,195 source passes, four skips, six source/package checks,
507 installed contract passes, two matching skips, and seven replay assertions.
Its installed Hermes API v2 run passes scoped lifecycle tools and persistent
transcript identities, and retains provisional receipts without extracting
facts from them. It uses fixture providers; native model lifecycle and outage
acceptance remain open. The prior candidate's two later-discovered gaps and
failed installed regressions remain in the same archive.
The [isolated patched Hermes v4](../2026-09-19/native-hermes-local-v4/README.md)
passes 16 single-turn scope, recall, and raw-capture checks. The installed CLI
is unchanged, and full native lifecycle acceptance remains open. Earlier
package and native failures retain their original artifacts and bindings.
The local model matrix is now
terminal and incomplete in five profiles; 50 observations had missing
correction targets following `no_durable_facts` extraction. The
[terminal lifecycle archive](../2026-09-19/lifecycle-incomplete-local-v2/README.md)
does not establish T18 success. The encoding/retrieval follow-on comparisons
never started in that queue. The [independent v2 queue](../2026-09-19/supplemental-launch-v2/README.md)
has since started under unchanged bindings. Its [encoding comparison](../2026-09-19/encoding-terminal-local-v1/README.md)
is terminal with 34 partial observations out of 1,800. All automatic-policy
decisions were observed, but complete-only quality differences remain
conditional diagnostics. Retrieval is now terminal and incomplete; its
conditional paired diagnostics are preserved separately.
The [retrieval terminal archive](../2026-09-19/retrieval-terminal-local-v1/README.md)
is also incomplete: 589 of 600 session observations are complete, with 900
query rows and 3,600 strategy results. Hybrid, lexical, and vector have zero
paired quality delta against full on eligible rows. Neither comparison closes
quality acceptance.
Historical running-state descriptions below are superseded.

The [new baseline SSD performance evidence](../2026-09-19/performance-recovery-v1/README.md)
passes both unchanged latency gates in three repetitions: recall p95
245.214-252.222 ms and capture p95 28.804-36.062 ms, all 150 acknowledgements
committed. This uses real query embeddings, 10,000 synthetic stored vectors,
fixture capture embeddings, WAL/FULL durability and no cognition calls. It
does not establish semantic quality, full-module latency, native-host deadline
acceptance, or a causal speedup ratio against runs on different storage.

The [retrieval parity checkpoint](../2026-09-19/retrieval-parity-checkpoint-v1/README.md)
at `8fc79f0` passes 1,143 source tests, four skips, all six source/package checks,
440 newly observed installed contract cases, two matching skips, and seven
replay assertions. In-memory retrieval now matches SQLite on exact-score ties
and empty/nonpositive requests. All 46 baseline query/repetition pairs match
between backends in the repeated fixed-vector control. Both the controlled and
legacy replacement gates still fail. Backend parity is established for those
cases; semantic quality, real-model benefit, and host acceptance are not.

The [storage-location checkpoint](../2026-09-19/storage-location-checkpoint-v1/README.md)
at `7bcabde` passes 1,131 source tests, four skips, and all six source/package
checks. The wheel is byte-identical to the previously installed `39b9c79`
artifact; its 428 installed contract passes and seven replay assertions are
reused, not newly observed. Explicit benchmark storage paths now distinguish
SSD and HDD measurements. Three capture-only SSD repetitions pass the original
100 ms p95 gate; all three HDD repetitions fail. All 300 events committed.
Shared-machine I/O limits causal attribution; this is not native capture or
recall acceptance. The [raw placement evidence](../2026-09-19/capture-placement-v1/README.md)
preserves every repetition. The [fair offline retrieval control](../2026-09-19/fair-comparison-diagnostic-v1/README.md)
removes the earlier cross-project regression, but overall replacement gates
remain failed and synthetic vectors support no production superiority claim.

The [embedding-client checkpoint](../2026-09-19/embedding-client-checkpoint-v1/README.md)
at `39b9c79` passes 1,123 source tests, four skips, six source/package checks,
428 installed contract cases, two matching skips, and replay checks. Client
reuse reduces setup overhead in a fixed-vector HTTP fixture, without proving
real recall/capture latency success. The separate synthetic replacement gate
fails on both this source and `fc72cc2`; an all-CI-gates pass or superiority
claim is not supported. The running real-model matrix remains frozen at
`fc72cc2`, and earlier native observations retain their original bindings.

The latest [mixed-host replay and restart observation](../2026-09-18/native-crosshost-recovery-v4/README.md)
uses the installed `fc72cc2` package. Three same-event pairs delivered by native
Codex and Claude processes resolved to stable event/job identities. Eight normal
messages and two extraction-ineligible provisional records survived restart;
the outbox emptied and both scope controls survived. All four native turns
exited zero. One pairing timeout and degraded Claude identity diagnostics remain
recorded. Overlap was instrumented and fixture-service restoration was explicit.
This closes the bounded mixed-host replay observation, not automatic daemon
recovery, complete capture, real-provider quality, or daily-use acceptance.
Earlier records below retain their original source bindings and limitations.

The earlier [retrieval strategy checkpoint](../2026-09-18/retrieval-comparison-checkpoint-v1/README.md)
is bound to `fc72cc2`: 1,105 source passes, four skips, all six repository
checks, and nine installed adapter passes. It isolates vector, lexical, hybrid,
and full retrieval against identical accepted snapshots. Complete reports
require all expected comparison rows and verified provider provenance; quality
kinds have separate intervals. The wheel is byte-identical to the preceding
package. At that preparation checkpoint the artifact was `NOT RUN`, with zero observations; the later terminal retrieval archive supersedes that execution status. These
checks establish implementation behavior, not comparative model quality or
GPU, native-host, latency, or pilot acceptance.

The preceding [applied encoding comparison checkpoint](../2026-09-18/encoding-comparison-checkpoint-v1/README.md)
is bound to `aefc338`. Its 1,085 passing source tests, four skips, complete
source/package checks, 390 installed contract passes, two matching skips,
seven replay assertions, and ten installed comparator service passes support
implementation and package claims. They do not establish that surprisal
outperforms a length/noise rule. The three comparison artifacts are offline
preparations labeled `NOT RUN`; real-model observations and utility claims
remain open. Earlier observations below retain their original source bindings.

The newer [competing service recovery source check](../2026-09-18/competing-service-recovery-v1/README.md)
fixes a startup race: a second launcher could return a dead descriptor while
the new lock owner was still initializing. Competing launchers now verify
authenticated health before adopting an owner. Five added cases cover the
race, startup timeout, authentication rejection, and invalid health data.
The full source suite passes 1,054 tests with four skips; 47 focused checks
also pass. Simultaneous native host acceptance remains open; the package
checks below validate the installed service protocol.

The earlier [competing recovery package checkpoint](../2026-09-18/competing-recovery-package-checkpoint-v1/README.md)
at `a5a9101` passes 1,054 source tests with four skips, all six source/package
checks, 384 selected installed cases with two matching skips, and seven
synthetic replay assertions. The installed selection includes the five
competing-launcher regressions. Prior native observations retain their
original wheel bindings; this package does not establish native concurrency
or real-provider recovery.

The preceding [stale descriptor checkpoint](../2026-09-18/stale-descriptor-package-checkpoint-v1/README.md)
at `37e446f` passes 1,049 source tests with four skips, all six source/package
checks, 379 selected installed cases with two matching skips, and seven
synthetic replay assertions. Seven new cases cover delayed replacement,
live-owner reuse, failed or stalled startup, authentication rejection, and
an owner recovering while its competing launcher exits. The client probes
authenticated health before launching a replacement and waits past a known
stale descriptor without deleting it. Earlier native evidence remains bound
to its original wheel; configured real-provider recovery remains open.

The preceding [outbox selection checkpoint](../2026-09-18/outbox-selection-package-checkpoint-v1/README.md)
at `c25ed98` passes 1,042 source tests with four skips, all six source/package
checks, 372 selected installed cases with two matching skips, and seven
synthetic replay assertions. An explicitly supplied empty outbox now retains
its selected recovery file and stricter write policy. Three new cases cover
empty/nonempty selection, restart replay, and PII blocking. This checkpoint
adds package evidence; earlier native observations retain their original
source bindings.

The [storage-worker termination checkpoint](../2026-09-18/storage-process-kill-v1/README.md)
adds seven cases at `76940d2`: capture after event insertion, before commit,
and after commit; forget after suppression insertion, after source deletion,
before commit, and after commit. The actual SQLite worker is terminated.
Fresh reopen, 12-table hashes, integrity checks, repeated outbox replay,
derived-state cleanup, and retained local/foreign controls pass. The clean
source suite passes 1,067 tests with four skips, and all seven new cases
pass against the existing `983920a` wheel. Runtime/build Git objects and
installed wheel bytes are verified unchanged. These fixed-provider tests
cover selected storage boundaries; native host, provider, commit-instruction
interruption, and operational-pilot acceptance remain separate.

The [capture/forget transaction checkpoint](../2026-09-18/capture-forget-package-checkpoint-v1/README.md)
at `983920a` prevents late capture, replay, and correction from restoring
erased raw sources after a forget commits. SQLite rechecks source IDs,
verified Claude aliases, automatic assertion suppression, and forgotten
correction targets inside its write transaction. Fresh explicit reteaching
remains supported. Six added cases pass in both source and installed runs.
The full source suite passes 1,060 tests with four skips; the installed
wheel passes 390 selected cases with two matching skips and seven replay
assertions. An earlier full run had one degraded bridge result. Its failure
and subsequent passing diagnostics are preserved; the cause remains open.
This checkpoint does not rerun native hosts or change the frozen model evaluation.

A [concurrent native Codex recovery probe](../2026-09-18/native-codex-concurrent-recovery-v2/README.md)
on the installed `a5a9101` wheel used two separate Codex homes and one shared
outbox. Two outage turns queued four messages. After explicit fixture-service
restoration, two fresh native processes automatically replayed that queue.
A test barrier overlapped delivery of the same event, and Windows TCP ownership
plus process ancestry tied the requests to both native processes. All eight
messages across four sessions had one event, completed job, and evidence link
each. The outbox emptied, both controls survived, and restart preserved the
database. Generated hooks were unchanged and all four native processes exited
zero. The failed v1 home-path fixture is retained. Recovery-a displayed an MCP
startup interruption, and PowerShell emitted Import-Clixml diagnostics; those
PTY observations remain explicit limitations. This closes one bounded Codex
concurrency observation. The later mixed-host v4 observation above adds bounded
Codex/Claude evidence; real-provider recovery, complete native acceptance, and
the operational pilot remain open.

A [native startup-outage probe](../2026-09-18/native-codex-startup-outage-v1/README.md)
on the installed `c25ed98` wheel retained the real user prompt and final answer
in its outbox while malformed fixture configuration prevented service startup.
The native process exited zero; the original runner failed because the answer
omitted a requested period. That failure is preserved. A separate recovery
helper verified the actual messages, then explicitly started the fixture service
and replayed both events. An immediate second replay had no pending events.
After service restart, replaying the same events left all four core table counts
and complete-row hashes unchanged. Each message had exactly one stored event
and one evidence row. This establishes native outage capture followed by explicit
client recovery with fixture providers. That run did not exercise automatic
native-hook replay; the later two-session observation below does.

In a [fresh two-session probe](../2026-09-18/native-codex-native-recovery-v1/README.md),
the first native Codex session queued both messages during startup failure.
After explicit fixture-service restoration, the second session's unchanged
generated hooks drained that queue automatically, without a direct replay call.
Both processes exited zero. All four messages retained their correct session,
turn, role, project, and profile, with one evidence link each. The outbox emptied,
and complete-row hashes stayed unchanged across maintenance and service restart.
This verifies one native automatic replay observation. The exact triggering hook
was not traced; the separate process-kill observation follows below.

A [forced service-termination probe](../2026-09-18/native-codex-service-kill-v2/README.md)
then killed the actual disposable service after SQLite committed a native prompt
and before its HTTP response. The event and job survived under their original IDs.
After explicit fixture-service restoration replaced the dead owner's descriptor,
a fresh native Codex session automatically drained the queue. All four native
messages had one job and one evidence link each, with unchanged table hashes
across maintenance and restart. Both native processes exited zero. The first
session's Stop timeout and the second session's MCP startup warning remain
recorded as terminal observations; this is not an all-hooks acceptance result.
Arbitrary crash timing, concurrent hosts, real-provider recovery, and complete
host acceptance remain open. The failed v1 PID preflight is preserved separately.

Three fresh [native Codex sessions](../2026-09-18/native-codex-stop-repeat-v1/README.md)
on the `c25ed98` wheel used the generated hooks unchanged. The late native
hook start/completion pairs measured 1,015.009 ms, 1,031.318 ms, and
963.769 ms, each below the unchanged three-second deadline. Their placement
after the final answer and before turn completion identifies them as Stop by
temporal inference; the native log omits hook names and exit status. All three
sessions retrieved the scoped fact, excluded the foreign fact, and persisted
the user prompt and final answer across shutdown. A separate verifier checks
the evidence; the original harness failure reports remain preserved because
Codex added hook-trust metadata after approval. Three fixture observations
do not establish tail latency, the 100 ms durable-capture p95 gate, or complete
native lifecycle acceptance.

The preceding [Windows permission source validation](../2026-09-18/windows-private-acl-v1/README.md)
passes 1,039 tests with four skips and all static checks. Actual Windows ACL
tests exposed an incorrect inheritance-protection flag. The repair applies the
protected current-user policy and checks exact existing permissions before
skipping a write. Nine regressions ran, including later drift and failed-query
repair. Earlier package checks did not establish this protection property.
The [fresh installed package](../2026-09-18/windows-private-acl-package-checkpoint-v1/README.md)
at `5d8cacf` passes 369 selected cases with two skips and seven replay checks.
Its [native v3 probe](../2026-09-18/native-codex-private-acl-profile-v3/README.md)
completed and persisted both roles with protected fixture permissions. Shell
entry-to-exit was 876 ms with instrumentation, versus 2,052 ms in the preceding
old-wheel profile. These are individual observations, not a controlled speedup
estimate or repeated uninstrumented deadline acceptance.

The preceding [installed package validation](../2026-09-18/assistant-suppression-package-checkpoint-v1/README.md)
at `5af3d15` passes 1,030 tests with four skips, all six source/package checks,
360 selected installed cases with two matching skips, and seven synthetic replay
assertions. Exact normalized forget
suppressions now apply before source-role attribution and again at atomic
extraction commit using the persisted event role. Eight regression cases include
independent scope/profile isolation, unrelated proposals, explicit reteaching,
support preservation, and retained unknown paraphrases. Native v9 also includes
this repair; the frozen Ollama evaluation remains separately bound.

The preceding [similarity validation](../2026-09-18/related-forget-similarity-v1/README.md)
passes 1,022 tests with four skips and static checks. It adds bounded advisory
similarity discovery to the explicit related-forget review, with 13 regressions
covering selection, replay suppression, stale state, scope, and dependencies.
The vectors are controlled fixtures. Native acceptance and real embedding
quality are not established, and discovery is explicitly non-exhaustive.

The preceding [UUID source validation](../2026-09-17/uuid-write-safety-v1/README.md)
passes 1,009 tests with four skips, Black, and strict mypy after the shared UUID
write-safety repair. It includes the host-context authority changes. This is a
source checkpoint, now supplemented by the installed verification below.
Native-host evidence remains bound to its earlier source revisions.

The [native v9 run](../2026-09-18/native-codex-claude-continuity-v9/README.md)
on installed `5af3d15` passes 15 bounded lifecycle checks: cross-host correction
survives restart, reviewed deletion removes the target and five captured copies,
direct replay does not restore the target, and final Codex recall and answers
exclude both convention values while preserving the unrelated preference.
The run exercised similarity suggestions with fixture lexical vectors. All
Codex processes exited zero, but reverse and final Stop hooks timed out after
three seconds. Complete capture, native hook replay, Hermes, Desktop, and
real-model quality remain unverified.

The earlier [native v7 attempt](../2026-09-18/native-codex-claude-continuity-v7/README.md)
on installed `ac139dd` committed related deletion before Claude exceeded its
phase budget. SQLite reopen confirms the target and five selected copies absent,
with unrelated and foreign facts retained. Replay and final recall did not run.
[V8](../2026-09-18/native-codex-claude-continuity-v8/README.md) reached
the reverse Codex phase but timed out with tool approval pending. Its deadline
includes operator time; forgetting, replay, and final recall were not run.
Neither attempt establishes native post-forget acceptance.

The earlier [native v6 attempt](../2026-09-17/native-codex-claude-continuity-v6/README.md)
verified selected-fact deletion by Claude and direct service replay suppression
after Codex teaching and cross-host correction. Its final Codex answer still
repeated the deleted value from a surviving unknown captured statement, and the
host skipped the second required query. The overall result is FAILED. This is
not evidence of whole-topic erasure or accepted post-forget host behavior.
The [v5 wrong-query failure](../2026-09-17/native-codex-claude-continuity-v5/README.md)
remains preserved alongside it. The native fixtures do not use real cognition.

The preceding complete package verification was frozen at source `ac139dd`:
1,022 tests passed with four skips and its fresh installed wheel passed 352
selected contract cases. The current `5af3d15` package checkpoint supersedes it
for release binding. Source, fixtures, and installed environments were on C:,
outside the development checkout. Neither package checkpoint establishes native
plugin-session acceptance.
The interruption tests bind child imports to the parent package origin. The
new import guards likewise use fresh subprocesses and the installed package.
The earlier [capture interruption checkpoint](../2026-09-17/capture-durable-package-checkpoint-v1/README.md)
preserves its initial test-origin gap and the supplementary run that repaired it.

The [bidirectional native observation](../2026-09-17/native-codex-claude-continuity-v4/README.md)
on installed ebe1b0d verifies Codex store, Claude read/correct, and a fresh Codex
read of version 2 across two service restarts. Actual native completion records
and independent SQLite reopen support the claim. Foreign-project facts were
excluded. Both Codex processes still exited 0xc0000005, reverse Stop timed out,
and Claude capture remained degraded. No-integration controls exited zero;
the crash trigger remains unproven.
MCP-only and hooks-only controls likewise exited zero. The hooks-only control
retained both conversation roles across database close/reopen despite its
Stop timeout. These are bounded observations with fixture providers; see
[isolation controls](../2026-09-17/native-codex-isolation-controls-v1/README.md).

A later [combined control](../2026-09-17/native-codex-combined-control-v1/README.md)
retained hooks and MCP under explicit unelevated mode, retrieved the scoped
target, retained both capture roles across reopen, and exited zero. Stop still
timed out. One success does not explain or invalidate the earlier crashes.
The [Claude diagnostic repair](../2026-09-17/hook-diagnostic-reasons-v1/README.md)
adds fixed reason codes with 68 passing focused tests; it leaves identity
ambiguity degraded and is not new installed or native acceptance evidence.

The earlier [native captured-context observation](../2026-09-17/native-codex-claude-continuity-v2/README.md)
shows a corrected asserted target at version 2 alongside unknown captured
history, visibly qualified in Claude's actual retrieval context. The runner
failed a fixed-cardinality assertion and did not run its second service restart.
The [next attempt](../2026-09-17/native-codex-claude-continuity-v3/README.md)
failed at Codex shutdown before Claude started. These are preserved partial
observations, not passing end-to-end native acceptance.

The client now distinguishes an exited service from a live process reaching
its startup deadline, while preserving capture buffering and an existing
owner. Six startup cases pass in source and the installed wheel. A separate
[installed failure probe](../2026-09-17/startup-exit-diagnostic-v2/README.md)
returned the child's numeric exit code promptly. It does not explain the
earlier intermittent full-suite startup failure.

The lost-acknowledgment repair keeps capture on the durable outbox path when
the service commits an event but its HTTP reply is lost. Four injected HTTP
and SQLite cases cover ordinary and provisional captures, restart, and one
or two lost replies. All four fail before the repair and pass in source and
the installed wheel. The first full-suite attempt at source `6cde465` failed
an auto-start health check; an unchanged isolated diagnostic and a fresh full
run passed afterward. Its cause remains unproven, and the
[failed run](../2026-09-17/lost-ack-package-failure-v1/README.md) remains preserved.

The [concurrent revision regression](../2026-09-17/retrieval-concurrent-revision-v1/README.md)
reproduces stale candidate scoring and context-budget violations while a
correction commits during state classification. The repair reloads candidates
after that await. Six cases cover correction and forget, with graph expansion
on and off; all six fail on the old source and pass in the new source and wheel.
The active real-model run retains its original source and does not measure this
repair. Native Claude v17 remains bound to `8d865a5`; the separate v18 run uses
`dedfcfd` and includes this repair and the Claude startup default.

The earlier complete package checkpoint was frozen source `6600dc2`, with
865 tests passed and 42 installed contract cases. It remains historical
evidence and does not override the newer checkpoint. Source `56182d9` also remains
preserved with 898 passing source tests and 87 passing installed contract cases.

Historical source `312b8d6` passed 858 tests with four skips. Its
[fresh installed-wheel check](combined-package-checkpoint-v2/README.md) then
finished with 40 passing tests and two continuity failures. A subsequent
[installed diagnostic](installed-continuity-diagnostics-v1/README.md) reproduced
HTTP read timeouts and multi-second retrieval metadata updates. A later
[SQL timing profile](installed-continuity-sql-profile-v3/README.md) passed both
cases and located commit as the dominant storage cost in that run; the earlier
multi-second stalls did not recur and no repair is claimed. The separate installed replay
probe did not run. The new C: pass does not prove the earlier D: timeouts cannot
recur or close the 10,000-fact recall gate. The
[drive comparison](storage-drive-calibration-v1/README.md) supports storage
latency as a contributing factor, without establishing an exclusive cause.
The combined `9aba209` run's
benchmark compatibility failure and its repair remain preserved in the
[failed checkpoint](combined-package-failure-v1/README.md). Claude plugin runtime
repair `ec0f95e` passed 50 focused tests with two opt-in skips and is now included
in the `6600dc2` full source checks; native plugin-session acceptance remains open.
The [subsequent native plugin check](native-plugin-checkpoint-v1/README.md)
observed all three lifecycle hooks, seeded recall, MCP connection/tool inventory,
restart persistence, and reconstructed Stop replay. Its original exact-two raw
capture gate still fails because a separate provisional user event remains;
that event's exclusion from fact extraction passes. Native MCP tool execution
and the complete correction/forget conversation remain unproved.
The [v11 native read attempt](../2026-09-17/native-claude-mcp-v11/README.md)
subsequently left the MCP server pending and made no tool calls. Its successful
hook-context recall did not close native MCP retrieval acceptance.
The [v12 native read check](../2026-09-17/native-claude-mcp-v12/README.md)
observed successful health and retrieve calls, matching the intended
database/profile/project and returning the project-scoped seed. Its bounded
read, identity, restart, and replay criteria all passed. An offline diagnosis
of v13 recovered the nested correction payload and verified its target and
replacement hash, but the native attempt still contained degraded capture
diagnostics and did not run restart or forget. The [v15 attempt](../2026-09-17/native-claude-lifecycle-v15/README.md)
reported a pending plugin with zero tools and made no native calls. Native
correction and forgetting were unobserved at that point. Extraction and complete
cross-host continuity remain open.
The separate [hook-time identity observation](../2026-09-17/native-claude-hook-identity-v3/README.md)
found no prompt ID and no transcript file at UserPromptSubmit; identifiers
appeared by Stop. Both capture hooks still emitted diagnostics, so this
association limit remains an open native acceptance condition.
The [v16 startup-wait control](../2026-09-17/native-claude-lifecycle-v16/README.md)
then observed a connected catalog and passed all six native correction-phase
checks. Its direct post-maintenance retrieval failed before restart and forget,
so the complete lifecycle remains failed. Degraded hook capture is retained
separately from the successful native correction operation.
The later [v17 lifecycle check](../2026-09-17/native-claude-lifecycle-v17/README.md)
at installed `8d865a5` passed all 15 fact-lifecycle criteria: native correction,
consolidation, retrieval, forgetting, restart persistence, no resurrection after
reconstructed Stop replay, and preservation of an unrelated fact. Its lexical
vectors are a transport fixture, not a semantic-quality evaluation. Degraded
hook capture keeps the overall result partial; extraction and full native
capture acceptance remain open.
The [v18 lifecycle run](../2026-09-17/native-claude-lifecycle-v18/README.md)
repeated all 15 fact-lifecycle criteria successfully using installed `dedfcfd`,
the unchanged v17 runner, and the new per-server startup default with global
nonblocking startup enabled. Both tool inventories were ready. Capture remained
degraded, so the overall result is partial. A service response logged a
connection abort without an observed fact-criterion failure; its cause is
unproven. No semantic-quality claim follows from the lexical fixture.
The later [exact-query comparison](../2026-09-17/claude-v16-retrieval-diagnosis-v2/README.md)
verified the corrected target remained stored. All fixture embeddings were
identical, and ranking ties plus context budgeting excluded that target.
The broader diagnostic query does not count as a passed native lifecycle.
An implemented capability does not close its quality, native-host, or pilot
acceptance gate.

## Claims and current boundaries

| ID | Claim | Evidence and supported scope | Acceptance still open |
| --- | --- | --- | --- |
| C01 | Persistent local memory | SQLite/service regression checks cover durable events, jobs, restart, retry, and fact persistence. The [SSD checkpoint](ssd-package-checkpoint-v1/README.md) passes selected lifecycle contracts. [Mixed-host v4](../2026-09-18/native-crosshost-recovery-v4/README.md) adds bounded native replay and unchanged records across restart. | Behavior under recurring slow-storage stalls, continuous daily operation, and the two-week pilot. |
| C02 | Project, workspace, and global isolation | Scope and lifecycle regressions are included in that package. Registry identities and membership replace folder-name identity. | Complete native-host isolation matrix. Passing fixtures do not prove every external client supplies correct identity. |
| C03 | Surprisal-gated encoding | Mechanism tests cover distance bounds, scoped prediction history, explicit-write bypass, and the later symmetric embedding role. | Real-model retention/noise benefit over baseline and simple filtering. The historical 40 percent claim remains unverified. |
| C04 | Evidence-based mutable memory | Expected-version corrections, revisions, graph refresh, and canonical handling have regression coverage. Source `46cc101` additionally rejects a conflict proposal that would relabel an existing canonical assertion. | General semantic support of extracted proposals. The collision rule is conservative and does not establish entailment. |
| C05 | Affective routing | The [local held-out result](../2026-09-19/mechanism-local-v1/README.md) observes a 0.34 paired top-1 improvement with synthetic gold states, 95% interval [0.18, 0.50]. The real classifier parsed 300 outputs, but only 33 passed fidelity. | Reliable classifier fidelity and end-to-end retrieval benefit within the declared accuracy and latency budgets. Gold-state scoring benefit does not establish classifier-driven benefit. |
| C06 | Sleep consolidation and compression | Durable maintenance, evidence-linked processing, retention, and failure/retry machinery are implemented. [Development extraction evidence](preference-contract-development-v1/README.md) preserves omissions and unsupported proposals. | Held-out supported precision, critical retention, and information-preserving compression. A valid JSON response is not a supported fact. |
| C07 | Associative graph retrieval | The [local held-out result](../2026-09-19/mechanism-local-v1/README.md) passes structural correctness, including 144 cycle observations, but has zero eligible cases out of 50 against a minimum of 40. The [capacity diagnostic](graph-capacity-diagnostic-v2/README.md) also retains its negative development results. | Associative retrieval benefit remains unmeasured. Eligibility coverage failed; structural correctness does not establish useful recall. Defaults and evaluation gates remain unchanged. |
| C08 | Independently usable, composable modules | Standalone and integrated module regression checks exercise write filtering, scope, graph state, and lifecycle behavior. | Completed real-model baseline, full-pipeline, and five module-disabled comparisons. Mechanical correctness does not establish usefulness. |
| C09 | Inspectable, correctable, forgettable memory | Facts expose status, version, evidence, and ranking traces. Persistent-service checks and [native v9](../2026-09-18/native-codex-claude-continuity-v9/README.md) cover bounded correction, reviewed deletion of selected copies, and replay suppression. | Complete host/capture acceptance and arbitrary paraphrase erasure are not established. Logical forgetting does not erase historical backups or external host transcripts. |
| C10 | Truthful local setup and readiness | Configured runtime, explicit providers, generated interpreter-bound host commands, and input-specific readiness checks are implemented. | Full native installation acceptance. Embedding revision is an operator-managed identity label; the runtime does not automatically verify a mutable Ollama tag's digest. |
| C11 | Shared memory across Codex, Claude Code, and Hermes | [Native v9](../2026-09-18/native-codex-claude-continuity-v9/README.md) passes 15 bounded criteria for Codex teaching, Claude correction and reviewed forgetting, corrected and post-forget Codex recall, scoped controls, and replay suppression across restarts. [Mixed-host v4](../2026-09-18/native-crosshost-recovery-v4/README.md) adds bounded concurrent replay. [Host records](../../client-compatibility-matrix.md) retain the v6 failure and distinguish fixture providers from model quality. | Two v9 Stop timeouts and degraded Claude identity diagnostics remain. Complete capture, real-provider quality, native Hermes continuity, Desktop, and pilot acceptance remain open. |
| C12 | Proven retrieval, compression, or replacement advantage | The [September 19 local run](../2026-09-19/performance-local-v1/README.md) measured recall p95 4,063.309 ms against 500 ms and capture p95 135.186 ms against 100 ms; both fail. The [prior lab result](performance-corrected-v4/README.md) remains preserved. Source and hardware differences prevent attributing timing changes to code alone. | Three subsequent SSD baseline repetitions on `de6cdcf` pass recall/capture gates with real query embeddings and synthetic stored vectors/capture providers. That bounded result does not establish full-module latency, completed real-model quality acceptance, retained-information accuracy, or external pilot readiness. No general superiority or replacement claim is accepted. |

## Evidence bindings

The following indices bind raw files by byte count and SHA-256. Their READMEs
identify the source revision, configuration, and limitations. Historical results
remain attributed to the source that produced them.

- [Package verification index](access-metadata-package-verification/evidence-index.json): source `2494e31`, including schema 8. No later embedding-role or conflict-collision code is covered by this full checkpoint.
- [Historical assistant-suppression package index](../2026-09-18/assistant-suppression-package-checkpoint-v1/evidence-index.json): source `5af3d15`, 1,030 tests with four skips, six source/package checks, 360 selected installed cases with two matching skips, and seven synthetic replay assertions.
- [Combined checkpoint index](combined-package-checkpoint-v2/evidence-index.json): source `312b8d6`, passing source/package checks and the failed installed continuity check. The separate installed replay probe did not run.
- [Service/runtime package checkpoint index](../2026-09-17/service-runtime-package-checkpoint-v1/evidence-index.json): source `56182d9`, all six source/package checks, 87 installed contract cases, and seven synthetic Stop replay assertions. This is separate from native host and real-model evidence.
- [Host provenance package checkpoint index](../2026-09-17/host-provenance-package-checkpoint-v1/evidence-index.json): source `8d865a5`, 903 passing source tests with four skips, 110 passing installed contract cases with two matching skips, and seven synthetic Stop replay assertions.
- [Historical SSD checkpoint index](ssd-package-checkpoint-v1/evidence-index.json): source `6600dc2`, all six source/package checks and 42 installed contract cases. It remains separate historical evidence.
- [Native plugin index](native-plugin-checkpoint-v1/evidence-index.json): source `6600dc2`, failed budget-limited v9, v10 native plugin observations and remaining capture gates, plus the installed adapter timing preflight.
- [Installed diagnostic index](installed-continuity-diagnostics-v1/evidence-index.json): valid timeout reproduction plus the excluded first diagnostic whose fixture was incorrectly placed inside Git.
- [SQL timing index](installed-continuity-sql-profile-v3/evidence-index.json): two passing instrumented cases with separate SQLite execute/commit timings; not a repaired or isolated performance result.
- [Performance index](performance-corrected-v4/evidence-index.json): frozen d94 run with real query embeddings and synthetic load facts. It is not a semantic-quality test.
- [Metadata-write comparison index](access-metadata-comparison-v1/evidence-index.json): four fixed-order baseline/candidate arms. Baseline timing varied sharply, so the run does not establish a reliable speedup or close the full recall gate.
- [Graph diagnostic index](graph-capacity-diagnostic-v2/evidence-index.json): frozen d94 source and vectors, three capacities, development cases only, no new model calls.
- [Graph score sensitivity index](graph-weight-diagnostic-v2/evidence-index.json): twenty bound offline development observations. One eligible target enters top five at weight 0.4; four cases remain unreachable. This does not justify a default change or establish held-out utility.
- [Extraction development index](preference-contract-development-v1/evidence-index.json): development evidence, not held-out precision acceptance.
- [Native Codex index](native-codex-lifecycle-v4/evidence-index.json): installed-package preflight and the incomplete native attempt. It does not establish why the client cancelled the call.

The v4 model queue retains its frozen d94 code, model digests, dataset,
options, and output paths. See [evaluation status](evaluation-running.md) for
the power-interruption recovery. Completion and results must be checked before
promoting its running phases to accepted evidence. Later embedding-prefix
comparisons require separate outputs and bindings.

## Publication boundary

The [paper errata](../../paper-errata.md) is a local correction draft. No paper
publishing source has been located in this repository, and the hosted paper,
website, and package have not been changed by this work. Absolute competitor,
infinite-scale, production-readiness, and practical-superiority claims remain
unsupported here. Publication requires separate authorization.
