# Real-model evaluation status

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

The latest verified [deletion coverage checkpoint](../2026-09-19/forget-coverage-checkpoint-v1/README.md)
at `1a11058` passes 1,439 source tests with 4 skips, all six
source/package checks, 608 isolated installed cases with 4 matching
skips, and seven replay assertions. Related-forget receipts now report
target-only or reviewed selection and retained exact, advisory, and eligible
source counts within the validated preview at commit. Direct forget has no
preview counts. Both explicitly leave semantic erasure unproven. Tool guidance
asks the model to report that coverage. Native v7 ran this package and
failed all three lifecycles before any deletion receipt was reached.
Earlier native and model results retain their original runtime bindings.

The previous verified [source-selection rejection package](../2026-09-19/related-forget-error-checkpoint-v1/README.md)
on `e454a85` passes 1,427 source tests, 604 installed cases, all six package
checks, and seven replay assertions, with four skips in each suite. These
are package checks; native observations and frozen model comparisons retain
their original runtime and model bindings.

## Development tool comparison

The [local model tool diagnostic](../2026-09-19/model-forget-development-v1/README.md) on installed
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

## Latest native-host observation

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

## Previous native v6 observation

The [fixed Hermes v6 cohort](../2026-09-19/hermes-native-cohort-v6/README.md)
on `db932be` completed zero of three lifecycles. All three attempts passed
teaching, restarted recall, correction, and restarted corrected recall.
Two deleted only the target and retained conversational copies. The third
selected additional facts without their required source IDs and received
degraded HTTP 500 results. The later `e454a85` package repairs that specific
error response, but has no native-model acceptance result. Three-host
continuity remains NOT RUN. Earlier cohorts remain separate evidence.

The [historical-assertion contract review](../../plans/2026-09-19-historical-assertion-contract-review.md)
explains why the missing-correction-target rows do not establish a runtime
bug. Valid `no_durable_facts` responses left no accepted target. The proposed
12-case development matrix separates current assertions, historical reports,
and corrections with or without an accepted fact. Its expectations have not
been independently human validated; the original negative results remain.

## Current checkpoint: September 19, 2026, local GPU replacement

An independent [supplemental v2 queue](../2026-09-19/supplemental-launch-v2/README.md)
started at 19:50:51 UTC. Encoding ended at 22:24:29 UTC with all three
reports marked `incomplete`. Each records 600 observations: full has 589
complete and 11 partial, no-surprisal has 592 complete and eight partial,
and length-noise has 585 complete and 15 partial. That is 34 partial rows
out of 1,800; complete-only metrics cannot establish overall acceptance.
Retrieval started at 22:24:30 UTC and finished at 23:38:37 UTC with an
incomplete manifest. It recorded 589 complete and 11 partial session
observations, 900 query rows, and 3,600 strategy results. The [terminal
retrieval archive](../2026-09-19/retrieval-terminal-local-v1/README.md)
preserves the reports, denominators, and conditional paired differences. All
three alternative strategies have zero paired quality delta against full on
the eligible rows; this does not establish complete-run quality acceptance.
The [terminal encoding archive](../2026-09-19/encoding-terminal-local-v1/README.md)
preserves all six report/raw-output files, denominators, and conditional
paired differences. It does not establish complete-run quality acceptance;
the original queue's unstarted follow-ons below
remain historical failures. The source, dataset, models, and protocols remain
frozen. Concurrent CPU-only development work qualifies timing comparisons.

The first [native Hermes lifecycle attempt](../2026-09-19/hermes-native-lifecycle-local-v1/README.md)
failed in teach because the asserted target omitted its final period and an
auxiliary title-generation request had null options. The [successor Hermes
discovery preflight](../2026-09-19/hermes-native-discovery-preflight-v3/README.md)
passes import and six-tool discovery without inference or service requests;
the successor lifecycle remains a separate pending run.

The newer [Hermes continuity package](../2026-09-19/hermes-continuity-checkpoint-v1/README.md)
at `78c63ed` passes 1,172 source tests (four skips), all six package checks,
484 isolated installed cases (two skips), and seven replay assertions. Its
[patched native Hermes run](../2026-09-19/native-hermes-local-v4/README.md) passes
16 single-turn scope/recall/raw-capture checks. These results do not rerun or
replace the earlier real-model matrix or establish full host acceptance.

The separate [combined recovery checkpoint](../2026-09-19/loopback-recovery-checkpoint-v1/README.md)
at `de6cdcf` has passed source/package and isolated installed checks: 1,160
source passes, four skips, 474 installed contract passes, two matching skips,
and seven replay assertions. Three fresh [SSD performance repetitions](../2026-09-19/performance-recovery-v1/README.md)
on that source pass the unchanged baseline gates: recall p95 245.214-252.222 ms,
capture p95 28.804-36.062 ms, with all 150 captures committed. These runs use
real query embeddings, synthetic stored vectors/capture provider, and no
cognition calls. They do not change the older model matrix's source or
reinterpret its failed completeness gate, and do not establish full-module
latency or native-host acceptance.

The original local main and supplemental queues ended as terminal failures; the earlier
live-process observations below are historical. The main lifecycle worker
returned reports for all seven profiles, but five were incomplete. Baseline
and `ablate-sleep_consolidation` completed 600 observations each. Across the
other five profiles, 50 of 3,000 observations had a missing correction target
after automatic extraction returned `no_durable_facts`. These are observed
memory-pipeline outcomes, not provider timeouts. The strict completeness check
failed the coordinator; the supplemental encoding and retrieval comparisons
never started. Original reports and raw outputs are preserved in the
[terminal lifecycle archive](../2026-09-19/lifecycle-incomplete-local-v2/README.md).
Completed execution is separate from quality acceptance. Resuming the original
checkpoints skips already-recorded partial rows; retrying only those rows until
extraction succeeds would bias the experiment. Diagnose the extraction and
failure-accounting contract on development cases before any fresh bound run.
The [subsequent read-only diagnosis](../2026-09-19/lifecycle-extraction-diagnosis.md)
finds explicit model rejection of historical-note wording, with valid,
non-truncated structured responses. Temporal/assertion ambiguity remains to
be resolved on development cases; the held-out labels and results stay intact.

The [retrieval parity checkpoint](../2026-09-19/retrieval-parity-checkpoint-v1/README.md)
at `8fc79f0` has completed source/package and isolated installed checks: 1,143
source passes and 440 installed contract passes. Its offline fixed-vector
baseline now has identical retrieved IDs between stores for all 46 paired
queries/repetitions. The legacy and controlled replacement gates both remain
failed. This source change does not alter the running `fc72cc2` model queue or
its supplemental comparisons. Main and supplemental processes were verified
live, and the full-profile checkpoint continued advancing during verification.

The separate [storage-location checkpoint](../2026-09-19/storage-location-checkpoint-v1/README.md)
has completed all source/package checks at `7bcabde`: 1,131 tests pass with four
skips. It binds the same installed wheel as `39b9c79`. The
[capture-only placement diagnostic](../2026-09-19/capture-placement-v1/README.md)
passes three SSD repetitions and fails three HDD repetitions under the original
100 ms p95 gate. These synthetic-provider runs do not replace the failed full
performance phase. A full recall remeasurement remains pending until the
serialized model evaluations finish. Main coordinator 19032, lifecycle worker
46876, and supplemental supervisor 61188 were verified live during this update;
the full-profile checkpoint was advancing. No incomplete quality results are
interpreted here. Their source remains frozen at `fc72cc2`.

The user reported that the lab machine no longer boots. The original `.35`
coordinator and worker chain were stopped after identity and ancestry checks;
execution handle `78705` returned terminal exit code 1. Thirty-seven result
and checkpoint files have verified backup copies. Original outputs and the
historical lock remain in place. The [retirement record](../2026-09-19/remote-evaluation-retirement-v1/README.md)
preserves the incomplete run without rewriting its quality or queue state.

A fresh local queue started at 05:46:55 UTC on clean source `fc72cc2`, with
coordinator PID 19032 and development-mechanism launcher PID 47992. It uses
`http://127.0.0.1:11434`, Ollama 0.34.1, and the exact original Qwen/Nomic
digests. The [launch preflight](../2026-09-19/local-evaluation-launch-v2/README.md)
verified source/package evidence and fully GPU-resident allocations at the
observed default contexts: Qwen 16,384 and Nomic 2,048 tokens. The new mutable
queue is `docs/validation/2026-09-19/real-model-local-v1/queue-status.json`;
work files are under `C:\Users\Ant\AppData\Local\Temp\mnemos-real-model-local-v1`.

Cold Qwen loading exceeded the unchanged 120-second inference timeout.
A separate loading-only probe completed in 149.623 seconds; the subsequent
warm preflight passed. The failed cold preflight and a helper's incorrect
Nomic endpoint request remain preserved. The local queue keeps the benchmark
settings and gates unchanged. It does not import remote observations or
establish model quality, latency success, host acceptance, or pilot readiness.
The applied encoding and retrieval-strategy comparisons remain separate,
pending serial runs. Colab remains tabled.

The local development-mechanism and lifecycle-preflight phases subsequently
finished, followed by the [performance phase](../2026-09-19/performance-local-v1/README.md)
at 05:51:09 UTC. Both latency gates failed: recall p95 4,063.309 ms and
durable capture p95 135.186 ms. The main queue advanced to held-out mechanisms.
A [supplemental supervisor](../2026-09-19/local-supplemental-launch-v1/README.md)
waits on the main coordinator and will verify completion/artifact hashes
before running encoding and retrieval comparisons serially. Their execution
and quality gates remain open. Performance findings do not change the frozen
quality run or its parameters.

The [local held-out mechanism phase](../2026-09-19/mechanism-local-v1/README.md)
finished at 06:14:34 UTC. All ten recorded artifact hashes match. Structural
graph checks passed, but none of the 50 graph cases met eligibility, against
the unchanged minimum of 40. Overall mechanism coverage therefore failed.
Affective scoring with synthetic gold states raised top-1 from 0% to 34%;
the real classifier passed fidelity on only 33 of 300 parsed outputs. This
does not establish end-to-end affective benefit or human acceptance. The
coordinator advanced to lifecycle worker PID 46876, verified live separately.
Encoding and retrieval comparisons remain queued behind main completion.

The separately frozen [embedding-client fix](../2026-09-19/embedding-client-checkpoint-v1/README.md)
at `39b9c79` passes source and isolated-wheel checks. Its disposable HTTP
fixture shows lower setup overhead, while both real latency gates remain
failed pending remeasurement. The legacy synthetic replacement gate fails on
both snapshots. The active model run and its queued comparisons still use
unchanged `fc72cc2`; these implementation results are not mixed into them.

## Historical checkpoint: September 18, 2026, 07:26 UTC observation

The existing execution session remained live. At 07:26:45 UTC, coordinator
`26488` and workers `47588` and `46036` were alive with their original
September 17 start times. The full-profile checkpoint had 42 completed
observations and 42 unique IDs, next session index 13, status `running`, and
mtime 07:11:53 UTC. Cases heldout-001 through heldout-014 each had three
recorded repetitions. The read-only `/api/tags` probe to `.35` exceeded five
seconds. These are liveness and checkpoint observations, not accepted
quality results. No frozen source, evaluation settings, models, or server
configuration were changed.

The separate [stale descriptor package checkpoint](../2026-09-18/stale-descriptor-package-checkpoint-v1/README.md)
at `37e446f` passed 1,049 source tests with four skips, all six source/package
checks, 379 installed cases with two matching skips, and seven replay
assertions. That client repair is outside the frozen real-model evaluation.

## Historical checkpoint: September 18, 2026, 06:02 UTC observation

The existing execution session and coordinator/worker processes `26488`,
`47588`, and `46036` remain alive. At 06:02 UTC, the full-profile checkpoint
mtime was 05:41:27 UTC, with 39 completed observations, 39 completed IDs,
next session index 12, and status `running`. A read-only `/api/ps` request
to `.35` again exceeded five seconds. These are execution observations,
not accepted quality results. The queue, model, and server were unchanged.

The package checkpoint at that observation was source `c25ed98`: 1,042 source tests with
four skips, all six source/package checks, 372 installed cases with two
matching skips, and seven replay assertions passed. See the
[outbox checkpoint](../2026-09-18/outbox-selection-package-checkpoint-v1/README.md).
This repair is outside the frozen real-model evaluation.

## Historical checkpoint: September 18, 2026, 05:32 UTC observation

The unchanged coordinator is still live: process 26488 and its lifecycle
worker chain were verified through the operating system, and the existing
execution session remains active. The observed chain was coordinator `26488`,
lifecycle worker `47588`, and worker child `46036`. The latest durable
full-profile checkpoint
was written at `05:11:18 UTC`; it records 38 completed observation records,
38 completed IDs, and next session index 12. These are execution counts, not
accepted quality results.

Recent lifecycle logs contain repeated Ollama failures. The latest recorded
categories are prediction `All connection attempts failed` and embedding
`WinError 10060` connection timeouts, most recently observed at `05:31:46 UTC`.
A read-only `/api/ps` request during this turn to
`192.168.86.35:11434` timed out after five seconds. No process was restarted,
and no model, server, queue, or configuration state was changed.

The package checkpoint at this observation was source `5d8cacf`: 1,039 tests passed with
four skips, all six source/package checks passed, and the installed wheel passed
369 selected cases with two matching skips and seven replay assertions. See the
[package checkpoint](../2026-09-18/windows-private-acl-package-checkpoint-v1/README.md).
The Windows permission, assistant-suppression, similarity-review, host-context, and UUID
write-safety work is outside this frozen real-model evaluation. Its validation
must not be treated as evaluation quality evidence.

## Historical checkpoint: September 17, 2026, 22:27 UTC

The resumed held-out mechanism worker completed with exit code zero at
19:48 UTC. The coordinator then started lifecycle worker PID 47588, observed
live at this checkpoint. The seven-profile lifecycle matrix is running. The
baseline completed all 600 observations across 200 sessions; the full profile
has three completed observations. Ollama on `.35` responds with both originally
pinned model digests.
The completed mechanism phase has not been used to tune the frozen settings,
and no quality conclusion is accepted before the complete comparison.

The [placement snapshot](../2026-09-17/ollama-placement-v1/README.md) reports
Ollama 0.34.1 and `size_vram: 0` for both loaded models, indicating CPU placement.
The user expects GPU execution and reports two RTX 3090 cards with 24 GB each.
Remote GPU availability and Ollama startup logs have not been inspected.
The Colab path remains tabled.
Historical placement is unknown; this snapshot does not establish that the
power interruption changed it. The active run remains unchanged while the
placement discrepancy is investigated. Any recovery must preserve the
existing evidence and record the execution-condition change explicitly.

The v4 queue resumed at 18:29 UTC after the local machine's power interruption
and restart at 18:09 UTC. The original process handle and worker PIDs were
absent. Recovery first timed out reaching Ollama without changing any files.
Once the endpoint returned, preflight verified the original model digests,
clean frozen source, runner hash, and every completed-phase artifact digest.
The stale lock and queue state were preserved before retiring that lock.
The unchanged coordinator resumed with `--resume`; PIDs 26488 and 35548 were
observed live. The earlier phase entry for PID 66348 is historical despite its
retained `running` label. The new phase entry identifies the resumed worker.
The [interruption snapshot](../2026-09-17/queue-interruption-v1/README.md) and
[recovery record](../2026-09-17/queue-recovery-v1/README.md) preserve both states.

The interrupted run uses frozen `d94ded6`, `qwen3.5:9b`
with thinking disabled and `nomic-embed-text:latest`. Development mechanism,
lifecycle preflight, and 10,000-fact performance phases completed. The
held-out mechanism phase finished; the seven-profile lifecycle matrix
is continuing. Recovery retained 62 completed case/pair IDs, 112 graph
observation rows, and 150 affective observation rows, including partial cases.
Those counts are execution progress, not quality outcomes.
No held-out quality conclusion is accepted yet.

The corrected performance measurement recorded recall p95 1,088.094 ms
against a 500 ms gate and capture p95 64.279 ms against 100 ms. See the
[preserved performance evidence](performance-corrected-v4/README.md).
The later [metadata-write comparison](access-metadata-comparison-v1/README.md)
does not establish a reliable speedup or close the recall gate.

The current coordinator is `.venv/run_thinking_evaluations_v1.py`;
status and outputs are under `real-model-v4/`, with persistent checkpoints
under `.venv/real-model-work-v4/`. The queue retains its original source,
dataset, model-digest, and option bindings. Later product changes, including
embedding roles, conflict collisions, concurrent retrieval, Claude startup,
lost capture acknowledgments, and service exit diagnostics, are not part of
this run. The complete
package checkpoint at that earlier observation was separate source `a607a5a`, with 924 tests passed and
four skipped, followed by formatting, typing, JavaScript, build, and Twine
checks. Its fresh installed wheel passed 130 selected contract cases with two
matching platform skips and
the synthetic Stop replay assertions on C: SSD fixtures. These checks do not
promote the frozen evaluation's quality results or resolve the earlier D:
storage timeouts. See the [package checkpoint](../2026-09-17/startup-exit-package-checkpoint-v1/README.md).

Current model pins:

- Cognition: `qwen3.5:9b`, digest
  `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`.
- Embeddings: `nomic-embed-text:latest`, digest
  `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.
- Temperature 0, output budget 512, provider timeout 120 seconds,
  three repetitions, `think=false`, no explicit embedding prefixes.

The v3 queue terminated with a lifecycle serialization failure. Its
[failure archive](lifecycle-serialization-failure/README.md) and earlier
results are preserved. It is not the active evaluation. The sections below
retain historical status and estimates, not current readiness claims.
See the [claim and evidence ledger](claim-evidence-ledger.md) for remaining
acceptance gates.

## Historical v3 checkpoint

At this earlier checkpoint, the fresh v3 evaluation queue was running from frozen
`b7a9d9d`. All four integrated lifecycle preflight cases passed. The 10,000-fact
performance phase completed: capture passed at 57.410 ms p95; recall failed at
1,314.583 ms p95 with two stalls inside durable retrieval metadata updates.
The held-out mechanism phase is now active; lifecycle follows it. The repaired
mechanism development run completed all observations and passed its graph
correctness controls; it established no graph recall benefit. See the
[development observations](mechanism-development-v2.md).
Complete repository verification passed: 624 tests passed,
2 skipped, followed by formatting, typing, JavaScript syntax, packaging, and
installed-wheel lifecycle checks. The [verification record](graph-v4-verification.md)
preserves the logs and artifact hashes.

The serialized queue runs lifecycle preflight, 10,000-fact performance,
held-out mechanism evaluation, then the seven-profile lifecycle matrix. Its
status and logs are in `real-model-v3/`; persistent fixtures and checkpoints
are in `.venv/real-model-work-v3/`. The coordinator is
`.venv/run_graph_v4_evaluations.py`, with its SHA-256 recorded in
`real-model-v3/queue-status.json`. It requires matching model digests, a clean
frozen checkout, passing broad/wheel checks, and completed development
correctness controls. Resume rejects a still-active prior coordinator or
child process. Negative quality and coverage results are retained separately
from execution and correctness failures.

The superseded v2 lifecycle matrix is stopped. It preserved 600 baseline
observations and 12 full-profile observations before a target-mapping defect
was confirmed. All 60 baseline correction attempts and 60 forget attempts
failed in the evaluator before reaching the corresponding service operation.
The replacement contract will name the prior source event and separate
replacement content from the operation instruction. Every profile must be
rerun under a new dataset/source binding; no existing result will be rewritten.

Baseline provider callbacks verified real embeddings in all 600 observations
and cognition in 150. Baseline deliberately disables cognition-dependent
modules, so the other 450 observations have embedding-only evidence. The
artifact has 480 complete and 120 partial observations and correctly reports
an incomplete run. Its 4 percent exact supported-fact proxy, 20 percent
answerable recall, and zero no-answer accuracy are not accepted quality
results: the correction/forget mapping defect contaminates lifecycle outcomes,
and exact frozen-text matching is narrower than semantic correctness.

Processes 47644 and 55876 were stopped intentionally after command-line
identity verification. Do not resume this v2 run or its old coordinator. The
files below are retained diagnostic evidence. The replacement lifecycle run
has not started. The original mechanism development run was also stopped after
a production graph correctness failure. A repaired development run is now
complete from detached checkpoint `b7a9d9d`.

The completed development run uses graph builder v4 and the complete
`2026-09-17-module-evaluation-protocol-v2.md` amendment. It retains five graph
cases, five affective pairs, and three repetitions, with unchanged authored
labels and model pins. Its artifacts are in `mechanism-development-v2/` and
persistent fixtures in `.venv/mechanism-development-work-v2/`. The binding is
`ac58f07c79c20e91bee95394c7e5f47348a4d51fe897861abe465211fff272b5`.
The frozen-vector digest is
`1370c4ad2cde46bfb9c333f75eaaf8a53bf978c0b916b5aedead530b755efc36`.
Full check logs and pytest XML are preserved in `graph-v4-verification/`.

The mechanism development run used detached source `e9d8ca2` in
`.venv/mechanism-frozen-v1`, five development graph cases and five development
affective pairs, with three repetitions. Its artifacts are in
`mechanism-development-v1/`; persistent fixtures are in
`.venv/mechanism-development-work-v1/`. Live provider metadata matches the
model pins below. The manifest binding is
`4738f773f7a56919c459ba1ea36c69d9893085ba16a7775f527305b8545109b9`.
Its first completed graph repetition changed both persisted edges and ranked
results when the fact insertion order was reversed. The energy oracle still
matched production. Processes 29796 and 31636 were stopped after command-line
identity verification; `mechanism-development-v1/stopped.json` records the
completed measurement hash. The databases and partial next repetition remain
intact. No held-out mechanism case was executed.

This development run cannot meet the held-out coverage minimums and cannot
establish quality acceptance. A preceding offline fixture completed three
production SQLite graph repetitions with synthetic embeddings and passed the
graph correctness controls. It made no model calls.

The lifecycle v3 amendment preserves every original v2 event, fact, and query.
It adds source-event target identifiers and separates correction content from
the action instruction. Since the v2 labels were already observed, v3 is a
repaired rerun, not new unseen held-out evidence. A new development fixture
now exercises actual correction and forget dispatch before the matrix restarts.

The first real v3 development preflight used source `634b460` and attempt
`8ce175be0c894d83928e8309ecb3724b`. Three of four runs passed: baseline
correction preserved the corrected fact ID and version after restart, and
both baseline and full profiles passed forget, replay suppression, and restart.
Full-profile correction was blocked before correction dispatch: its extraction
response repeated `{"operation":"add"}` objects without content or evidence
until the unchanged 512-token budget was exhausted. JSON parsing failed and
the durable extraction job remained pending. The evaluator correctly reported
no accepted target instead of guessing a fact to correct. Raw responses and
manifests are preserved in `lifecycle-target-development-v3/`. This is a
development result, not quality acceptance.

The extraction schema repair is committed in `b5a2834`. Every operation
alternative now carries all required fields. The old schema combined common
object constraints with partial `oneOf` branches. The llama.cpp schema reader
selects union alternatives before reading sibling object constraints, which
explains the observed incomplete objects; see its
[schema reader](https://github.com/ggml-org/llama.cpp/blob/master/common/json-schema.cpp)
and [grammar converter](https://github.com/ggml-org/llama.cpp/blob/master/common/json-schema-to-grammar.cpp).
These upstream sources corroborate the failure mechanism; the lab server's
embedded converter revision was not independently identified.

A fresh source-bound run from detached `b5a2834` passed all four real Ollama
development cases under attempt `b3fb69233d634934949a2432df377b59`. Both profiles
preserved corrected identity and version across restart, and both passed forget,
replay suppression, and restart. Model pins, token budget, prompts, and scenario
labels remained unchanged. Runtime evidence and operation validation remain in
place. Artifacts are preserved in `lifecycle-target-development-v4/`; this
development gate does not establish held-out quality or graph correctness.
The matrix remains stopped while the repaired graph checkpoint is validated.

The stopped run used frozen source commit `5e852e7` in
`.venv/eval-frozen-v2`. It started on September 16, 2026 local time, after
source-hash preflight confirmed 118 Python/configuration files. Its source
hash is `dd0631c56f5270679a84eee0d3f33c2c6f9c86f6cd7dcd4e4dcc98fde196d949`.

The lifecycle experiment ran against the lab Ollama endpoint, preserving
independent databases and checkpoints:

1. Lifecycle: 200 held-out synthetic sessions, seven profiles, three
   repetitions, 4,200 observations.
2. Competing-memory supplement v1: held before execution after a design audit.
   Its 25 candidates all fit inside the production direct candidate limit,
   and it lacks independently labeled state-appropriate targets. It cannot
   establish graph or affective benefit. A valid preregistered replacement
   remains required; no supplemental acceptance result is claimed.

The initial lifecycle estimate was about 42 hours. The withdrawn supplement's
35 to 70 hour estimate is no longer a scheduled commitment. The machine and Ollama
endpoint need to remain available. Completed checkpoints can be resumed with
the same dataset, source, model, and option hashes after interruption.

The coordinator status is `real-model-v2/queue-status.json`; per-phase logs
and completed reports are in `real-model-v2/`. Per-observation checkpoints
and isolated SQLite files are in `.venv/real-model-work-v2/` relative to the
implementation worktree. The coordinator was first stopped to prevent
its automatic handoff to the invalid supplement. The lifecycle worker
continued until the later target-mapping failure described above.
Do not relaunch the old queue script, which still includes the withdrawn
supplement. No task scheduler or system startup entry was installed.

Provider pins recorded in every profile manifest:

- Cognition: `gemma3:4b`, digest
  `a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a`.
- Embeddings: `nomic-embed-text:latest`, digest
  `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.
- Temperature 0, output budget 512, provider timeout 120 seconds,
  top-k 5, context budget 1,200 tokens.

Execution completion and quality acceptance are separate. A complete
observation can contain failed quality outcomes. No interim result justifies
promoting the biomimetic retrieval profile over the baseline default. The
supplement records whether graph targets were already direct semantic
candidates; such cases cannot establish graph utility. The datasets remain
synthetic and do not replace the two-week daily-use pilot.

The initial five-observation attempt under `real-model/` has invalid source
provenance and is excluded. See the [progress record](progress.md). Frozen
package evidence is in [frozen-build.json](frozen-build.json).
