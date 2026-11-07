# Bounded retrieval-plus-support development comparison

This is the single end-to-end quality increment agreed in the
[convergence plan](2026-11-07-candidate-convergence.md). The installed product
stays at `f5a944ea`. The adapter lives under `benchmarks/local_memory`; it does
not change a host default or qualify the baseline pilot configuration.

## Fixed inputs and arms

Reuse the 68 known questions from `answer-support-development-v1.json`,
`answer-support-temporal-development-v1.json` and
`sparse-choice-development-v1.json`. Preserve their text, labels and reference
date, including the prior explicit November 6, 2026 reference date for the
first fixture. Reconcile all query keys and labels against the retained
68-question comparison before execution. These are development cases, not fresh
acceptance data. Forty questions are answerable and 28 have no declared answer.

Seed every accepted fact declared in each original scenario, including scope
controls and stored conflicts. Do not seed only the candidates that previous
support experiments received. Preserve source excerpts and assertion status.
Use real local BGE-small embeddings. Disable extraction and all experimental
memory modules. Create a disposable SQLite snapshot for each scenario, then
give both arms separate identical copies for every paired observation. Record
the snapshot hashes, complete facts/evidence, engine settings and embeddings.
No arm may learn from a prior query's access counters or selected facts.

Both arms call the actual `LocalMemoryService.retrieve` with memory view,
baseline profile, project/global scope, top-k five and a 2,048-unit conservative
UTF-8 byte budget. The baseline returns that response directly. The treatment
passes only those returned IDs to the existing QA-v2/contrast-v5 intersection.
It receives no gold labels. Require matching candidate IDs and rendered
baseline context across each pair; record and investigate any mismatch.

Use the existing pinned QA model, QA worker, Qwen3 14B digest, Ollama binary,
contrast prompt, probability threshold, source-span policy and component
deadlines. Keep their settings and model assets bound by hash. Preserve raw
requests, provider replies, QA token arrays/spans, retry attempts and selector
decisions. Do not change models, prompts, thresholds, labels or output codecs
during this increment.

The adapter checks the retrieval response against current scoped facts and
retained evidence before selection. The intersection checks the entire cohort
around provider calls. The adapter then packs whole selected facts under the
same byte limit and checks the cohort once more before returning context.
Changed or unavailable evidence, degraded retrieval and provider/contract
failure return unavailable with no context. They never count as correct
abstention. Stored conflicts retain the existing unresolved policy.

## Schedule and measurements

Run two paired repetitions: 136 query pairs and 272 arm observations. Shuffle
the 68 question keys with seed `2026110704`; alternate arm order by question
index and repetition. Record model loading and one synthetic warmup separately.
Retain every attempted observation, failure and retry. The first measured
observation starts run one. A second run is allowed only after one demonstrated
integration defect has been documented and repaired. Keep the first result.
Semantic or latency failure does not justify that rerun.

Record each arm's complete elapsed time from retrieval dispatch through final
context, plus retrieval and support spans in the treatment. Report complete
p50/p95 directly; do not add component percentiles. Retain the candidate IDs,
selected IDs, decisions, context bytes, byte-counter identity and any facts
omitted by the final budget. Show per-question results and paired differences.
Use scenario-cluster bootstrap intervals with 10,000 resamples and seed
`2026110704`; the repetitions are dependent measurements, not new samples.

Report ranking misses separately from expected facts rejected by support,
unsupported selections, budget omissions and operational failures. Measure
declared-answer fact retention and selected-fact precision against the original
labels. Those labels do not contain a separate critical-fact annotation, so
this run cannot claim the goal's critical-fact retention acceptance. Report
that field as unmeasured rather than inventing labels after seeing results.

The development screen retains the original quality thresholds: 95% supported
selection precision and expected-fact retention, 90% Recall@5 and correct
abstention, zero scope/lifecycle failures and complete warm recall p95 <=500 ms.
Operational failures are failures. Also report no-answer cases with available
candidates separately from empty stores. The small scenario stores cannot
establish the 10,000-fact latency requirement even if this screen passes.

## Preparation and stopping rule

Before inference, commit the adapter and deterministic boundary checks. Bind
the runner, installed/source artifact relationship, model files, dependencies,
request settings, source snapshots and schedule in a private run manifest.
Keep input preparation, model warmups, measured observations and audit separate.
Do not alter frozen source while a run is active.

The preceding support-only result already exceeded the latency target. This
comparison establishes behavior after real ranking; it is not a predicted
latency fix. If quality or complete latency fails, record the failed candidate
and return the remaining scope/design decision. Do not open another codec
series or spend fresh cases on it. Fresh acceptance requires a passing and
plausible candidate, a new artifact/configuration freeze, the original sample
sizes and three repetitions, and explicit critical-fact labels fixed before
reading the new results. The six original goal obligations remain open.
