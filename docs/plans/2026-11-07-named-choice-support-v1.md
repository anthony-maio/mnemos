# Named probability-map support comparison

Freeze this protocol and the drivers before inference. The prior indexed sparse
format failed both semantics and its zero-omission contract. This follow-up
retains named question and option keys. It changes only output serialization and
the instructions needed to describe that output. The serialized STATE and
QUESTIONS body must match the complete-distribution baseline byte-for-byte.

Use all 68 existing development questions from the
[previous comparison](2026-11-07-sparse-choice-support-v1.md): the original 44
and the subsequent 24. All are now known development data, not fresh acceptance
data. Keep the full accepted candidate cohorts, source excerpts, versions,
scope and reference date. Verify all 68 semantic requests against the prior
frozen requests. Keep gold labels separate from model inputs; the runtime may
hash the labels but must not parse them.

The baseline remains `contrast_v5` through `OllamaJudge`, with required complete
named probability distributions and an explicit chosen option. The candidate
is the development-only `NamedChoiceJudge`: each named question maps directly
to a map of named option probabilities. An omitted option means exactly zero;
explicit zeros are also valid under this new contract. All positive mass must
be present and sum to one within 0.01. Never renormalize, round, infer confidence
from a selected label or repair a malformed response. Decode into the same full
typed distribution. Equal maxima select the first offered option and cannot
pass the unchanged 0.8 support threshold. Duplicate or foreign keys, missing
questions, invalid values and missing mass are operational failures.

Retain the original baseline prompt, candidate state/criteria serialization,
5,120-byte prompt cap, 1,024 output-token cap and 15-second support deadline.
Preflight every complete prompt and schema without a model call; do not truncate
inputs. Preserve the normal source firewall, eligibility rules, snapshot/revision
checks and provider retry policy. No retrieval, extraction or QA model runs.

Use the same local Qwen3:14b Q4_K_M model, digest
`bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`,
Ollama 0.40.9, temperature zero, thinking disabled, 8,192 context tokens, one
parallel request and one loaded model. Keep the prior 68-question order from
seed 202611071. Run both arms twice, alternating the first arm within each pair
and reversing that assignment in repetition two: 272 observations. Load the
model with an empty request, then retain one unrelated warmup per arm.

Use new disposable stores and an owned private server. Retain every attempt,
raw response, timing and failure. Freeze source, helpers, dependencies, model
assets, inputs, labels, rendered requests and schedule. Verify database
projections and integrity, unchanged bindings, model unload, server identity
and private-port closure. Preserve the original failed sparse-format run.

The independent audit must reconstruct full distributions, source selections,
versions and grades from saved raw replies. Report answer hits, clean hits,
correct empty abstentions, unsupported selected IDs and operational failures by
cohort, arm and repetition. Report generated tokens, complete support latency,
provider component times, paired differences and selection disagreement.
Percentiles use nearest rank. Repetitions do not add independent semantic cases.

Baseline agreement is not acceptance: the baseline already fails the quality
gate. A smaller output must preserve supported answers and improve abstention,
then pass fresh acceptance and the full latency budget before adoption. Keep
the 95% precision and critical-fact retention, 90% recall/abstention, and 500 ms complete-recall
requirements. Support-only timing cannot prove complete recall or the 100 ms
hook requirement. Production defaults, live hosts and publication stay unchanged.
The actual two-week pilot remains inactive at zero days.
