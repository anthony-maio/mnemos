# Sparse choice support development comparison

Freeze this comparison before model execution. The leading support composition
still fails the 500 ms recall requirement. Its retained 132 contrast replies
have a median 143 output tokens and 2,573.35 ms of provider-reported generation,
versus 22.21 ms of prompt processing. Test whether a smaller output contract
reduces that cost without losing the source and applicability decisions.

## Inputs and treatments

Use the unchanged 44 known hybrid inputs from the live QA/contrast comparison
and 24 newly written development questions in
`benchmarks/local_memory/fixtures/sparse-choice-development-v1.json`. The new
questions cover 10 scenarios, with 13 answerable and 11 absent answers. Together
there are 68 questions, 40 answerable and 28 absent. These are development data,
not fresh acceptance data. The prior QA, contrast and retrieval results stay
unchanged.

Both arms use the entire same accepted candidate cohort, retained excerpts,
versions, scopes and reference date for each question. There is no extraction,
retrieval, QA inference or candidate prefiltering in this experiment. Build
disposable stores from the declared inputs. Keep labels in a separate grading
file that the model runner hashes but never parses or includes in a request.
Verify every old state/question pair
against its original recorded request before running either arm.

- Baseline: the original `contrast_v5` criteria and required complete named
  probability distributions through `OllamaJudge`.
- Candidate: the same criteria and full state through the development-only
  `SparseChoiceJudge`. Its ordered JSON array contains one probability map per
  question. Keys are explicit option indexes; omitted indexes encode exactly
  zero. All positive probabilities must be present and sum to one within the
  original 0.01 tolerance. Never normalize, infer mass from a label or round
  probabilities. Expand indexes back into the original complete typed answer.

The candidate uses compact JSON for the prompt state/questions and includes an
explicit option-index map. This tests the complete format change, not an isolated
effect of removing a particular output field. A highest-probability tie chooses
the first offered option; tied distributions cannot clear the retained 0.8
selection threshold. Invalid JSON, duplicate indexes, wrong row counts, foreign
options, non-positive entries and missing mass are operational failures.

## Fixed execution

Use local Qwen3:14b Q4_K_M, digest
`bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`,
Ollama 0.40.9, temperature zero, thinking disabled, 8,192 context tokens,
one parallel request and one loaded model. Keep the 5,120-byte rendered prompt
limit, 1,024 output-token limit, 0.8 support threshold, original provider retry
policy and 15-second support-operation deadline. Preflight every rendered
request and schema without model calls; do not truncate oversized inputs.

Run two repetitions with both arms for all 68 questions: 272 observations.
Shuffle question order using seed 202611071. Alternate which arm runs first
within each pair; reverse that first-arm assignment in repetition two. Freeze
the complete schedule, labels, inputs, requests, source, helpers, dependencies
and local model assets. Preserve all observations and HTTP attempts. No selective
rerun, model switch, threshold adjustment or omitted failure is allowed.

Load the model with an empty request, then use one unrelated warmup question per
arm before measurement. Save warmups separately. Use an owned private server,
verify its process identity and listening port, and stop only that verified
process tree. Unload the model and verify unchanged model assets, input files,
store projections, SQLite integrity and foreign keys after execution.

## Interpretation

Report answer hits, clean hits, empty abstentions, unsupported selected IDs,
operational failures and output-token counts for each suite, arm and repetition.
Report complete support latency, provider prompt/generation durations, paired
latency changes and selection differences. Percentiles use nearest rank. Repeat
observations do not increase the number of independent semantic cases.

Reconstruct every distribution, source selection and grade from the saved
requests and raw replies. Compare the two arms before considering any later
runtime composition. A lower component latency does not satisfy complete recall
or hook budgets, and agreement with the baseline does not repair its errors.
The original quality thresholds, fresh evaluation, actual daily-use pilot and
delivery obligations remain. Do not change production defaults or live hosts.
