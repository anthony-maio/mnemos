# Named probability-map support comparison

Keep this codec experimental. Across 136 observations per arm, the named format
retains 74 of 80 answerable hits, correctly abstains on
33 of 56 absent-answer observations, and selects
41 unsupported facts. There are 1
operational failures. Its support-only p95 is 3209.62 ms,
before retrieval is included. The measured development quality checks
do not all meet their numeric thresholds;
critical-fact retention and fresh acceptance are not established by this run.

The independent audit passed. That establishes the recorded evidence, not
acceptance of the candidate. Production remains product `f5a944ea`.

## Development quality

All 68 questions are known development inputs: the original 44 and the 24
introduced in the previous sparse-format experiment. There are 40 answerable
and 28 absent questions, each repeated twice per arm. Repetition does not add
independent semantic cases. The full cohorts and all 127 declared fact/evidence
pairs are unchanged.

| Cohort | Arm | Repetition | Answerable hits | Clean hits | Correct abstentions | Unsupported facts | Operational failures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original 44 | baseline | 1 | 27/27 | 27/27 | 12/17 | 5 | 0 |
| Original 44 | named | 1 | 25/27 | 23/27 | 10/17 | 13 | 0 |
| Original 44 | baseline | 2 | 27/27 | 27/27 | 12/17 | 5 | 0 |
| Original 44 | named | 2 | 26/27 | 22/27 | 10/17 | 15 | 0 |
| Subsequent 24 | baseline | 1 | 12/13 | 12/13 | 9/11 | 2 | 0 |
| Subsequent 24 | named | 1 | 11/13 | 11/13 | 7/11 | 5 | 1 |
| Subsequent 24 | baseline | 2 | 12/13 | 12/13 | 9/11 | 2 | 0 |
| Subsequent 24 | named | 2 | 12/13 | 11/13 | 6/11 | 8 | 0 |
| Combined | baseline | Both | 78/80 | 78/80 | 42/56 | 14 | 0 |
| Combined | named | Both | 74/80 | 67/80 | 33/56 | 41 | 1 |

Expected-ID precision among selected facts is 84.78% for baseline and
64.35% for named output. This metric uses authored expected IDs; it is
not a measured general factual-precision rate. Operational failures count as
neither answer hits nor successful abstentions. Baseline agreement does not
establish correctness, since the baseline retains its own errors.

## Output and timing

| Measure | Baseline median / p95 | Named median / p95 |
| --- | --- | --- |
| Support time (ms) | 2871.01 / 4065.08 | 2267.63 / 3209.62 |
| Generated tokens | 143.00 / 212.00 | 109.00 / 162.00 |
| Provider generation (ms) | 2506.95 / 3714.81 | 1911.10 / 2813.96 |
| Provider prompt processing (ms) | 174.83 / 246.42 | 175.69 / 246.18 |

The named arm is faster in 135 of 136 pairs, with a
paired median change of -549.26 ms. Selections or
assessment status differ in 25
pairs. Every timed observation, including failures, is retained. Percentiles
use nearest rank. Provider component percentiles must not be added together.

The timer covers normal support snapshot construction, prompt construction,
provider execution, decoding and support-result construction. Loading, warmups,
fixture seeding and grading are outside it. No retrieval, extraction or QA model
runs. These timings cannot establish complete recall or hook latency.

## Contract and evidence

All 68 semantic requests and baseline rendered prompts match the prior frozen
run. The STATE and QUESTIONS body is byte-identical between arms. Only the
output contract and its instructions change. The named format retains question
and option names, defines omitted options as zero, permits explicit zeros, and
requires complete probability mass within 0.01. It does not normalize values
or infer probabilities from a selected label. The original 0.8 support threshold,
source firewall, eligibility and revision checks remain intact.

The saved-reply analysis finds 135 valid named replies and
6 valid replies that assign the first option to every
candidate; 0 of those are answerable observations.
All per-question missing and unsupported IDs remain in the private evidence.
The analysis makes no new model call and changes no grade. The earlier sparse
format's 72 contract failures and severe answer loss remain unchanged.

Source `8cf31899bcc75408834eb13634dcc65c453cf52f` uses local Qwen3:14b Q4_K_M under Ollama 0.40.9,
temperature zero, thinking disabled and 8,192 context tokens. The 5,120-byte
prompt cap, 1,024 output-token cap and 15-second support deadline are unchanged.
Maximum baseline/named prompts are 4902/
4854 bytes; none is truncated.

All 272 scheduled observations complete, with 274 model
requests, 274 raw responses and 274 HTTP attempts,
including one unrelated warmup per arm. The audit verifies 181
bound files, 8 model/runtime assets and 68 disposable
databases. Declared accepted data, database bytes and bindings remain unchanged.
Model unload, owned-server exit, private-port closure and recorded process exit
verify. No live host configuration changes.

All 132 focused checks pass, including 32 new named-format checks. Strict mypy
and Black pass. The prior 3,674-test full suite retains its original source
binding and does not cover the newly added codec tests. Private archive SHA-256:
`dd139e4398558923d5eca3afcb185de4ecb7f201241bcb03f25b455977e4ce3a` (423 members).

The 95% precision and critical-fact retention, 90% recall/abstention, fresh
acceptance, complete latency, canonical delivery and actual two-week pilot
obligations remain. The pilot is inactive at zero days. No default or publication
state changes.

- [Frozen protocol](../../plans/2026-11-07-named-choice-support-v1.md)
- [Aggregate receipt](named-choice-support-v1-receipt.json)
- [Earlier failed sparse format](sparse-choice-support-v1.md)
