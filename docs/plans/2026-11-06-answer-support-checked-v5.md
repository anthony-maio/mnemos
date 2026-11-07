# Three-check answer-support comparison v5

Status: prospective development protocol. Commit the implementation, runner,
datasets and this protocol before inference. Use a fresh output directory and
one attempt. Earlier scores, failed runs, labels and thresholds remain frozen.

## Question and treatments

The v4 failure audit found fully assessed false selections at reported
probability 1.0. Errors included missing requested values, unsupported past or
future applicability, and unresolved conflicts. Test whether separate choices
for these decisions improve abstention while retaining supported answers.

Compare candidate-at-a-time `joint_v1` with `checked_v4`, implemented at
`e5373791971ec6ade560bcf149f48918d6b3042e`. Both receive the explicit reference
date November 6, 2026. The checked treatment receives the same retrieval
result's eligible accepted facts and retained sources, and asks three choices:

1. Does this candidate supply the requested value or qualified status?
2. Does one of its own sources establish that answer at the requested time?
3. Does supported comparison evidence leave that answer valid for the request?

Support requires `present`, an own `evidence:N` choice, and `clear`. Every
selected component must meet the unchanged 0.8 threshold. The returned score
is the minimum component score, not a calibrated joint probability. An absent
value, unsupported source or unresolved comparison cannot produce support.
Provider errors and changed dependencies receive no abstention credit.

Reference candidates by index without repeating their fact/source payload in
the questions. Keep all qualified statements and historical, negative,
uncertain, proposed and quoted controls. A comparison claim can override a
candidate only when its own retained evidence supports the relevant conflict.
Changing any comparison fact, source or eligibility invalidates the whole
result. All prior protocols remain available and production retrieval is
unchanged.

This is a combined treatment of question decomposition, wording and comparison
context against the existing joint baseline. It does not isolate the causal
effect of any one component or establish calibrated model probabilities.

## Frozen inputs and runtime

Retain the [v4 protocol's](2026-11-06-answer-support-temporal-v4.md) two datasets,
model, embedding, input freezing, grading, limits and cleanup requirements:

- Development dataset SHA-256:
  `2c388cb8a603d4a1463496e0a8243a0a22496026c2a0e1d42dca640ad05d9aef`.
- Temporal development dataset SHA-256:
  `9adc7a464e4387802e2a76e2f63e4d8572ba398da7969f2cc84b15855bc8a807`.
- Qwen3 14B Q4_K_M digest:
  `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`.
- Ollama 0.40.9 executable SHA-256:
  `0957bdaa7b52e163cdc7feb1275653f27c6b32e8ba309e957bdbb83493e97a4b`.

The 22 reused development scenarios contain 44 questions: 27 answerable and
17 without an answer. Freeze 176 retrieval inputs across vector, lexical,
hybrid and full arms, then run both treatments for 352 observations in one
repetition. These are 44 questions, not 352 independent samples. No extraction
runs. Use BGE small English v1.5, top-k five and a 1,200 UTF-8-byte context
bound. The two treatments share exact candidate IDs, versions, retained sources
and snapshot hashes. Comparison context cannot add facts beyond returned IDs.

Keep context 8,192, parallel requests one, maximum loaded models one,
temperature zero, thinking off and maximum output 1,024 tokens. Require full
probability distributions, unit sum within 0.01 and a highest-probability
selected option. Keep the 15-second operation deadline, 64,000-byte total typed
request limit and 5,120-byte rendered prompt limit. Actual checked_v4 request
construction measured 4,473 bytes at most per prompt and 10,383 per query over
all eligible fixture facts. That offline check made no provider calls.

Use a new private loopback runtime on port 11437. Keep the existing input
shuffle seed 20261106 and treatment-order seed prefix `temporal-v4:` so only
the candidate treatment changes. Warm-load with an empty prompt, preserve
first-call timeouts, record device occupancy, and do not interrupt another
application's work. No model download or cloud fallback is authorized by this
protocol. Unload the model and stop only the owned private runtime afterward.

## Evidence and decision

Preserve every attempted observation and raw request/reply. Independently
verify source and dataset hashes, identical pairing, original store hashes,
complete request/response accounting and cleanup. Report hits, clean hits,
abstentions, unsupported facts, failures, latency and token/request counts by
treatment, arm and dataset. Review the exact missing-value, temporal, conflict,
historical and qualified-statement controls without changing their labels.

The existing 90% answerable-recall and 90% abstention targets remain. This is
development evidence; passing it would still leave fresh evaluation, separate
extraction results, context-efficiency measurement and the actual pilot open.
Freeze a selected candidate before constructing the separate fresh evaluation
of at least 200 scenarios and three repetitions. No default promotion follows
automatically from this comparison.

## Invocation

Use the committed worktree and verified private runtime:

```powershell
.venv/Scripts/python.exe -m benchmarks.local_memory.support_temporal_comparison `
  --candidate-protocol checked_v4 `
  --dataset benchmarks/local_memory/fixtures/answer-support-development-v1.json `
  --dataset benchmarks/local_memory/fixtures/answer-support-temporal-development-v1.json `
  --output .venv/answer-support-checked-v5 `
  --endpoint http://127.0.0.1:11437 `
  --source-root . `
  --protocol docs/plans/2026-11-06-answer-support-checked-v5.md `
  --embedding-cache .venv/support-runtime/embedding-cache
```
