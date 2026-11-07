# Temporal answer-support comparison v4

Status: prospective development protocol. No v4 inference has run. Commit the
candidate, runner, datasets and this protocol before starting the comparison.
Earlier experiments, labels, scores and thresholds remain unchanged.

## Question and treatments

The capacity comparison improved abstention, but both 14B treatments selected a
historical queue when asked about the current queue. Test whether explicit
temporal instructions and comparison evidence reduce that error while preserving
historical answers, undated present-tense facts and qualified negative answers.

Compare candidate-at-a-time `joint_v1` with `qualified_v3`. Both receive the
same query and explicit reference date, November 6, 2026. The new treatment also
receives the same retrieval result's eligible accepted facts and retained source
excerpts as comparison context. This is a combined treatment, not an ablation
isolating the effect of prompt wording from comparison context.

`qualified_v3` may use supported comparison evidence to identify an explicit
replacement, time limit or unresolved conflict for the same subject,
relationship and environment. It must still select evidence belonging to the
candidate being assessed. A proposed, quoted or unsupported replacement cannot
override a supported fact. A historical fact can answer a historical question.
An ingestion or access timestamp does not establish when a fact was valid.

The reference date is caller-supplied and frozen; the checker does not read the
wall clock. Both treatments receive it, so a date-only improvement in the
baseline remains visible. Changing a comparison fact, source or eligibility
during judgment invalidates the whole new-treatment result. Failed or partial
assessments receive no abstention credit.

## Data and identical retrieval inputs

Use these two unmodified datasets and report them separately as well as pooled:

- `answer-support-development-v1.json`, SHA-256
  `2c388cb8a603d4a1463496e0a8243a0a22496026c2a0e1d42dca640ad05d9aef`:
  ten previously used scenarios and twenty regression questions.
- `answer-support-temporal-development-v1.json`, SHA-256
  `9adc7a464e4387802e2a76e2f63e4d8572ba398da7969f2cc84b15855bc8a807`:
  twelve new development scenarios and twenty-four questions designed after
  reviewing v3. These are not held-out data.

The combined set has 44 questions: 27 answerable and 17 without a supported
answer. Cases cover replacements, history, scheduled changes, proposals,
expired periods, undated facts, different environments and attributes,
unsupported source claims, unmarked conflicts, attribution and quoted dates.
Gold labels are manually specified before inference. Preserve them during the
run; annotate any later ambiguity without rewriting the original score.

Seed one original accepted-fact store per scenario with extraction disabled.
Use BGE small English v1.5 embeddings and the existing vector, lexical, hybrid
and full comparison arms. The graph arm opts in explicitly; ordinary application
defaults remain unchanged. Keep top-k five and the 1,200 UTF-8-byte context bound.

Freeze retrieval once per query and arm. Both treatments consume identical IDs,
facts, versions, reference dates and snapshot hashes against the unchanged
original store. Comparison context is limited to those returned IDs; it cannot
search the store for extra evidence. Keep foreign scope and conflicted-fact
eligibility controls. Grade candidate coverage separately from support choices.

This yields 176 frozen inputs and 352 observations in one repetition. There are
44 questions, not 352 independent samples. Preserve each input's dataset identity
for separate regression and new-development reporting.

## Runtime and limits

Use the private loopback Ollama 0.40.9 runtime and the existing Qwen3 14B Q4_K_M
artifact, digest
`bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`.
The runtime executable SHA-256 is
`0957bdaa7b52e163cdc7feb1275653f27c6b32e8ba309e957bdbb83493e97a4b`.
No model download, cloud fallback or live host configuration change is needed.

Set context to 8,192, parallel requests to one, maximum loaded models to one,
temperature zero, thinking off and maximum output to 1,024 tokens. Limit each
rendered prompt to 5,120 UTF-8 bytes, reserving 2,048 positions for template
overhead under the conservative byte bound. The offline worst-case fixture
preflight measured 4,445 bytes for the new treatment and 2,482 for the baseline.
It used all eligible fixture facts and made no model calls.

Keep the support threshold at 0.8 and the whole-operation deadline at 15 seconds.
The 64,000-byte typed-request budget now counts state repeated across candidate
calls, including comparison context. Require complete probability distributions,
unit sum within 0.01 and selection of a highest-probability option. Do not fill,
normalize or describe these model estimates as calibrated confidence.

Warm-load the model with an empty prompt, verify its digest and active context,
then run the frozen inputs in a deterministic shuffled order with seed 20261106.
Shuffle the two treatments for each input using its stable input identity.
Retain first-call timeouts; empty-prompt loading does not establish warmed
generation. Record device occupancy and model residency. Do not interrupt
another application's work to improve latency.

## Evidence and decision

Use new output directories. Bind source, protocol, dataset and runtime hashes
before inference. Preserve every attempted observation and raw request/reply,
including failures. Verify pairing, original store hashes, source hashes and
request-to-response accounting independently. Unload the model and stop only
the private runtime owned by the comparison.

Report answerable hits, clean hits, correct abstentions, unsupported facts,
failures, timeouts, request/token counts and latency by treatment, retrieval arm
and dataset. Review temporal failures against exact retained evidence. Do not
accept a blanket exclusion of dated or uncertain facts as an improvement.

The existing 90% answerable-recall and 90% abstention targets remain. Report the
historical and undated controls explicitly, along with all operational failures.
This study cannot establish fresh evaluation, extraction quality, context
efficiency, native-host reliability or pilot completion. Freeze the selected
candidate before constructing the separate fresh evaluation of at least 200
scenarios and three repetitions.

## Invocation

Run from the committed improvement worktree with its isolated environment and
the private runtime described above:

```powershell
.venv/Scripts/python.exe -m benchmarks.local_memory.support_temporal_comparison `
  --dataset benchmarks/local_memory/fixtures/answer-support-development-v1.json `
  --dataset benchmarks/local_memory/fixtures/answer-support-temporal-development-v1.json `
  --output .venv/answer-support-temporal-v4 `
  --endpoint http://127.0.0.1:11436 `
  --source-root . `
  --protocol docs/plans/2026-11-06-answer-support-temporal-v4.md `
  --embedding-cache .venv/support-runtime/embedding-cache
```
