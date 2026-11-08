# Discriminative support feasibility v1

## Decision and scope

Evaluate one bounded alternative to the closed QA/contrast intersection. The
failure attribution identified lost historical answers and accepted unresolved
conflicts. This candidate uses one sequence classifier for question-specific
support and directed evidence-pair contradiction. It generates no answer text.

Select `tasksource/deberta-small-long-nli` revision
`9a77395d4d3751be9e2a69c4ae318491d9b3fffb` for this development screen.
Its [model card](https://huggingface.co/tasksource/deberta-small-long-nli)
advertises NLI and entailment-based label classification, a 1,680-token context,
and Apache-2.0 licensing. The candidate will use a stricter 512-token input cap.
The repository exposes a standard sequence-classification architecture and
safetensors weights. Verify the pinned configuration and label mapping before
loading. Use `trust_remote_code=False` and safetensors only.

The [MS MARCO MiniLM card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
describes passage ranking. A ranking score does not establish answer support,
so that model is not selected as the acceptance classifier here.

The hypothesis template below is an unvalidated zero-shot task formulation.
The card does not establish that this model understands temporal answerability,
that its scores are calibrated, or that it can meet mnemos's latency gate.
The screen tests those uncertainties; it does not assume them away.

## Fixed inputs and decision

Use the first repetition's 68 recorded baseline candidate sets from
`retrieval-support-development-v1`. Keep the original question, reference date,
candidate order, fact text, source excerpts, scope and assertion status. Do not
retrieve additional facts or expose expected IDs to the classifier. Keep labels
in a separate grading file. Explicit stored conflicts remain ineligible.

For each eligible retained excerpt, classify this pair:

```text
Premise: {exact retained excerpt}
Hypothesis: This passage provides a supported answer to the question "{exact query}" for the time asked about. If no time is specified, use {reference_date} as the reference date.
```

Retain all three raw logits and their three-class softmax. An excerpt witnesses
support only when entailment is at least 0.8. Do not renormalize by removing the
neutral class. Do not read the score as a calibrated correctness probability.

For every pair of eligible candidates, classify each directed pair of retained
excerpts as premise and hypothesis, preserving dates and all qualifiers. A pair
with contradiction at least 0.8 in either direction vetoes both participating
candidates. Evaluate these pairs even if one candidate failed the support check;
a contradictory source must not disappear merely because it was not selected.
Unrelated candidates remain independently eligible. For multiple excerpts, any
support witness can qualify a candidate; any contradictory cross-candidate
excerpt pair can veto it. Keep evidence IDs for both witnesses and vetoes.

This conservative veto may overreject dated replacements or unrelated facts.
Measure that failure instead of dropping temporal qualifiers, changing labels
or adding exceptions for the known cases. An unselected candidate is not a
deletion or a stored conflict update. No memory writes are permitted.

## Bounds and execution

Preparation may inspect the pinned card, configuration and tokenizer metadata.
Before inference, freeze the runner, dependency versions, exact weights and
tokenizer files, model revision, input and grading hashes, and the schedule.
No live project material enters model requests. Use a separate disposable model
runtime; do not alter the validated optional pilot installation.

Tokenize without truncation. Reject a question as operationally unavailable if
any required pair exceeds 512 tokens or if more than 128 pairs are needed.
Never silently discard a source or candidate to meet the limit. Use batches of
at most 32 pairs, model evaluation/inference mode, float32 and one fixed device
recorded before running. No device fallback during measurements. Check shape,
finite logits and exact label mapping for every output.

One unrelated warmup is recorded separately. Run exactly one measured pass over
all 68 known questions in lexical key order. Retain every attempted pair, raw
logit, token count, failure, selected ID and elapsed time. There are no semantic
retries, prompt variants, threshold sweeps or model substitutions in this run.
An infrastructure failure may be repaired once with the failed attempt retained;
a quality or timing failure closes the candidate.

## Grading and next gate

Report supported-selection precision, expected-fact retention, answerable recall,
correct abstention with available candidates, operational failures, and per-case
loss/veto reasons. Reuse original grades unchanged. Require 95% precision and
expected-fact retention, 90% recall and abstention, and zero operational or scope
failures before considering integration. Report critical-fact retention as
unmeasured: these known labels cannot establish it.

Time tokenization, batching, inference and selection together per question;
report cold loading and warmup separately. A support-only p95 above 500 ms
rejects feasibility. A value below 500 ms is not a complete-retrieval latency
pass. Do not add historical percentiles or claim host-token savings from the
classifier tokenizer. Report classifier tokens separately from returned context.

Only a passing screen can justify an installed complete-retrieval comparison
with identical accepted snapshots, scope, embeddings, candidate visibility and
context budgets. That comparison must revalidate source/version fences, include
10,000-fact timing, and pass original gates before consuming fresh acceptance
cases. No result here activates the pilot, changes defaults, publishes a release
or satisfies the actual two-week use requirement.
