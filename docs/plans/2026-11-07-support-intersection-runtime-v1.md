# Runtime composition of the saved support components

Implement the fixed rule from the [saved-selection diagnostic](2026-11-07-support-intersection-diagnostic-v1.md):
select a complete fact only when corrected extractive QA and contrast-v5 both
select its ID. The previous 44-question result is known development evidence.
It does not establish a combined runtime or fresh quality acceptance.

Both components receive one immutable snapshot containing the literal query,
explicit reference date, complete eligible candidate cohort and retained source
excerpts. Its fingerprint binds those fields, versions, scope, caller context
and every supplied ID's eligibility. The second component receives the full
cohort even when QA selects nothing. Filtering its input by the QA decision
would be a different experiment.

Each response must identify this snapshot and cite the selected fact's own
retained evidence. Preserve complete qualified facts in baseline order. When
the components cite different sources for one fact, keep both witness sets;
agreement on the fact does not establish agreement on the same excerpt or
prove entailment. Do not combine model probabilities into a new confidence.

Recheck membership and the entire comparison before and after component work,
and around each internal provider await. Any correction, forgetting, evidence
change or eligibility change invalidates the whole result. Access counters and
retrieval traces are not claim changes. Provider failure, timeout, wrong-snapshot
replies and malformed selections are unavailable, not successful abstention.
Caller cancellation and configuration/storage errors remain visible. Both
components share one 30-second ceiling; the contrast component retains its
original 15-second operation ceiling.

The QA adapter keeps the v2 decoder, source-only offsets, boundary-whitespace
clipping, strict positive margin over CLS, 30-answer-token limit, 64-question-token
limit and 512-total-token limit. Its predictor must reject oversized input before
inference without truncation. Preserve complete input IDs, masks, offsets and
raw logits. The contrast adapter uses the existing `contrast_v5` prompt builder,
all-candidate batch, 0.8 probability threshold, complete distributions and the
same retained-source checks. No production retrieval mode or model dependency
is enabled by this implementation.

First run a model-free compatibility replay on the same 44 development inputs.
Reconstruct disposable SQLite fact/evidence views from the verified input
projection; do not mutate the original databases or call this a replay of their
complete lifecycle state. Feed the original QA repetition-one logits through
the actual QA adapter and the original hybrid contrast responses through the
actual typed judge and contrast adapter. Require exact saved query, date,
candidate text/order, source records and contrast question/state equality.
Recompute selections through the new composition, then compare them with the
saved intersection. Keep expected answer labels outside selection code.

Preserve all 44 results and any input mismatch. This preflight cannot establish
new inference stability, combined model latency, semantic quality or fresh
acceptance. A subsequent live composition must pin the same local model assets,
provider settings and unlabelled inputs before model startup, run three actual
repetitions, log both components and report every failure. Keep the existing
90% recall/abstention and 500 ms recall requirements. No pilot day, live
configuration change or default promotion follows from the preflight.
