# Post-hoc support intersection diagnostic

Compare the saved selections of the corrected extractive QA decoder and the
Qwen3:14b contrast checker on their 44 shared development questions. Both
component results are already known. Choosing these components is informed by
their earlier results: the contrast checker retained clean answerable selections,
while QA abstained more often. This is a post-hoc diagnostic, not a new model
experiment, fresh evaluation or validation of a deployed composition.

Use only the hybrid baseline and `qwen3-14b-contrast` treatment from the v6
comparison. Use repetition one from the corrected QA v2 re-decode, and verify
that all three recorded QA repetitions selected the same IDs. Qwen has one
recorded observation per question. Do not count reused Qwen output as three
independent runs. The seven other QA questions have no counterpart in this Qwen
comparison and are outside this explicitly matched analysis.

Before joining, verify saved artifact hashes and exact query text, reference
date, snapshot identity, baseline order, fact content, version, scope,
assertion status and retained evidence. Match by case/query identity, never by
which answer succeeded. A mismatch stops analysis. Preserve all 44 matched
questions, including every failure, in the denominator.

The only decision rule is intersection: keep a complete candidate fact only if
both component selections contain its ID. Preserve baseline order and full
qualified content. No answer labels, model scores, dates or case-specific rules
enter this operation. If either component failed, the combined result is
unavailable; an empty failed result cannot earn abstention credit.

Recompute any-answer hits, clean hits, fractional recall, empty abstentions and
unsupported selections for QA, contrast and their intersection. Report every
remaining failed question. Retain original component outputs and scores. A
selected-ID overlap calculation cannot establish source truth, generalization,
runtime revalidation, concurrent correction/forget safety or operational
reliability of a combined implementation.

Historical component timings may be listed separately. Their sum is not a new
end-to-end latency measurement and does not establish the 500 ms recall gate.
Sending only QA-selected candidates to Qwen would change its input and require
a separate experiment; do not infer that altered checker's decisions here.

This analysis makes zero model or retrieval calls, leaves the ongoing frozen
fresh retrieval comparison unchanged, and does not alter production defaults,
the pilot configuration or acceptance thresholds.
