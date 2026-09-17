# Completed baseline, incomplete comparison matrix

The frozen sourceb7 baseline completed 200 sessions with three repetitions:
600 observations. The full and five leave-one-module-out profiles are still
running. This checkpoint cannot establish a module benefit or full-pipeline
quality.

| Measurement | Observed | 95% Wilson interval |
| --- | --- | --- |
| Exact answerable Recall@5 | 120/300, 40% | 34.6% to 45.6% |
| Exact critical-fact retention | 120/300, 40% | 34.6% to 45.6% |
| Empty retrieval on no-answer queries | 0/300, 0% | 0% to 1.3% |
| Exact supported-text and evidence proxy | 120/1,440, 8.3% | 7.0% to 9.9% |

These estimates fail the planned quality gates. The baseline has no enabled
biomimetic modules and does not call the extraction model; its embedding
verification passed, while the report's real-model gate is `NOT VERIFIED`.
Sixty correction operations and sixty forget operations committed. The
retrieval scope validator recorded 2,160 valid rows and zero foreign,
malformed, or missing-identity rows.

The precision label needs care. The frozen scorer requires an exact match
to a supported gold assertion, matching scope, and valid user evidence.
All 1,320 rejected outputs lacked that exact text-and-scope match. The score
does not establish that those outputs are semantically false. The artifact's
note says paraphrases are unjudged, but its implementation actually counts
unmatched output as false in this exact-match proxy. Preserve that discrepancy;
do not present 8.3% as independently adjudicated factual precision or change
the frozen scorer to improve the result.

Likewise, the no-answer metric requires an empty fact list. It tests memory
retrieval abstention, not a downstream assistant's final answer. The measured
session latency includes setup, lifecycle operations, and retrieval and cannot
replace the separate 10,000-fact warm-recall measurement.

The derived report records the baseline report digest at observation, raw-output
digest, frozen dataset identity, and relevant source-blob hashes. The original
baseline report can gain paired comparisons when the matrix finishes, so it
remains in the active run directory until that phase closes. No thresholds,
dataset labels, or active evaluation code were changed for this checkpoint.
