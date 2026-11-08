# Controlled legacy chronology comparison

Test the installed content-recency repair on the existing eight contradiction
documents and eight known questions. This is a new development overlay, not a
replacement for the original fixture or failed production gate. Do not change
either package, original gold labels, original benchmark code or old results.
Do not use fresh acceptance questions for this comparison.

Freeze the original document bytes, this protocol, generated accepted records,
embedding state, expected query vectors, worker and controller before measurement.
Use the verified installed 0.6.1 and 0.6.2 environments. Common dependency
versions and the simple embedder implementation must match. No hosted provider,
extraction model, native host or live memory database is used.

Compare three routes: the 0.6.1 engine, the 0.6.2 engine and direct vector
retrieval on 0.6.2. Each route receives the same logical accepted snapshot,
project/global scope, query vector, eligible candidates, top-one output limit
and 2,048-byte context budget. The new content timestamp is supplied as a
prospective source field to all routes; 0.6.1 does not support storing it. Record
that intentional representation difference separately. Every common stored
field must still match before the corresponding request.

Use deterministic SimpleEmbeddingProvider vectors at dimension 384. Generate
document vectors once in original file order, followed by the global control.
Freeze that provider's document-frequency state. For each query, reset to the
same state, compute its vector using the actual provider and verify it against
the frozen expected vector. Do not substitute expected answers or fixed ranked
lists. Candidate instrumentation must observe the real store lookup without
changing its arguments, return values or ordering.

Add eight foreign-project copies of the original records, one disallowed
workspace copy, and one unrelated global control. Foreign copies retain their
content/vector and receive a newer timestamp. Only the eight project records
and global control are eligible. No query may return a foreign or workspace ID.
All routes use the same twenty-candidate request and the same eligible cohort.

Use four declared chronology conditions, each with and without prior access:

| Condition | Source chronology | Scoring |
| --- | --- | --- |
| Ordered creation | Original records have explicit one-second creation order; no later content rewrite is known. | Retain the original expected IDs. |
| Known rewrite | The new facts have older creation dates but explicit later content revisions; the old facts were created in between. | Retain the original expected IDs. |
| Tied chronology | All creation and known content times tie. | Report candidate returns and access invariance; do not credit temporal correctness. |
| Unknown legacy rewrite | New records have an older creation date and version two, but no recoverable content revision time. | Report candidate returns and access invariance; do not invent current-answer gold. |

The indeterminate conditions do not edit the original gold. They explicitly
withhold a new temporal-correctness claim because the visible source chronology
cannot support it. Record the raw returned IDs for every condition. Nonempty
retrieval is not successful abstention.

Prior access touches all old project records at the same fixed later activity
time, leaving content, creation, revision history, vectors, scope and IDs intact.
Freeze the engine's logical query clock. Start each query from a fresh copy of
its condition's snapshot so earlier benchmark queries cannot change the input
of later comparisons. This differs from the old sequential diagnostic and must
not be presented as reconstructing that run.

Run both memory and SQLite stores, three repetitions and all eight questions.
Rotate route order across repetitions and reverse prior-access order on the
middle repetition. This yields 1,152 observations: 384 matched route triples
and 576 matched access pairs. Repetitions are repeated known cases, not new
independent questions. Retain every result, exception, elapsed time, actual
candidate trace, common input snapshot and before/after record projection.

Report Recall@1/MRR for the two determinate conditions, exact access-invariance
counts for all conditions, forced-return counts for indeterminate conditions,
scope failures and context-budget failures. Preserve per-question results and
per-repetition ranges; do not treat the repetitions as an independent sample
for confidence claims. Instrumented request latency includes the actual query
embedding, retrieval and context serialization, with fixture creation excluded.
It is small-corpus diagnostic timing, not a 10,000-fact or native-command gate.

The repair's causal check requires zero access-induced selection changes in
0.6.2 for each matched condition, zero scope or budget failures and no lost
expected answers relative to 0.6.1's untouched determinate controls. Separately
report any wrong expected answers, even when access invariance passes. Do not
convert these checks into a replacement production gate, semantic acceptance,
or a release-ready claim.

Allow one measured run. Permit one correction and rerun only for a demonstrated
runner integration defect, preserving the first attempt. A semantic miss or
poor latency is a result, not grounds for tuning the fixture, prompts, model,
ranking or thresholds. Finish with a read-only audit, exact artifact bindings
and a report that preserves the original failed gate and remaining obligations.
