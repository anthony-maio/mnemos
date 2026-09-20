# Saved graph seed and ranking audit

This audit examines repetition zero across the five cases, two embedding
contracts, and three relation arms of the terminal role-aware measurement.
The verifier checks all 30 rows against the original report: actual seeds,
source-relation membership, effective seed neighbors, relevance floor, direct
candidate membership, target scores/ranks, and the fifth returned score.

In the prefixed scope case, source associations raise the target total from
0.701867 to 0.724175, while the fifth returned score stays 0.763440. Its rank
improves from 31 to 20; returned IDs do not change. The target was already
reachable with semantic edges. In other cases, the selected seed lacks the
relevant source relation or the target remains outside the candidate pool.

These traces establish seed, candidate, and ranking bottlenecks. They do not
explain why the embedding model prefers a distractor or establish that a new
seed strategy would improve held-out retrieval. Available source provenance
is distinguished from active source edges in the semantic-only arm.

The original 18 lifecycle failures and zero top-five target observations are
unchanged. No model calls, thresholds, criteria, or production defaults were
changed. See the JSON for exact values and the Markdown audit for each case.
