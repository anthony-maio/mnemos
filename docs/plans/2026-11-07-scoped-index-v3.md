# Scoped lexical membership probe

V2 preserved every returned row and reduced median vector search from 87.21 ms
to 24.63 ms. Its materialized lexical join regressed from 101.98 ms to 3099.89 ms;
the query plan scans the eligible relation for each FTS candidate. Preserve both
results and the earlier setup failure.

V3 changes only the lexical candidate filter: use rowid membership in the indexed
scope subquery, then fetch identifiers for tie ordering and load full winning
chunks after ranking. SQLite can build a lookup for membership instead of
repeatedly scanning the materialized relation. No materialized-CTE syntax is
required. Preserve BM25, scope conditions, top_k=30 and identifier tie order.

Use a new checkpointed copy with the same expression index. Repeat only the two
warmup and ten observed lexical pairs, using the same ten fixed inputs and
alternating order. Freeze SQL and all bindings before execution. Compare all
30 complete result rows exactly and verify database bytes afterward. Keep V2's
vector result unchanged. These remain storage development controls, not a
full-hook latency pass or fresh quality evaluation.
