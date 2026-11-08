# Lexical scope query development probe

Compare the current lexical SQL with a candidate that visits FTS matches first
and checks the same scope predicate on each matched chunk. Both apply scope
before LIMIT and retain bm25 ranking and ID tie order. No product change or
acceptance claim follows from this component probe alone.

Use two fresh copies of the existing frozen 10,000-fact seed. Open them read
only. Freeze source/seed hashes, queries, scopes and SQL before timing. Derive
12 queries from the first six word tokens of 12 evenly spaced facts in ID order,
plus an absent term and the common words `the project`. Compare unrestricted,
first available profile/project, and absent profile scopes, with limits 1 and
20. Use one warmup and five timed paired rounds, alternating arm order.

Retain full SQL results (IDs and bm25 scores), query plans, each duration and
exact pair comparisons. Verify both copies and original seed hashes after the
run. Failure or output drift rejects the candidate. Summaries report all arms
and scopes, including regressions. Do not tune or repeat this cohort. No model
calls, extraction, live databases, host settings or frozen scores are changed.

A useful result permits a scoped implementation with isolation, view, tie,
transaction and arbitrary-predicate regression coverage. Complete installed
latency and fresh quality evaluation remain separate requirements.

Preflight found an empty WAL sidecar and a seed predating the current scope
index. No timing observations ran. Retain both failed preflights. On each
copy only, create the exact scope index declared in current canonical source,
verify all chunk rows unchanged, close the preparation connection, and hash
the prepared file. Run the fixed comparison read-only and verify these hashes
afterward. The comparison measures ranking SQL only, without model parsing or
full chunk hydration; it cannot establish the complete method's latency.
