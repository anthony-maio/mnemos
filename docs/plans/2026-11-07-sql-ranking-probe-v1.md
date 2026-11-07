# Narrow ranking projection probe

The resident diagnostic attributes about 123 ms to vector storage search and
111 ms to lexical storage search per prompt, while real query embedding takes
about 5 ms. The current SQL selects full chunk payloads while sorting candidates.
Test whether ranking a smaller projection before loading the winning chunks
reduces that cost without changing the selected rows.

Open the prior pristine 10,000-fact seed with SQLite read-only and immutable flags.
Use the installed candidate's columns and scope compiler, the same profile and
project/global memory view, and top_k=30 as used by ordinary five-fact recall.
Keep exact cosine scoring, BM25 scoring and identifier tie order unchanged.
The proposed CTE selects rowid, identifier and score, applies the original limit,
then joins the winning chunks and explicitly restores rank order.

Use the original query manifest at indices 0, 5, 10, ..., 45 for lexical queries.
For vector-query controls, use the existing document vectors at fact indices
`index * 317 % 9000` for index 0 through 9. These are storage controls, not query
embedding or retrieval-quality measurements. Perform two warmup pairs followed
by ten observed pairs for each path. Alternate arm order and keep all attempts.
Compare the complete returned rows exactly, including vector-query scores.

Freeze the seed hash, installed files, driver, protocol, SQL and input identities
before execution. Record both query plans, wall times and exact row comparisons.
Verify database bytes are unchanged afterward. Do not alter product code in the
probe, replace full-command results, claim a latency gate pass, or treat these
known synthetic inputs as fresh quality evidence. A successful probe supplies a
candidate implementation for broader semantic and scope regression tests.
