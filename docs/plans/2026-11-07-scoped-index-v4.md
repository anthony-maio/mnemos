# Keep scoped membership outside FTS rowid lookup

V3 preserved the exact rows but became slower: its query plan uses FTS index
`0:=M1`, combining a rowid lookup with MATCH for each eligible row. Median lexical
time was 14,567.31 ms. Keep that result and its original inputs unchanged.

V4 applies unary plus to the FTS rowid in the membership condition. This preserves
the integer rowid value while preventing that condition from becoming an FTS
rowid constraint. The intended plan scans MATCH candidates once, then checks the
indexed scope membership set. BM25, tie order, selected columns and limits stay
unchanged. This syntax does not require materialized-CTE support.

Repeat the same two warmup and ten observed lexical pairs in a new checkpointed
copy, with frozen bindings and alternating arms. Require exact equality of every
full result row, preserve query plans and verify both databases afterward. A
successful storage probe permits source implementation and regression checks;
it does not establish complete-hook latency or native-host acceptance.
