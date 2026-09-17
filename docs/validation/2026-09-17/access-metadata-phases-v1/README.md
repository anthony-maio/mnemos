# Access metadata phase timing

This archive contains two completed, source-bound runs of the 100-call access
metadata probe. Each run populated 10,000 deterministic 768-dimensional facts,
closed and reopened the SQLite store, then measured 100 five-ID
`mark_retrieved_many` calls with the existing WAL, `synchronous=FULL`, and
autocheckpoint settings. The transparent connection proxy timed the access
INSERT OR IGNORE, access UPDATE, commit, and full call separately. No
checkpoint was issued during the timed sequence. Both runs preserved 10,000
chunk rows, 10,000 access rows, and access counts of 100 for all five IDs after
the final close and reopen.

The runs were executed in observed order D, then C. D had total-call median
104.0086 ms, p95 313.7343 ms, and maximum 674.0321 ms. Its commit stage had
median 103.8346 ms, p95 313.5566 ms, and maximum 673.843 ms. The access INSERT
and UPDATE p95 values were 0.1576 ms and 0.0439 ms. C had total-call median
1.96225 ms, p95 2.58 ms, and maximum 3.0604 ms. Its commit stage had median
1.875 ms, p95 2.4211 ms, and maximum 2.9624 ms. The derived JSON uses a
nearest-rank p95 over the 100 calls.

The D commit result localizes most of that run's measured call time to the
durable commit boundary, while the C result shows a much faster later run.
The order was not randomized, the runs were not isolated from other machine
work, and this is not an end-to-end retrieval measurement or a performance
gate. The C observation is a viable default-path diagnostic, not a claimed
speedup. It does not distinguish filesystem scheduling, WAL state, locking,
or other environmental causes by itself.

The source is the clean `a607a5a6b0a8c2d4f0c4ed00e3b21167cf96f854` snapshot.
The executed phase runner and original fixture helper are stored as `.py.txt`
with their exact hashes. Reports and manifests are copied byte-for-byte, and
`evidence-index.json` records every archived file except itself.
