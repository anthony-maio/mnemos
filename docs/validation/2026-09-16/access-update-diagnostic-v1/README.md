# Access-update storage-width diagnostic

Two disposable 10,000-row fixtures used the installed d94ded6 SQLiteStore,
WAL/FULL, and 100 committed updates of the same five IDs. The vector fixture
used deterministic 768-dimensional vectors; the control had no embeddings.
The vector arm ran first. The held-out evaluation queue was also active.

The vector arm measured p50 38.1 ms, nearest-rank p95 2,043.45 ms, and
maximum 7,810.56 ms. The no-vector control measured p50 51.5 ms, p95
170.84 ms, and maximum 1,513.33 ms. These observations support investigating
wide-row writes but do not isolate their causal contribution from fixture
order, filesystem activity, or checkpoint state. This is not a recall-gate rerun.

The WAL files were already large after insertion and did not grow during
timing; file size does not count overwritten WAL frames. A passive checkpoint
was queried after each timed sequence. No durability setting was reduced.

The helper records its hash and checks the imported installed storage against
the pinned commit before executing. It writes the report at completion, so
there is no separate pre-execution manifest. The supplemental observation
checks unchanged helper/source hashes and actual fixture removal afterward.
The original report and helper remain unchanged. No live database was used.
