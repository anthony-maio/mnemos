# Legacy benchmark recency diagnosis

The 0.6.1 candidate's legacy retrieval path can select an old fact as current
because ingestion timestamps tie, then reinforce that choice by treating its
last access as a content update. An isolated trace reproduces both mechanisms.
The original failed release gate remains unchanged.

The trace uses candidate `9ea3852a`, the existing eight-document contradiction
fixture, its eight queries, simple local embeddings and a mock LLM. One
observational pass covers each store. Six controlled replays vary timestamps
and prior access while preserving content, embeddings, scope and query order.
All HTTP and non-loopback connections are blocked. No model service, native
host or live memory database is used.

In the observed in-memory pass, `deploy-old` and `deploy-new` share exactly the
same `updated_at`. The old fact has higher semantic similarity, so the recency
sort cannot distinguish them. The first query returns the old fact and changes
its timestamp. The next "current" query sees the old fact as newer and returns
it again. The SQLite pass gives those facts distinct ingestion timestamps and
returns the newer fact for both queries.

| Diagnostic condition | Memory expected answers | SQLite expected answers |
| --- | --- | --- |
| Observed ingestion clock | 6/8 | 8/8 |
| All ingestion timestamps tied | 2/8 | 2/8 |
| Explicitly ordered ingestion timestamps | 8/8 | 8/8 |
| Ordered timestamps, then access the old facts | 0/8 | 0/8 |

The controlled pairs have identical stored snapshots and candidate traces for
all 24 query pairs. The prior-access arm changes access metadata only; content
and embeddings remain identical. This shows why slowing ingestion or retrying
the benchmark until it passes would not resolve the underlying problem.

Ten relevant files match the public base byte for byte, including the engine,
benchmark, types, configuration and fixture. The in-memory store's syntax tree
also matches, and the simple embedder matches after excluding annotations.
The patch does not introduce new ranking logic. This is static attribution,
not a matched runtime test of the public base. The original two failing
benchmark repetitions did not retain timestamp traces, so their exact per-query
outcomes cannot be reconstructed from this later diagnostic.

The next retrieval repair needs a clear distinction between content revision
time and access time, including an honest fallback for legacy rows whose
content timestamp cannot be recovered. Tied content times cannot prove which
contradictory assertion is current. A benchmark repair also needs explicit,
versioned chronology and a prior-access case. Neither a sleep between inserts
nor a favorable timestamp substitution qualifies the existing release gate.
These are separate retrieval and evaluation changes; they have not been added
to the bounded storage patch.

The [receipt](legacy-recency-diagnostic-v1-receipt.json) records 64 diagnostic
query calls, the source comparisons and the private archive. Its read-only
audit verifies all controlled pairs, preserves the original release evidence,
and confirms fourteen watched live paths and both product candidates remain
unchanged. The 8/8 counterfactual is a diagnostic result, not a replacement
score or fresh acceptance. The 0.6.1 candidate remains unreleased and does not
pass its existing gate. The frozen `f5a944ea` product uses a different memory
path; this diagnostic makes no new quality claim about it.
