# Scoped SQLite ranking index

SQLite retrieval now ranks candidates through an index containing the current
profile, scope, workspace and source-view values, then loads complete winning
chunks. This avoids fetching every candidate's full metadata payload during
scope filtering. Exact cosine, BM25, identifier tie order and the existing scope
compiler remain unchanged. SQLite updates the expression index atomically with
the authoritative chunk, including writes from an already-open client.

Lexical retrieval scans MATCH candidates once and checks their rowids against the
indexed scope set before applying the limit. Unary plus on the FTS rowid prevents
SQLite from converting that membership check into a separate MATCH lookup for
every eligible row. The implementation uses ordinary CTE syntax; it does not
require materialized-CTE support. Arbitrary-predicate and missing-extension
fallbacks retain their previous behavior.

## Development measurements

Both comparisons use checkpointed copies of the same 10,000-fact seed. Each path
has two warmup pairs and ten observed pairs, with alternating arm order. Vector
controls use existing document vectors; lexical controls use ten previously fixed
query strings. These are storage measurements on known inputs, not fresh quality
evaluation or complete command timings.

| Selected candidate | Original median ms | Indexed median ms | Median paired change ms |
| --- | ---: | ---: | ---: |
| V2 vector ranking | 87.21 | 24.63 | -63.38 |
| V4 lexical ranking | 81.29 | 24.38 | -57.50 |

The indexed candidate was faster in all ten observed pairs for each selected
path. Every pair returned exactly the same 30 full rows, in the same order;
vector scores also matched. No complete-hook latency pass follows from this table.

All attempts remain available. V1 failed during preparation because its new index
was still in WAL when an immutable reader opened the copy. It retained one
baseline warmup observation. Closing the setup connection at process exit then
checkpointed that index, explaining the one recorded prepared-file hash change;
the original seed stayed unchanged. V2 explicitly checkpoints and closes setup
before freezing the databases.

V2's materialized lexical join was slower, at 3099.89 ms median, because its plan
repeatedly scanned the eligible relation. V3's direct membership condition was
worse, at 14567.31 ms: SQLite pushed the rowid condition into FTS MATCH lookups.
V4 retained the membership condition outside that constraint and produced the
lexical result above. The slow variants and their exact outputs were preserved.

## Regression and installed evidence

Twenty-four new checks verify complete-row equivalence, tied ordering, profile
and project/workspace/global scope, source views, legacy defaults, metadata types,
access counters, correction, deletion, reinsertion, index creation on an existing
database, rollback, concurrent-client visibility and fallback behavior.

All 190 related existing checks pass, including real worker termination at capture
and forget transaction boundaries, storage startup, graph writes, retrieval views,
provider outage, concurrent revision and host/service retrieval behavior. The
source total is 214 distinct checks. Black and strict mypy pass for the changed
storage module. These checks do not replace the older full-suite result.

Source `5735c502febbf495cda3e8c336a8b7069a298738` builds with isolated Hatchling
1.33.0 and passes Twine. Wheel SHA-256:
`ec9f6927afe9297c893ef017bdb6ddd95519f00ffa6d27520c07a95dda1bce26`.
All 94 installed package files and 83 Python sources match the built artifact
and checkout; 57 dependency versions match the previous installed baseline.

The installed storage methods created the index on a fresh seed copy and passed
20 full-row comparisons against the saved successful probe results. This check
made no model calls. It proves installed storage behavior, not a current native
host workflow or the complete retrieval service's latency.

An independent audit checked all 97 probe observations, including 80 timed samples,
the failed preparation state and the installed-method results. Authoritative
facts, sources, evidence, binary vectors and access records remained identical
to the seed. No live database or host configuration was changed.

The verified private archive contains 236 logical files in 137 content-addressed
members. SHA-256:
`fdd54f7196e0069e05377cebe8d6334f7ea85227c7e6e316221106d977ea4413`.
All twelve documentation checks pass.

The next check is a complete installed comparison with real query embeddings,
durable capture and full host output. The prior resident hook result still fails
the 100 ms requirement. Current native-host acceptance and actual pilot use remain
open; the pilot is inactive.

- [Initial protocol and regression requirements](../../plans/2026-11-07-scoped-index-v1.md)
- [Checkpoint preparation](../../plans/2026-11-07-scoped-index-v2.md)
- [First lexical membership probe](../../plans/2026-11-07-scoped-index-v3.md)
- [FTS constraint repair](../../plans/2026-11-07-scoped-index-v4.md)
- [Verification receipt](scoped-rank-v1-receipt.json)
- [Prior resident diagnosis](resident-hook-diagnostics-v1.md)
