# Scoped ranking index

The prior narrow-projection probe saved only about 5 ms per search. Its query
plans still read chunk metadata to apply source-view and workspace predicates.
Test an expression index that contains all current scope predicate values and
the identifier, so ranking can avoid fetching full chunk payloads for every
eligible candidate. SQLite maintains this index with the authoritative rows;
there is no independently updated cache or new authorization rule.

Use a new copy of the pristine 10,000-fact seed. Keep the original read-only seed
and installed package unchanged. Add only the proposed index to the copy, then
open both databases read-only. Compare the original queries with ranked-row
queries using that index. Lexical search may materialize the bounded-width
eligible rowid/identifier relation before joining FTS, then load winning chunks.
Keep exact cosine/BM25 scores, scope predicates, limits and identifier tie order.

Reuse the prospectively selected ten vector controls and ten lexical strings
from the previous SQL probe. Run two warmup pairs and ten observed pairs per
path, alternating arm order. Retain every full result row and compare pairs
exactly. Freeze SQL, index DDL, input identities, driver and database hashes before
measurement. Verify original bytes and authoritative rows remain unchanged.
These are development storage timings, not complete hooks or fresh quality.

If the candidate warrants implementation, retain arbitrary-predicate and missing
extension fallbacks. Preserve support for older SQLite builds without materialized
CTEs. Test same-score ordering, source views, profiles, project/workspace/global
membership, legacy scope defaults, metadata types, corrections, deletes, access
metadata, existing-database index creation, rollback and concurrent clients.
Run the related storage and retrieval checks before building or making a new
complete installed performance claim. The 100 ms and 500 ms limits stay fixed.
