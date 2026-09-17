# Recall latency investigation

Date: 2026-09-17

This note records a bounded, read-only diagnosis of the recall latency failure
in `docs/validation/2026-09-16/real-model-v3/performance-10k.json`. No model
calls, product edits, durability changes, or disk benchmarks were performed.

## Observed evidence

The run completed 33 warm retrieval queries over 10,000 synthetic rows. The
database was 324,862,864 bytes after loading and reported WAL mode with
`synchronous=FULL`.

The p95 recall was 1,314.583 ms, failing the 500 ms gate. The two largest
observations were:

- Query index 1: 1,359.907 ms total, with 1,083.845 ms in the actual
  `mark_retrieved_many` store call.
- Query index 4: 1,314.583 ms total, with 924.395 ms in the actual
  `mark_retrieved_many` store call.

The other actual store stages remained ordinary at those indexes. Semantic
retrieval took 62.593 and 56.739 ms. Lexical retrieval took 0.353 and
122.988 ms. The five primary-key `get` calls took about 1.2 ms in aggregate.
Both slow updates completed successfully and each received five chunk IDs.

The separate `mark_and_refresh_ms` follow-up probes were 42.859 and 42.213 ms
at those indexes. They are not the actual calls made inside engine retrieval,
so they do not explain the latency spike.

The preceding instrumented artifact,
`docs/validation/2026-09-16/performance-10k-instrumented-v1.json`, used the
same 10,000-row load size and also reported WAL plus FULL synchronous mode.
Its actual `mark_retrieved_many` maximum was 151.374 ms and p95 was 115.447
ms. There was no intervening latency optimization. The difference is evidence
of intermittent write or filesystem behavior, rather than a demonstrated
deterministic retrieval-plan regression.

## Source findings

`SQLiteStore.mark_retrieved_many` in `mnemos/utils/storage.py` performs one
`UPDATE memory_chunks ... WHERE id IN (...)` and then calls `self._conn.commit()`
when no transaction is already active. The IDs are primary keys. The method
does not invoke FTS rebuild, vector rebuild, graph refresh, or reconsolidation
processing.

The connection setup enables `PRAGMA journal_mode=WAL` and
`PRAGMA synchronous=FULL`. The source does not set `wal_autocheckpoint`, so
SQLite's default automatic checkpoint policy remains a credible contributor
to occasional long commit times. This is a hypothesis because the artifact
does not record WAL size or checkpoint results around each update.

The `memory_chunks` table stores the full content, serialized embedding,
metadata, and access fields in one row. The performance fixture gives every
row a 768-dimensional embedding and reports a 324 MB database. SQLite row
updates may rewrite pages containing the full row record even when only access
columns change. That can enlarge WAL writes and make a FULL synchronous commit
more sensitive to storage latency. This is also a hypothesis until page and
WAL deltas are measured.

The FTS update trigger is declared for `UPDATE OF content`, so the access
update does not fire it. The vector mirror has a delete trigger but no update
trigger used by this method. There is therefore no source evidence that the
slow calls rebuild FTS or vector indexes.

## Minimal offline probe

Use disposable SQLite files and keep the production durability settings
unchanged:

1. Record page size, `wal_autocheckpoint`, journal mode, synchronous mode,
   database bytes, and WAL bytes before and after each five-ID update.
2. Compare two otherwise identical 10,000-row stores: one with the 768-value
   serialized embeddings used by the fixture, and one with null embeddings.
3. Time the SQL update separately from the commit. Do not replace the commit,
   lower synchronous mode, or defer the write.
4. In the primary natural-WAL arm, do not issue checkpoint calls during the
   update sequence. Record WAL growth and checkpoint state only at the end so
   the probe does not perturb the automatic checkpoint behavior.
5. In a separate diagnostic arm, issue passive checkpoints outside the timed
   region and label that arm as an intervention rather than combining it with
   the natural-WAL measurements.
6. Repeat enough updates to observe both ordinary calls and an outlier, and
   preserve the raw timings and SQLite settings.

Interpretation:

- Similar update time with and without embeddings, but an outlier after WAL
  growth, supports an automatic checkpoint or storage flush explanation.
- Higher update and WAL growth with embeddings supports wide-row rewriting.
- Low update time followed by a high commit time localizes the cost to the
  durability boundary.
- No relationship to row width, WAL growth, or commit time leaves the cause
  unresolved. CPU scheduling, lock waits, instrumentation, filesystem stalls,
  and other environmental effects remain possible and should be documented
  rather than guessed away.

No immediate product optimization is justified. Changing FULL synchronous mode
or delaying access writes would alter durability semantics. If the probe
confirms wide-row rewriting, a separate migration study could move mutable
access metadata to a table keyed by chunk ID while retaining FULL commits.
That is a schema change and requires migration, backup, and consistency tests;
it should not be introduced during the held-out evaluation.
