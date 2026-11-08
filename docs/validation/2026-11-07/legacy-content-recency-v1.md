# Legacy content recency repair

Separate repair `777c5445` fixes the reproduced access-promotion bug in the
legacy `MnemosEngine` path. Reading an old fact no longer makes it rank as newly
written content. The original 0.6.1 candidate `9ea3852a`, its failed benchmark,
the three-file storage backport and frozen functional-memory product `f5a944ea`
remain unchanged. The repair is not part of either prepared package.

`MemoryChunk.content_updated_at` records a known content rewrite.
`reconsolidate()` sets it while preserving the ID and incrementing the version.
Access leaves that field unchanged. The existing `updated_at` and access
counter behavior remain available for activity-based retention. Both legacy
recency scoring paths and the final recency sort use the content timestamp.

When revision history is unknown, ranking falls back to original creation time.
The new field remains null, including for previously rewritten legacy rows.
This prevents access from inventing a revision but cannot recover lost history
or decide which contradictory assertion is current when content times tie.
Creation time remains a ranking heuristic, not supporting evidence for an answer.

SQLite schema 3 adds the nullable field inside a write transaction. Concurrent
opening of an existing WAL database is covered; a failed migration rolls back
and closes its connection. Old clients deserialize rows positionally and are
not compatible with the new column. Any future deployment needs older clients
stopped and an original database backup. No live database was opened here.

| Verification on `777c5445` | Result |
| --- | --- |
| Complete legacy suite | 344 passed, no failures or skips |
| Content-recency regressions in that suite | 28 passed |
| Strict typing | 31 source files passed |
| Python formatting | 66 files passed |
| Retained database migration | Six rows and six vectors preserved in each of two copies |
| Additional migration controls | One graph edge and one feedback record preserved |
| Watched live paths | Fourteen unchanged |

The initial 22-case regression run had 21 failures and one pass. It reproduced
access-induced rank reversal in both stores, including top-one retrieval. Its
concurrency fixture also encountered the pre-existing first-open WAL setup
race. The fixture was corrected to use the existing release's WAL layout; the
separate baseline run then failed on the missing revision field. Both records
are retained. Concurrent first creation and WAL conversion are not claimed
fixed by this repair.

The migration checks use the six-row SQLite database retained in the original
release archive. One copy is unchanged before migration; another adds explicit
graph and feedback controls. Existing chunk values, vectors, source metadata,
search results and control rows match across migration and read-only restart.
Every legacy revision timestamp stays null. Integrity and foreign-key checks
pass. These are data-preservation checks, not a repeated quality benchmark.

Two evidence-script mistakes are preserved: the first file hash was taken with
an initial connection still open, and the second incorrectly required an empty
WAL file to disappear after a read-only connection. The completed check closes
all connections and requires zero remaining WAL frames before binding the files.
The read-only audit independently verifies the final bytes and row projections.
SQLite documents the companion-file behavior for [read-only WAL databases](https://www.sqlite.org/wal.html#read_only_databases).

The [protocol](../../plans/2026-11-07-legacy-content-recency-v1.md) preceded the
implementation. The [receipt](legacy-content-recency-v1-receipt.json) binds the
exact source, original failures, migration controls and reconstructable six-file
patch. Its 36-member private archive has SHA-256
`b75f41125fcf39476c6bb214b20a818e3ee63b4ed2f138b33388c1134a0063db`.
Verification used the recorded local Python 3.12 environment; the full supported
platform matrix and installed-package validation of this repair remain open.

The original benchmark code, fixtures and scores are unchanged. Its failed
release gate has not been rerun or replaced. A later chronology benchmark must
declare source ordering and prior-access cases prospectively. This repair does
not resolve the frozen functional-memory candidate's support-quality or timing
failures, qualify its native configuration, or start either pilot. Nothing was
published or activated.
