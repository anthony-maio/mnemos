# Combined legacy repair candidate

Candidate `b3508f6d344afd9521ef2db981f3480ceeb8ae46` combines the prepared transaction, Windows plugin launch,
array-typing and content-recency repairs on the recorded PR #7 base. It passes
353 source tests on Windows Python 3.11.13, including all 28 recency regressions.
Strict mypy passes 31 source files and Black passes 67 Python files in the
existing Python 3.12.11 validation environment.

This closes a source-integration gap between separately tested repairs. The
transaction savepoint method is syntax-tree identical to the prior combined
storage candidate. The recency engine, types, tests and documentation match
the original recency commit. The automatic storage merge preserves PR #7's
content-only FTS update behavior; its FTS regression tests pass alongside the
caller-owned graph transaction tests and migration tests. No conflicts required
manual resolution, and the worktree is clean.

## Upgrade boundary

This candidate uses legacy schema 3. Opening an old database adds a nullable
content-revision timestamp; access timestamps are not treated as known content
history. The repair prevents reading stale content from promoting its recency.
It cannot recover missing revision history or resolve tied contradictory facts.
Older clients deserialize rows positionally and cannot safely share the upgraded
database. Any later deployment must stop those clients and retain the original
database backup. No live database has been opened or migrated here.

All earlier candidates remain intact, including the storage-only candidate and
its installed artifact. The new combined source still reports version 0.6.0;
that is inherited development metadata, not a release identity. A release must
prepare consistent version metadata and document the schema boundary before
publication. No release artifact was built in this increment.

## Evidence limits

The [receipt](legacy-integrated-v1-receipt.json) binds the candidate, complete
source-suite result, original repair, patch and private evidence archive.
The archive retains raw logs, JUnit records, source origins, dependency versions
and helpers. Test and typing processes exited zero. Existing installed legacy
package files are unchanged.

The original failed 0.6.1 benchmark and the separate controlled 0.6.2 chronology
comparison remain unchanged. Neither result is transferred to this commit, and
no benchmark was rerun to replace a failure. The new source suite is integration
evidence, not fresh retrieval quality, a complete-command latency pass, installed
package acceptance, a native-host check or daily-use evidence. Canonical 0.7
delivery, host configuration and both inactive pilot plans are unchanged.
