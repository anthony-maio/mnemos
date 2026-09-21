# Atomic review persistence

Thirteen focused SQLite lifecycle tests pass. The attached JUnit report covers
restart, mixed fact/review completion, lease expiry, stale target and source
rejection, duplicate indices, pruning retention, shared-source suppression,
target-only forget, transaction rollback after inserts, schema upgrade and
backup restore with newer suppressions.

The same execution session also passed 65 storage/backup/forget regressions,
31 service/suppression regressions and 28 durable-job/source-contract tests.
These are console observations, not separately archived JUnit reports. Black
passed for six changed Python files; Python 3.10 mypy passed for storage and the
two new test files after correcting optional-value assertions in tests.

The implementation adds schema 9 with the existing pre-migration backup path.
All databases used here are isolated test databases. No live migration ran.
The service does not yet select this extraction path or expose review resolution
and discard. This is storage evidence, not a real-model quality result, full
source-suite checkpoint, installed-wheel check or host acceptance result.
