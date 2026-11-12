# Consistent local evidence inspection

Canonical runtime `f1314f6d40477d59f6a31ecbc904c3e7a3ce5c7f` fixes a reproduced
SQLite inspection race. A second connection could forget a fact after inspection
read its content but before it read evidence. The response then combined the old
fact with empty evidence and `no_source_evidence`, despite that fact having
retained evidence at the original read. The regression failed before the fix.

Inspection now holds the connection's reentrant lock and a unique read savepoint
through the fact, scope check, evidence, original-source status, revision history
and serialization. WAL readers see a single database state while another
connection commits. Releasing or rolling back the read savepoint preserves a
caller's pre-existing transaction. No write lock, output format, provider,
threshold or native budget change is involved.

The new cases prove coherent reads during concurrent forget and correction,
nested caller-transaction preservation on success and failure, exception cleanup,
and rejection of existing foreign-profile and foreign-project facts before
evidence access. The concurrent writer commits; inspection returns the coherent
earlier state, and the next read sees the deletion or corrected version. It is
not a promise that a response stays current after it returns. Existing optimistic
write/version checks remain necessary.

Validation passed 44 focused tests plus 72 broader storage, service, durable-job,
correction, review and scope checks. Five snapshot tests were rerun successfully
after adding the foreign-scope cases. The focused set includes real stdio MCP
inspection. Black and `git diff --check` pass. Independent source review found no
blocking transaction/concurrency issue; its scope-test suggestion was applied.

Mypy remains red with the same two `object is not indexable` errors in untouched
forget-preview code. Shadowing both changed modules with the unchanged baseline
reproduced those two errors at baseline storage lines 4073-4074; the candidate
reports them at 4088-4089. No static-checking pass is claimed. Private JUnit
receipts and unchanged baseline sources are retained in
`.venv/inspection-snapshot-v1` in the improvement worktree.

This covers each individual SQLite inspection, not an atomic multi-fact batch.
It neither explains nor fixes the prior Claude 180-second timeout. That native
failure, the closed paging experiment, failed semantic configurations, complete
latency gates and original pilot requirements remain open or failed as recorded.
No model or native lifecycle was rerun, and no live database, configuration,
installed package or pilot changed. The new runtime fix is committed and pushed
to `wt/backup-20261108-delivery`; it has not been packaged or published.
