# Captured evidence collision repair

An explicit correction can now absorb a same-scope, same-profile canonical
destination that is unverified captured evidence at version 1 with no revision
history. The target keeps its ID and advances its version. The transaction
transfers source evidence before deleting the duplicate, removes its vector,
invalidates graph derivations, and records retired captured IDs. Graph refresh
removes retired nodes. Reviewed forgetting suppresses original source events,
historical text, current text and retired captured IDs.

The merge refuses a destination with an unfinished targeted job, preventing
a queued correction from losing its target. Asserted facts, conflict records
and revised destinations retain the collision rejection path. This patch does
not implement a general merge between two independently accepted facts.

The service regression exercises correction, restart, reviewed forgetting and
raw source replay, with spreading both disabled and enabled. An injected error
after duplicate deletion and target upsert proves rollback restores both rows
and their evidence; retry then completes. An identical-text foreign-project
control remains unchanged. A repeated captured row's retired ID is suppressed
after forgetting. A separate storage regression preserves a pending correction
that targets the potential duplicate.

The initial fixture lacked required capture fields and was repaired before
the correction failure was reproduced. The correction then failed before the
implementation, as expected. A later graph assertion incorrectly expected a
node with spreading disabled; it now checks target graph content only when
spreading is enabled, and checks retired-node absence in both configurations.

Validation: the expanded correction, maintenance, service, related-forget,
graph-persistence and reconsolidation selection passed 74 tests before the final
retired-ID and pending-job guards. The final collision/merge selection passed
13 tests. Black and git diff --check pass. Mypy passed the changed storage and engine files with
--python-version 3.13 before those final guards. Default Python 3.10 type checking
still encounters the installed NumPy stub syntax issue documented in the prior
follow-up. This is source-level evidence; full-suite, final type checking and
rebuilt installed-wheel Hermes lifecycle verification remain open.

The original failed installed-wheel v34 artifact remains preserved. No live
memory migration, host change, or publication occurred. Completed capture jobs
retain their historical result IDs; replay acknowledges their source receipt
without creating a new fact. This patch does not add general fact-ID redirects.
