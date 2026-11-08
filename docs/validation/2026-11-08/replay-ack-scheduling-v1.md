# Replay acknowledgement scheduling

Canonical `ec4a36b2` moves replay acknowledgement I/O off the owner event loop.
It is pushed on `wt/backup-20261108-delivery`. No installed package, live
configuration, pilot setting, native timing evidence or PR #9 artifact changed.

Review identified a concrete scheduling coupling: replay_outbox called the
synchronous acknowledge_many method on the same loop used by HookDispatcher.
For SQLite this includes a FULL deletion commit. A deterministic regression held
an actual SQLite acknowledgement COMMIT open and requested another loop callback.
Before the repair, the callback could not execute until the observer released
the commit. That failed result is retained in
`.venv/replay-ack-scheduling-red-v1.xml`.

Replay now runs each existing acknowledgement batch in an executor. The outbox's
own locks, connection rules and FULL commit remain unchanged. A shielded future
is joined before cancellation propagates, including repeated shutdown requests.
The worker therefore cannot close its client/outbox while that acknowledgement
I/O remains active. Ordinary commit failures still propagate; if shutdown also
cancels the operation, cancellation propagates after the I/O outcome is observed.
The durable queue remains authoritative for recovery.

The regression now demonstrates that the next loop callback can run while the
acknowledgement commit is held. Additional tests cover repeated cancellation,
both successful and failed commits during shutdown, and ordinary failed threaded
commits retaining the original event ID after reopen. Review found no blocking
issue in the final diff; the reviewer inspected code and did not run tests.

## Validation

- Queue/replay suite: 28 passed, one skipped in 16.06 seconds. The skipped test
  attempts to rename an open SQLite database, which this Windows setup prevents.
- Additional scheduling/process-exit/queued-worker lifecycle suite: 17 passed in
  37.43 seconds, including exit before and after acknowledgement commit. Some
  scheduling cases overlap the first suite; these totals are not additive.
- Final four scheduling/cancellation/failure cases passed after the completed-
  future cancellation guard was added, in 2.63 seconds.
- Mypy accepts the changed client module; Black and diff checks pass.

Private JUnit records are `.venv/replay-ack-scheduling-green-v1.xml`,
`.venv/replay-ack-lifecycle-v2.xml` and
`.venv/replay-ack-scheduling-final-v3.xml`.

## Performance boundary

This proves event-loop responsiveness under the tested contention, not faster
complete capture. Capture append still uses the shared outbox lock and waits for
its own FULL commit. Authoritative Git resolution, fresh registry authorization,
privacy checks, scoped event identity and queued-versus-completed status remain
unchanged. The identity cache stays inactive.

The earlier attribution did not establish replay cleanup as the cause of its
tail. Its 376.71 ms p95 failure remains authoritative, and the original 100 ms
target remains unchanged. No new microbenchmark or complete timing comparison
ran. This is a demonstrated implementation change eligible for the remaining
prospectively frozen paired comparison after exact baseline/candidate artifact
preparation; it is not a promise that the comparison will pass.
