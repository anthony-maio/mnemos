# Full-sync SQLite outbox component comparison

Status: isolated candidate implemented, tested and measured. It is not selected by adapters, installed into the pilot, or promoted to canonical delivery.

The candidate uses SQLite WAL with synchronous=FULL and returns an append receipt only after COMMIT. It shares entry preparation with the JSON outbox, preserving sanitization, original event IDs, profile/host/session namespacing and conflicting-duplicate rejection. Acknowledgment deletes the row in another full synchronous transaction. Separate files are required; there is no automatic JSON migration. The owner must close the connection.

| Operation | JSON median ms | JSON p95 ms | SQLite median ms | SQLite p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Append through receipt | 68.82 | 87.34 | 40.40 | 50.23 |
| Acknowledge through return | 3.18 | 8.35 | 41.97 | 62.59 |
| Per-sample append plus acknowledgment | 71.13 | 90.04 | 80.44 | 104.50 |

SQLite append was faster in all fifty measured pairs, with a median paired difference of -26.78 ms. Acknowledgment was slower in all fifty, with a median difference of +38.63 ms. The complete component cycle was slower in 41/50 pairs. This supports investigating the queued-response boundary, where replay acknowledgment occurs later; it does not establish higher throughput or faster complete capture.

Both backends began each sample with an empty queue and received identical synthetic payloads. SQLite retained its connection. Five warmups preceded fifty alternating measured pairs. Pending reads, validation and reopening were outside timing. The first untimed pending() call initialized SQLite before the first append, so this run contains no cold-initialization measurement, including in warmups. This clarifies the protocol's warmup language; no cold latency is claimed. Append-plus-acknowledgment totals were calculated within each sample before percentiles.

The acknowledgment implementations have different persistence paths: SQLite performs a full synchronous delete transaction; the empty JSON queue is unlinked. This run verifies their logical outcomes and process-restart behavior, not equivalent power-loss guarantees. No flush was removed from either implementation to obtain these numbers.

Twenty-eight focused checks passed, including thirteen new backend checks; one replacement-path check skipped because Windows prevents renaming the open SQLite file. Coverage includes JSON/SQLite identity and payload parity, duplicate conflicts, failed commit rollback, failed acknowledgment retention, separate connections and processes, an external writer lock, rejected foreign databases, rejected JSON migration, copied payloads and sanitizer failure, and abrupt process exit without close followed by WAL recovery. The skipped replacement scenario remains unverified on a platform that permits it. Strict type checks passed for both changed source modules.

The independent audit verified the frozen helper/protocol/source hashes, all 110 raw records, schedule, derived event identities, exact paired payloads, non-overlapping arithmetic and every summary statistic. Both queues were empty after close/reopen; the final SQLite integrity check passed. The archive preserves source, tests, JUnit, protocol, helper, raw timings and the final database. Source stayed unchanged during measurement. Candidate commit: `c515bf31`.

The next step is explicit worker/client lifecycle integration in the isolated candidate, with installed capture/correct/forget/replay and restart tests before another complete command measurement. Connection cleanup, JSON backlog compatibility and queued-versus-completed status require review. A component gain does not waive the original 100 ms capture requirement, Git-policy invalidation, quality gates, fresh acceptance or actual daily-use evidence. The existing installed package and optional pilot remain unchanged.

See the [receipt](sqlite-outbox-component-v1.json) and [frozen protocol](../../plans/2026-11-08-sqlite-outbox-component-v1.md).
