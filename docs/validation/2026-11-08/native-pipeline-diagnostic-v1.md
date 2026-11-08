# Native pipeline diagnostic

Status: completed and independently audited. All 110 events and jobs completed, including warmups. Instrumented command p95 was Python 1029.22 ms and native 422.63 ms, both above the unchanged 100 ms requirement. This is stage attribution, not acceptance timing or a speedup comparison with an earlier run.

| Native interval | Median ms | p95 ms |
| --- | ---: | ---: |
| Complete command | 191.82 | 422.63 |
| Outside main marks | 37.17 | 72.63 |
| Connection setup | 1.92 | 2.53 |
| Request written to first response byte | 150.14 | 384.75 |
| Header reading after first byte | 0.13 | 0.20 |
| Worker project identity | 66.85 | 295.77 |
| Worker durable flush | 60.20 | 124.71 |
| Worker after dispatch | 0.14 | 0.43 |

Most native time in this run is spent waiting for the response. Worker measurements locate substantial time in authoritative project identity and durable flush. The remaining post-dispatch interval is small with the response-completion implementation. This does not prove a causal patch speedup: the old run occurred under different conditions. Transport parsing is not the leading measured target.

The next optimization must address identity cost without stale scope routing and persistence cost without acknowledging undurable data. The inactive identity cache still needs complete external Git-policy invalidation. Durable flush cannot simply be skipped or deferred past a success receipt. Outside-main residual includes process creation before main, timing-record output and exit.

The frozen protocol used five warmups and fifty alternating measured pairs per launcher, one resident queued worker and one real service. The installed response-completion wheel remained unchanged. Both arms had identical context-local worker instrumentation and an observer write. Only native used the separately built stage-timing binary. Nested stage durations overlap; table percentiles must not be added. Queue drain and completed jobs were checked outside the command timer. Request-to-first-byte contains both worker and transport work.

The audit verified the schedule, 110 response hashes, event/job identities, native mark order, worker nesting, full command samples, package/binary hashes, protocol/helper hashes and source binding. All 55 native traces, including warmups, have the expected marks and monotonic times. Eighty Python checks and two Rust deadline tests validated instrumentation before the run. No source, package or binary changed during measurement.

Reopened SQLite storage contains 110 events and 110 completed jobs. Owned process identities were independently checked absent, and worker and memory ports were verified closed. The intentionally terminated worker returned exit code 1; measurement returned 0. The private archive preserves raw timings, traces, receipts, database, helpers, source and protocol. Its first member-verifier attempt used Windows separators; normalizing ZIP member lookups corrected the verifier, and all original hashes passed without rewriting the archive. See the [receipt](native-pipeline-diagnostic-v1.json) and [protocol](../../plans/2026-11-08-native-pipeline-diagnostic-v1.md).

No live configuration, optional pilot installation or production defaults changed. This deterministic empty-store study does not establish native-host dispatch, scale, power-loss durability, quality, fresh acceptance or actual pilot days. Earlier timing failures remain unchanged.
