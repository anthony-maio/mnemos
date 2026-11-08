# Current capture stage diagnostic

Two fresh synthetic diagnostics used the canonical implementation at `4fbf6e58`. Each performed five warmup operations followed by thirty measured Git resolutions and SQLite appends. The second added timing wrappers around entry preparation and individual SQLite statements. Original functions still performed every operation. No cache, permission check or durable commit was bypassed.

| Stage, second diagnostic | Median | Nearest-rank p95 |
| --- | ---: | ---: |
| Authoritative Git project resolution | 66.06 ms | 319.96 ms |
| Complete SQLite append | 40.03 ms | 48.90 ms |
| Entry preparation, including sanitization | 0.17 ms | 0.32 ms |
| Connection checks and private-path protection | 2.72 ms | 7.24 ms |
| Private-path protection alone, nested above | 2.11 ms | 6.84 ms |
| FULL synchronous COMMIT, within append | 35.98 ms | 45.44 ms |

The first diagnostic measured Git resolution at median 68.11 ms and p95 338.76 ms, and append at median 42.51 ms and p95 70.69 ms. These are separate instrumented runs, not an optimization comparison. Wrapper overhead and machine scheduling remain part of the observations. Nested intervals must not be added, and separate percentiles must not be combined into an end-to-end percentile.

The append result directs attention to its durable commit rather than sanitization or repeated permission protection. The Git result keeps authoritative identity resolution on the critical path. Removing privacy checks would not address the measured dominant costs and would weaken the contract. The existing cache prototype still lacks a verified external Git policy invalidation owner; these measurements do not authorize enabling it or substituting a constant generation.

Both runs reopened their queues using distinct connections and observed all 35 acknowledged entries. Both verified FULL synchronous mode, SQLite integrity_check=ok and unchanged canonical source bindings. The first append includes schema initialization and was kept separately: 358.33 ms in v1 and 297.29 ms in v2. Raw observations and the exact helper are retained for each run.

These diagnostics omit native process launch, host transport, settings and registry checks, background replay contention and service completion. They establish neither the 100 ms complete capture gate nor native-host acceptance. Earlier complete-command failures remain authoritative for those requirements. No live configuration, memory database, original evidence or pilot setting changed.

The next implementation needs to preserve authoritative Git policy while avoiding repeated process startup, or otherwise reduce complete-command cost without acknowledging before a durable commit. Any identity reuse must detect external configuration/includes, repository and worktree changes, relevant environment changes and registry revocation. A narrower cache that silently ignores those changes is not an acceptable latency fix.
