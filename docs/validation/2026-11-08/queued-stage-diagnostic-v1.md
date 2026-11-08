# Queued command stage diagnostic

All 110 instrumented commands retained their original event identities, persisted one event each and completed their jobs. The audit verified the frozen schedule and helper hashes, receipts, stage nesting, artifact hashes, process cleanup and closed ports. This diagnostic does not replace the earlier command timing failure.

| Inclusive stage | Python median ms | Python p95 ms | Native median ms | Native p95 ms |
| --- | --- | --- | --- | --- |
| command | 371.31 | 876.62 | 219.23 | 490.07 |
| dispatch | 153.51 | 381.33 | 143.38 | 313.88 |
| enqueue | 78.21 | 112.15 | 70.98 | 101.32 |
| outbox_write | 74.03 | 103.40 | 67.91 | 94.67 |
| fsync | 66.69 | 94.08 | 61.97 | 88.41 |
| sanitize | 0.40 | 0.54 | 0.41 | 0.56 |
| client_protect | 2.38 | 4.84 | 2.22 | 5.54 |
| service_protect | 1.36 | 12.80 | 1.35 | 5.18 |
| lock_acquire | 1.85 | 13.26 | 1.79 | 5.70 |
| dispatch_outside_enqueue | 73.47 | 263.99 | 72.38 | 204.28 |
| outside_dispatch | 207.92 | 484.59 | 68.42 | 209.22 |

Nested stages overlap. Enqueue includes the outbox write, which includes fsync; adding these durations would double-count work. The two remainder rows subtract durations within each individual sample before calculating percentiles. They do not subtract independent percentiles.

The worker used context-local timers around unchanged installed methods. Background replay ran outside the dispatch timing context. Timer calls and receipt tracing add overhead; the original uninstrumented package files and native executable stayed unchanged. Both launchers used one queued resident worker, deterministic embeddings, disabled cognition, five warmups each and 50 alternating measured pairs. Completion and queue drain were outside the command timer.

Project identity resolution calls Git in the current source. That is a candidate explanation for dispatch work outside enqueue, not a measured attribution in this study. Outside-dispatch time also includes process launch and exit, HTTP handling, scheduling and observation work. A follow-up should isolate these boundaries before selecting a repair. Durable acknowledgement still requires the disk flush and permission policy; this diagnostic does not justify removing either.

No live configuration, live memory database, pilot installation or published artifact changed. Native-host acceptance, 10,000-fact latency, original quality gates and actual pilot days remain open. Raw private evidence is archived under the hash in the companion JSON receipt.
