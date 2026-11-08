# Queued command boundary diagnostic

All 110 instrumented commands retained their original event identities, persisted one event each and completed their jobs. The audit verified the frozen schedule and helper hashes, receipts, stage nesting, artifact hashes, process cleanup and closed ports. This diagnostic does not replace the earlier command timing failure.

| Inclusive stage | Python median ms | Python p95 ms | Native median ms | Native p95 ms |
| --- | --- | --- | --- | --- |
| command | 368.17 | 647.82 | 240.33 | 624.79 |
| dispatch | 157.32 | 389.14 | 149.55 | 310.76 |
| enqueue | 75.30 | 116.88 | 70.33 | 134.34 |
| outbox_write | 72.86 | 106.68 | 66.66 | 131.59 |
| fsync | 66.27 | 100.42 | 61.09 | 93.58 |
| sanitize | 0.41 | 0.44 | 0.41 | 0.51 |
| client_protect | 2.32 | 4.75 | 2.35 | 7.99 |
| service_protect | 1.32 | 6.29 | 1.37 | 6.44 |
| lock_acquire | 1.73 | 6.78 | 1.79 | 7.01 |
| dispatch_outside_enqueue | 73.69 | 266.58 | 73.92 | 213.00 |
| outside_dispatch | 202.75 | 373.62 | 69.89 | 285.61 |
| scope_resolve | 68.71 | 261.99 | 69.56 | 208.55 |
| project_identity | 68.67 | 261.95 | 69.52 | 208.50 |
| subprocess_run | 66.64 | 258.86 | 67.25 | 200.95 |
| settings_fingerprint | 2.52 | 3.37 | 2.52 | 3.96 |
| submit_ms | 176.37 | 404.45 | 170.12 | 335.06 |
| before_dispatch_ms | 0.26 | 0.50 | 0.27 | 0.34 |
| after_dispatch_ms | 20.37 | 34.12 | 19.62 | 54.56 |
| outside_submit | 180.69 | 348.77 | 46.82 | 266.63 |
| dispatch_other | 4.37 | 5.58 | 4.33 | 10.48 |

Nested stages overlap. Enqueue includes the outbox write, which includes fsync; adding these durations would double-count work. Remainder rows subtract durations within each individual sample before calculating percentiles. Submit includes dispatch; before/after dispatch partition its remaining time. Dispatch other excludes enqueue and scope resolution. They do not subtract independent percentiles.

The worker used context-local timers around unchanged installed methods and wrote each receipt after synchronous service submit returned, before HTTP serialization. Background replay ran outside the dispatch timing context. Timer calls and receipt tracing add overhead; the original uninstrumented package files and native executable stayed unchanged. Both launchers used one queued resident worker, deterministic embeddings, disabled cognition, five warmups each and 50 alternating measured pairs. Completion and queue drain were outside the command timer.

Every measured dispatch made one Git subprocess call for project identity resolution. The audit checks nested scope, identity and subprocess times, and reconciles submit duration with its before-dispatch, dispatch and after-dispatch intervals. Project identity is now a measured target for optimization. A resident cache would need explicit invalidation for changed repository markers, worktree/common-directory routing, filesystem identities and registry state; blindly retaining a path would weaken scope isolation. After-dispatch delay is also measured, but its cause remains unresolved. Outside-submit time includes process launch and exit, HTTP handling and observation work. A follow-up should develop and test identity invalidation before any performance claim. Durable acknowledgement still requires the disk flush and permission policy; this diagnostic does not justify removing either.

No live configuration, live memory database, pilot installation or published artifact changed. Native-host acceptance, 10,000-fact latency, original quality gates and actual pilot days remain open. Raw private evidence is archived under the hash in the companion JSON receipt.
