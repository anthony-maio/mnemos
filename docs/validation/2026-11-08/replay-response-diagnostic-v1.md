# Replay response scheduling diagnostic

Deferring replay notification until after native command exit reduced the observed wait between dispatch completion and service-submit return. All 110 events retained their original identities, persisted and completed. This supports replay scheduling as a contributor to the response delay; it does not establish a passing full-command target.

| Measure | Immediate median ms | Immediate p95 ms | Deferred median ms | Deferred p95 ms |
| --- | --- | --- | --- | --- |
| command_ms | 242.02 | 498.51 | 197.52 | 383.21 |
| submit_ms | 175.70 | 332.65 | 149.59 | 283.30 |
| dispatch_ms | 152.00 | 309.08 | 149.22 | 282.90 |
| before_dispatch_ms | 0.27 | 0.37 | 0.26 | 0.58 |
| after_dispatch_ms | 20.62 | 50.77 | 0.13 | 0.68 |
| project_identity | 68.63 | 230.78 | 67.22 | 230.12 |
| enqueue | 75.35 | 168.35 | 68.53 | 115.43 |
| fsync | 62.13 | 99.10 | 59.22 | 104.44 |

The frozen comparison used one queued worker, one real memory service, the verified native v3 launcher, unchanged installed package, deterministic embeddings and a fresh synthetic store. Five warmups per arm preceded 50 alternating measured pairs. Both arms used the same stage timers and receipt writer. Nested times overlap. The table does not subtract independent percentiles.

Immediate dispatch notified replay normally. Deferred dispatch suppressed its notification and restored the method before returning. After command exit, the parent sent an authenticated diagnostic operation to wake replay. Both arms used a 3600-second retry interval after the first dispatch. Each queue drained and its storage job completed before the next sample. These observer interventions are not product APIs or a deployed implementation.

The independent audit checked complete sample order, frozen helper and wrapper hashes, original event identity, correlated receipts, stage intervals and one Git resolution plus one fsync per command. Worker ownership cleanup, closed service ports, unchanged package/native hashes and reopened storage were verified. Both arms still missed the 100 ms command target.

Next is a production-shaped response-completion notification path, with tests for direct service callers, failed HTTP writes, cancellation and shutdown. Background timer and restart replay must remain recovery paths. The candidate must be rebuilt and measured without diagnostic endpoint intervention before any delivery or native-host claim. No live configuration, database or pilot installation changed.
