# Current native capture attribution

The current canonical package still fails the complete-command 100 ms capture target. This diagnostic performed five warmups and fifty measured native Stop commands using a disposable Git project, a SQLite outbox, authoritative Git resolution and FULL synchronous commits. No identity cache was enabled. All fifty measured commands exceeded 100 ms.

| Interval | Median ms | Nearest-rank p95 ms |
| --- | ---: | ---: |
| Complete native command | 142.18 | 376.71 |
| Service submission | 104.47 | 234.84 |
| Dispatch, inside submission | 104.19 | 234.59 |
| Authoritative Git identity | 56.20 | 181.64 |
| Durable queue append | 43.26 | 59.03 |
| FULL synchronous COMMIT, inside append | 41.96 | 56.90 |
| Queue connection checks, inside append | 1.15 | 2.19 |
| Private-path protection, inside connection checks | 0.92 | 1.80 |
| Command time outside submission | 33.17 | 168.13 |

These intervals overlap. Do not add their percentiles. Complete command minimum was 122.34 ms and maximum was 717.28 ms. The maximum durable commit was 202.53 ms. The outside-submission interval includes native launch, transport, trace output, response handling, process exit and scheduling; it does not attribute those costs individually.

Subtracting each sample's Git interval leaves median 81.08 ms and p95 236.69 ms, with eleven samples still above 100 ms. This is accounting, not a causal prediction of cache performance. It establishes that eliminating measured Git time alone would not make this recorded tail meet the target. Privacy-check removal is not justified: those checks are small compared with Git, durable commit and the external command interval.

Every acknowledgement was correlated with its original event ID, stored content and completed job. A fresh audit reopened all 55 events and jobs after the owner stopped. Queue startup confirmed synchronous=FULL and integrity_check=ok. Both owned ports closed, the owner exited normally, and process absence was verified. All installed package, native binary and active helper hashes remained unchanged.

The package is canonical source `d9ab39ba6b23815c3b0457c1d389d2f70fb3fd4c`, wheel SHA256 `5182601cb4f8f4739bb8472e255e2e0ee019ae27362a65829d363e7df0464a6f`. The [protocol](../../plans/2026-11-08-current-capture-attribution-v1.md) was committed before execution. The private evidence archive `.venv/current-capture-attribution-v1-evidence.zip` has SHA256 `d37305b5a82d19f223765cfd1a246a4ffaf832fbfe17deac976627efc0366040`. It contains raw timings, responses, freeze, cleanup, queue checks, reopened database, audit and the exact executed helpers. The preparation helper generated initial files; the executed owner additionally asserts FULL mode and queue integrity, and its final bytes are bound in the freeze and archive.

This consumes the one current complete-path attribution permitted by the convergence reset. It is an instrumented synthetic native command, not actual Codex host dispatch, a before/after comparison, fresh acceptance or pilot use. The lower total than older runs does not establish an improvement: artifacts, instrumentation and runtime conditions differ. Earlier failed timing evidence remains unchanged.

## Implementation decision

Keep the identity cache inactive. Do not spend the paired comparison on cache integration while its dependency boundary remains unapproved and the measured non-Git tail already exceeds the target. The next design review must address the complete acknowledgement path, including whether authoritative project resolution can move after a durable queued receipt without weakening authorization, privacy, event identity, forgetting or queued-versus-completed semantics. This is a hypothesis to inspect against the current code, not authorization to change those guarantees. FULL commits remain required. Any demonstrated implementation change still receives at most the planned frozen paired comparison; no repeat of this diagnostic is scheduled.
