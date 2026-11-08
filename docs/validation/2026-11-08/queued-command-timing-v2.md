# Installed queued capture command timing

Both installed Python command paths fail the unchanged 100 ms target. Queue-only admission provides a distinct durable boundary, but this experiment does not make the complete command fast enough.

| Arm | Measured commands | Minimum | Nearest-rank p95 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Synchronous capture | 50 | 366.02 ms | 872.21 ms | 1044.78 ms |
| Queued capture | 50 | 321.73 ms | 796.74 ms | 1064.46 ms |

Five warmup commands per arm preceded 50 alternating measured pairs. Each timer covered Python subprocess startup, the installed hook client, private authenticated-response trace writing, stdout/stderr collection, and process exit. Trace-writing overhead is included. Both resident workers and the real memory service were already running. No native Codex dispatch or outer shell startup was measured.

Every command response was correlated with a unique SQLite event and job. All 110 events persisted and all 110 jobs completed; the store was reopened after shutdown to verify those counts and completion states. The queue was drained before the next command. The source used deterministic embeddings, unavailable cognition, disabled experimental modules, and an initially empty synthetic store. This does not establish 10,000-fact retrieval, loaded backlog behavior, real-model performance, cold service startup, or general reliability. The small arm difference does not isolate its cause.

The first attempt stopped during the first warmup's durability check because the helper accessed source_ref as an event attribute instead of its metadata field. It completed no measured samples. That attempt, its output, and verified cleanup remain preserved separately. V2 corrected the verifier, added exact queued-source identity checks, and retained the frozen schedule and candidate. The installed package was unchanged before and after both attempts.

The independent audit recomputed the sample counts and p95 values, verified the exact paired schedule, matched response hashes and resident dispatch, and checked unique source/job identities. Both owned workers were terminated after ownership verification, their processes were absent, and all worker/service ports were closed. Controlled worker exit codes were 1; the run and audit helpers exited zero.

Private evidence is retained under `.venv/queued-command-timing-v1` and `.venv/queued-command-timing-v2`, including freezes, traces, command outputs, timing rows, SQLite state, and cleanup. Earlier failed timing studies remain unchanged. No live configuration, default, publication, or original pilot acceptance changed.

Next attribute command overhead before changing the delivery mechanism again. Separate Python startup/import, local transport, worker queue persistence, and trace costs with the same artifact. A native-launcher comparison must preserve the actual queued-state correlation and complete command boundary; an internal transaction timer cannot replace this failed result.
