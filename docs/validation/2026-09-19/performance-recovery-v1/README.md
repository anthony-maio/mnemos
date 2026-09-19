# SSD baseline performance after recovery repairs

Three fresh repetitions on verified source `de6cdcf1c05fdad1bb19ed8f52638eff5ed67ae4` pass
both unchanged baseline latency gates. Each uses 10,000 synthetic stored
vectors, 33 timed real Ollama query embeddings after three warmup queries,
and 50 timed durable captures after one warmup event. All 150 captures
committed. Capture uses the simple 768-dimensional fixture provider;
cognition calls are zero and all five modules are disabled.

| Run | Recall p95 ms | <= 500 ms | Capture p95 ms | <= 100 ms |
| --- | --- | --- | --- | --- |
| 1 | 251.321 | PASS | 36.062 | PASS |
| 2 | 252.222 | PASS | 28.804 | PASS |
| 3 | 245.214 | PASS | 31.766 | PASS |

All databases and outboxes use explicit C: paths on the workstation SSD;
SQLite retains WAL/FULL durability. Work directories are retained. The
[storage diagnostic](../capture-placement-v1/README.md) records the physical
SSD/HDD distinction. The [package checkpoint](../loopback-recovery-checkpoint-v1/README.md)
binds the source and installed checks used before these runs.

The first seed embedding took 6,691.178 ms with no models reported loaded
before execution. Later seeds took 250.613 and 205.151 ms. These are recorded
separately from warm recall, not discarded observations. The synthetic cold
Codex adapter subprocess took 722.871, 647.449 and 1,210.466 ms; this is not a
native Codex conversation or a native hook deadline acceptance result.

The earlier [local performance failure](../performance-local-v1/README.md)
remains preserved. These new runs change the source and select explicit SSD
storage on a shared workstation; they do not isolate a causal code-speedup
factor against that earlier run. They establish baseline latency under these
recorded conditions, not semantic quality, extraction accuracy, full-module
latency, native-host reliability, or pilot acceptance. No threshold, durability
setting, model digest, or scoring dataset was adjusted to obtain this result.

The full raw queries, component timings, commands, model bindings, source
hashes and helper bytes are archived here. Nested stage timings must not be
added to full retrieval latency. Evidence indexes preserve exact file bytes.
