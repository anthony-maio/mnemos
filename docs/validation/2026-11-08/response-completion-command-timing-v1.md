# Installed response-completion command timing

The actual installed response-completion implementation, with normal replay timers and no diagnostic notification intervention, preserved all 110 events across both launchers and completed their storage jobs, including warmups. Python command p95 was 598.64 ms; native command p95 was 621.55 ms. The unchanged target is 100 ms. The measurement completed successfully; that status does not establish a latency pass.

| Launcher | Measured commands | Minimum ms | p95 ms | Maximum ms | Meets 100 ms |
| --- | --- | --- | --- | --- | --- |
| Python | 50 | 291.85 | 598.64 | 678.75 | False |
| Native | 50 | 171.22 | 621.55 | 642.65 | False |

The frozen protocol used five warmups per launcher, then 50 alternating pairs against one warm queued worker and one real memory service. Timing starts before process creation and ends after stdout, stderr and exit are collected. Each command had a unique original event identity, a correlated queued worker receipt, one persisted event and one completed job. Queue drain and completion happened outside the command timer, before the next command. The independent audit checked the schedule, identities, receipt hashes, frozen helper hashes, sample counts and nearest-rank p95.

Both arms included the same worker-side trace write. The Python command used the installed module directly; the native command used the previously verified v3 binary. Installed wheel SHA256: `cfe4e9b480fb8232453d53e928a6a2ba840d26bb3a7a9106dc6bdc1968fd4c2e`. Native SHA256: `ec52c79a0ae613ae336836b3e82ddf853a601a0de3f7cd3aa0047318d06ee12b`.

This was an initially empty synthetic store with deterministic embeddings and cognition disabled. It does not establish native Codex host dispatch, 10,000-fact recall latency, power-loss durability, or actual pilot days. Earlier runs occurred at different times and are historical context, not a simultaneous baseline that isolates the patch speedup.

The owned worker process tree was stopped, worker and memory ports closed, and all 98 installed package files and the native binary stayed unchanged. Reopening storage confirmed 110 events and 110 completed jobs. No live configuration, pilot installation, or live database changed. The candidate remains isolated and unpublished.

Private raw evidence, commands, dependency identities, ownership checks, logs, receipts and database are retained in the archive identified by the companion JSON receipt. Original quality and pilot gates remain open.
