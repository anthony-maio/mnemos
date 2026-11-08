# Installed queued-worker process restart

The isolated wheel `f9160ee1ca1bc09f169c09f7ad3c61ca5e45019c66150b29d14903fbebb3f5c3` passed a separate operating-system worker restart check. The helper ran under the installed environment in isolated Python mode and verified all 97 package files before and after execution.

A first owned Codex worker received one synthetic Stop event while the memory service was unavailable. Its authenticated internal response reported queued state and the original source-event ID; host output remained `{"continue":true}`. The event was present in the durable outbox before the helper forcibly terminated the owned worker process tree. The same outbox bytes remained available after termination.

The helper then started the real memory dispatcher with SQLite, deterministic embeddings, unavailable cognition, and experimental modules disabled. A new worker process started with a different PID and drained the old event without a new hook request. The store contained one source event and one pending job, with no processed fact. Explicit maintenance completed that job and produced one passthrough fact. Pending capture and completed processing were recorded separately.

Both worker process trees were checked against their recorded ownership before termination. Both were verified absent afterward; worker ports and the memory-service port were closed. Worker exit code 1 reflects controlled taskkill termination, while the test helper exited zero. No unrelated process, live configuration, pilot database, or installation was changed.

Private evidence under `.venv/queued-worker-process-v1` retains the freeze, package identities, dependency inventory, owner snapshots, logs, queue receipt, event bytes, pending/completed state checks, and cleanup result. This is one deterministic installed scenario, not native Codex dispatch, real embedding/model evaluation, power-loss durability, or a reliability estimate. No acknowledgment latency was measured.

Next freeze a timing protocol that measures complete installed hook-command output and exit, correlates every queued acknowledgment with eventual durable job completion, and reports service admission separately. Preserve original failed timing results. The 100 ms requirement cannot be claimed from internal outbox or transaction timing, and native-host dispatch remains a separate boundary.
