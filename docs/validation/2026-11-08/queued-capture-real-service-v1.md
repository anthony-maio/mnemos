# Queued capture with the real memory dispatcher

The isolated queued-capture candidate now has a test using `LocalMemoryService`, SQLite persistence, the resident hook worker, and authenticated loopback delivery. Models are unavailable, embeddings are deterministic, and all experimental modules are disabled.

The test queues a Stop event while the memory service is unavailable, then restarts the worker against the real dispatcher. Two direct clients send the same source event while the worker replays it. All delivery receipts identify one job; the database contains one source event and one pending job, with no processed fact yet. Explicit maintenance completes that job. A same-ID correction increments the fact version, then reviewed-version forgetting removes it. A separate interpreter reopens SQLite and observes the persisted source suppression. Recreating the original outbox event and restarting both owners produces no fact, pending job, or source resurrection.

The new integration case and six existing capture/forget race cases passed together: seven checks. Formatting and diff checks passed. The first draft expected a `committed` boolean in the real capture receipt and failed before checking the lifecycle. Source inspection confirmed the existing contract uses `status="committed"`; the test was corrected without a production change. The original draft attempt and later disposable test directories remain separate.

This proves the bounded real-dispatcher lifecycle with deterministic embeddings. The separate process only verifies persisted SQLite state; the worker itself is restarted within the parent test process. It does not establish installed-package or native-host behavior, automatic extraction quality, operating-system worker restart, capture timing, or general reliability. No live configuration, pilot data, or production default changed.

Next build and install this exact candidate into an isolated environment and exercise worker process exit/restart with pending delivery. Retain queued versus completed evidence and original source identities, then measure complete acknowledgment latency. The 100 ms capture gate remains unproven.
