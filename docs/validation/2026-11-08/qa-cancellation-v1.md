# QA transport cancellation checks

Three model-free subprocess checks passed against the unchanged owned QA transport and concurrent support candidate `2a065f04f78e024dd6035e46fa0a2add54e9b83e`:

- A normal request returned a matching request ID and wire digest, then stopped cleanly.
- Caller cancellation during a blocked request propagated cancellation after stopping the worker.
- A transient failure in the concurrent contrast selector cancelled the blocked QA request and returned unavailable with no supported facts.

Each check used a newly launched disposable Python worker with the existing wire protocol. The stub loaded no model and performed no inference. Windows process ownership was checked using the unchanged helper before shutdown, and the verified interpreter was absent afterward. Every stopped worker rejected a subsequent request. Both cancellation cases marked the transport tainted and required forced termination; neither returned a successful abstention.

Caller cancellation and cleanup took 4,644.11 ms. The sibling-failure case, measured from composition entry through cleanup, took 4,492.86 ms. These are two single observations with different start boundaries, not comparable performance samples or percentiles. They establish a practical limitation: the shared timeout initiates cooperative cancellation, but the operation waits for process cleanup before returning. It is not a hard wall-clock response deadline.

The test retained the original transport and ownership helpers unchanged. Private evidence includes their hashes, the composition hash, stub and driver hashes, per-process ownership records, settings, logs, and terminal results. All three workers stopped and the driver exited zero. No live memory database, model server, pilot configuration, frozen evaluation input, or product default changed.

The result supports transport isolation for the tested failure paths. It does not establish model quality, cancellation of real inference, provider concurrency capacity, complete retrieval latency, or compliance with the 500 ms gate. A timing experiment must report cancellation cleanup separately and must not claim the configured timeout as its measured latency bound.
