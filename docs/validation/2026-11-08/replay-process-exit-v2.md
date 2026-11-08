# Replay recovery at the batch commit boundary

Two source regression cases passed in 8.30 seconds. Each queues 35 synthetic captures and starts the actual memory service. A separate replay interpreter exits with os._exit(73) at the first 32-entry acknowledgment boundary, either immediately before the deletion transaction or immediately after it commits. The parent verifies that the service has already persisted exactly 32 events and jobs, then stops and restarts the service against the same database.

Before-commit exit leaves all 35 original queue entries intact. After-commit exit leaves exactly the final three entries. Reopening the queue verifies the complete remaining records against their originals, including payloads and identity. Replay against the restarted service finishes with exactly 35 events and jobs, retaining every previously created event and job ID. Source event IDs match the original payloads and the queue ends empty. This exercises actual process termination and service persistence rather than synthetic duplicate-success replies.

The initial run reached the final assertion but compared source event IDs with the outbox's scoped envelope IDs. Both cases failed that incorrect assertion. The corrected test compares the payload IDs; the initial JUnit and fixtures remain preserved as v1, and the passing run used a fresh v2 temporary directory. No runtime code changed. The cases establish recovery at these two explicit boundaries, not arbitrary mid-transaction termination or power-loss behavior.

Test commit: 0684c983 in the isolated capture candidate. Test SHA256: 75227458536694bbbffdc0c0b0df62cbdf7cb55428de913dbf08fae7286d6423.

Passing JUnit: `.venv/replay-process-exit-v2.xml`, SHA256 5399ebc7aab5aa70458c5cd6e6125a0d6370f994e2de23fa3c454383769cfd23. Failed first-run JUnit: `.venv/replay-process-exit-v1.xml`.

This is source validation using the v7 environment's dependencies. It is not a rebuilt package or an installed-test result. The canonical runtime and live pilot remain unchanged. The original quality, latency and daily-use gates remain open.
