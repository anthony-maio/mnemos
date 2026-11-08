# Bounded replay batch comparison

Compare installed v6 and v7 with three paired backlogs of 35 synthetic captures, alternating execution order by pair. Each arm gets a fresh SQLite outbox and actual local memory service database. Both arms use the same project registry and exact payloads within each pair. No live configuration or memory is used.

Time one complete replay call per arm, including client connection, delivery and durable acknowledgment. Queue preparation, service startup, duplicate checks and integrity inspection are outside the timer. There is no warmup. Report every result and paired differences; three pairs cannot establish a tail-latency claim. This measures backlog replay, not capture latency or completed extraction.

Require 35 acknowledgments, zero failed or pending entries, exactly the original event IDs and 35 jobs. Redeliver all payloads through the actual service and require unchanged job IDs and event count. Check both databases for integrity after closure. Observe outbox commit counts: 35 for v6 and two for v7. Preserve failures and do not replace failed measurements silently.

Use the previously validated v6 and v7 installed environments. Freeze the runner hash before execution. Preserve per-arm inputs and results, package hashes and the execution log. Keep the candidate isolated regardless of the timing result.
