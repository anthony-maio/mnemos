# Installed backlog replay comparison

Batch acknowledgment reduced replay time in all three paired runs against the actual local memory service. Each run delivered 35 synthetic captures from a fresh SQLite outbox into a fresh service database. The median complete replay call took 4170.86 ms with v6 and 2573.86 ms with v7. These three pairs support a bounded backlog improvement; they do not establish tail latency, capture latency or completed extraction performance.

| Pair | Order | v6 ms | v7 ms | Difference ms |
| --- | --- | ---: | ---: | ---: |
| 0 | v6 then v7 | 4170.86 | 2646.25 | -1524.60 |
| 1 | v7 then v6 | 4225.13 | 2145.88 | -2079.24 |
| 2 | v6 then v7 | 3937.99 | 2573.86 | -1364.14 |

Every run acknowledged all 35 entries with zero failed or pending records. The actual stored event IDs matched the service's scoped identity calculation, and their source IDs matched the original inputs. Every run stored 35 jobs. Redelivering every payload through the actual service preserved the same job IDs and event count. Both databases passed integrity checks after closure. v6 committed 35 outbox deletions separately; v7 used two acknowledgment commits. Paired input hashes matched exactly.

The timer includes the complete replay call, client connection, delivery and durable acknowledgment. It excludes queue preparation, service startup, duplicate verification and database inspection. There was no warmup. Package hashes matched the earlier installed validation receipts. All six runs finished in one sequential controller session with exit status zero; each runner stopped its service and closed its client, queue and store before returning.

The first v1 attempt failed an incorrect benchmark assertion comparing stored scoped IDs directly with incoming IDs. That run and its original frozen runner remain under `.venv/replay-batch-benchmark-v1`. It is excluded from the six measurements. v2 checks both source identity and the scoped hash, with its runner frozen before execution. No product code changed for this correction.

The [prospective protocol](../../plans/2026-11-08-replay-batch-benchmark-v1.md) defines the comparison. The adjacent JSON records all measurements, package hashes and the evidence archive. The verified archive contains 41 files including inputs, results, databases, logs and runners; SHA256 c5e0172a69a8e6f497033559d67f9d77adc2928f6f08e7c5f3eae5e9ab12f715.

This candidate remains isolated. The 100 ms complete durable-capture requirement, answer-support quality, fresh acceptance and original daily-use pilot remain open. No live configuration, memory database or publication changed.
