# Direct lexical scope filtering regresses latency

The candidate preserved every ranked result but made scoped searches slower.
It is rejected; canonical product SQL remains unchanged.

| Scope | Baseline median / p95 ms | Candidate median / p95 ms |
| --- | ---: | ---: |
| unrestricted | 41.95 / 53.01 | 36.87 / 45.73 |
| project | 36.78 / 45.21 | 184.36 / 216.08 |
| absent | 2.16 / 4.79 | 142.57 / 189.55 |

The fixed comparison contains 504 paired rankings, including 84 warmup pairs.
All IDs, bm25 scores and tie orders match. Each table arm contains 140 timed
observations; p95 uses nearest rank. Timing covers ranking SQL only, without
full chunk hydration, parsing, embedding, host startup or capture.

Both arms read separate copies of the same 10,000-fact seed. The seed predates
the current scope index. Two preflights stopped before timing: the first rejected
an empty WAL sidecar; the second found the missing index. Their logs and helpers
are retained. The amended protocol adds the exact canonical index to both copies,
verifies all chunk rows unchanged, then freezes their hashes before timing.
Original seed and prepared-copy hashes remain unchanged after the comparison.

The candidate forces FTS matches first with CROSS JOIN and applies the same
scope predicate before LIMIT. Query plans confirm removal of the authorized-row
subquery. That removes a little work for unrestricted searches but loses the
benefit of the covering scope index for project and absent-profile searches.
The plans and timing support retaining the current strategy; they do not isolate
every source of runtime cost. No thresholds or queries were tuned after results.

An independent read of all 1,008 observations verifies the 504 equal result
pairs and recomputes the six timing summaries. The private evidence archive
retains queries, results, plans and failed preflights; database copies remain
separately hash-bound. The probe process exited zero.

This failed optimization does not change the original latency requirement.
No product, live database, host configuration or evaluation score was changed.
Future latency work must use a different hypothesis rather than repeat this
comparison. See the [receipt](lexical-scope-probe-v1-receipt.json).
