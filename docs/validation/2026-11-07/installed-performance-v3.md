# Installed performance: retrieval passes, complete hooks miss the target

The installed `749f1213` package completed all 155 planned observations against
10,000 synthetic accepted facts with real BGE embeddings. Warm recall meets the
500 ms p95 requirement. Both complete hook commands exceed the unchanged
100 ms requirement. All 102 captured events and their durable jobs survived a
service restart; no timed request failed its operational checks.

| Path | Measured attempts | Failures | Median ms | p95 ms | Maximum ms | Limit ms | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Warm HTTP recall | 50 | 0 | 299.41 | 403.37 | 773.11 | 500 | PASS |
| Complete Stop command | 50 | 0 | 905.64 | 2089.01 | 2729.79 | 100 | FAIL |
| Complete UserPromptSubmit command | 50 | 0 | 1287.49 | 2103.02 | 3131.65 | 100 | FAIL |

Three recall warmups and one warmup for each hook are retained separately.
The p95 uses inclusive linear interpolation. No failed or slow observation was
dropped, retried or replaced with a narrower request timer.

## What ran

This is the installed wheel, outside the checkout, with FastEmbed 0.9.1,
BGE-small-en-v1.5 and 384-dimensional embeddings. All cognition modules, LLM
extraction, judgment and query-state classification are disabled. The database
and runtime are on the system volume; the verified model cache is on the
development volume. Document embedding uses 79 real batches for 9,000 project
facts and 1,000 global facts. Every fact is visible to the scoped request.

Recall includes query embedding, ranking, access writes, evidence lookup,
context construction and loopback transport. Results have current versions,
retained source excerpts, at most five facts and a conservative 1,200-byte
context bound. This verifies the returned records and execution path, not
whether those records answer the question.

Hooks start the installed Python interpreter and unchanged proposed pilot
launcher for every command. Their timers include process startup, configuration
checks, the Codex adapter and transport; prompt hooks also perform recall.
Payloads are synthetic. No native Codex conversation or Claude session ran.

Fixture model construction took 858.95 ms, followed by 28.60 seconds for real
document embedding and loading. Cold service startup was 3904.78 ms and restart
was 3019.32 ms. These setup timers include Windows ownership verification
(2005.95 and 1192.04 ms respectively). They are cold processes with model files
already read during fixture construction, not cold operating-system cache
measurements. Ownership checks are outside warm recall and hook timers.

## Durability, audit and cleanup

The independent audit recomputed all timings and success checks from the raw
observations, verified 104 bound files and 17 model assets, and matched all
10,000 stored document vectors and vector-index entries to their recorded
embedding batches. SQLite integrity and foreign-key checks passed. Auditing
made no model calls and left the database bytes unchanged.

The final database contains the original 10,000 accepted facts plus 100 raw
chunks marked `unknown`. It contains 10,102 source events, 10,100 evidence
links and 102 durable capture jobs. One hundred jobs completed; two remain
pending. Durable capture therefore passed for all 102 events, while background
processing was not complete for two of them. These captures are not additional
accepted facts.

Both service interpreters were verified through their parent, executable,
exact command and creation time before authenticated shutdown. Both launchers
exited successfully, the interpreters are gone and the descriptor was removed.
No forced cleanup was needed. Total runner time was 196.08 seconds.

The complete evidence and bound inputs are preserved in a verified private
archive with 125 members, SHA-256
`9eb344adb7d555447cd3a67c95d6865d450b012e13d8a5331ef2a67e8f420bb3`.
The two failed setup attempts remain preserved separately. V3 repairs their
fixture identity and Windows process-ownership assumptions without changing
the product, workload or latency requirements.

This result leaves hook performance open. It supplies no semantic-quality
acceptance, general reliability rate, native-host acceptance or pilot days.
The private pilot remains inactive. Diagnose the complete command delay in a
separate experiment; preserve this result when evaluating any repair.

- [Aggregate receipt](installed-performance-v3-receipt.json)
- [Frozen workload and thresholds](../../plans/2026-11-07-installed-performance-v1.md)
- [Windows ownership amendment](../../plans/2026-11-07-installed-performance-v3.md)
- [V2 failed attempt](installed-performance-v2-failure.md)
- [Installed package checkpoint](checkpoint-749f1213.md)
