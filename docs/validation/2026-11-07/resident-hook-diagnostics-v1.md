# Resident hook cost diagnosis

Storage search, command startup and Git-based project resolution account for
most of the remaining cost in this diagnostic run. Real query embedding takes
about 5 ms. Changing embedding concurrency is therefore not the next supported
optimization. Narrowing the SQL sort projection preserves results but saves
only about 5 ms per search on these controls.

This run uses timing wrappers around the installed candidate's original functions.
It identifies work to investigate; it does not replace the failed 100 ms
[complete-hook comparison](resident-hook-performance-v1.md).

## Observed costs

The run contains five empty-command startup controls, one warmup and five observed
Stop commands, and one warmup and five observed UserPromptSubmit commands.
The following values are medians in milliseconds. Nested times overlap and must
not be added together.

| Observed work | Stop | UserPromptSubmit |
| --- | ---: | ---: |
| Complete instrumented command | 440.53 | 655.42 |
| Worker dispatch | 216.12 | 393.72 |
| Effective-settings check | 2.54 | 2.64 |
| Git-based project resolution | 170.99 | 60.56 |
| Adapter capture | 38.03 | 40.79 |
| Durable outbox append | 9.48 | 9.94 |
| Service capture transaction | 2.74 | 2.84 |
| Adapter retrieval | -- | 284.14 |
| Service retrieval dispatch | -- | 271.85 |
| Real query embedding | -- | 4.78 |
| Vector storage search | -- | 123.21 |
| Lexical storage search | -- | 111.49 |

The empty-command median was 217.32 ms, including the diagnostic entry script.
Its imports took 116.37 ms; time outside that script's measured body was 87.63 ms.
These are small, instrumented samples with substantial startup variation, not a
platform-wide lower bound or a new latency gate. They do establish that optimizing
only the service cannot remove the command's process and import cost.

## Read-only SQL probe

The current SQL selects full chunk payloads while ranking candidates. A development
probe instead selected rowid, identifier and score, applied the same limit, then
joined the winning chunks and restored their order. It retained exact cosine,
BM25, scope predicates and identifier tie order.

The probe opened the earlier pristine seed as an immutable, read-only database.
It used ten fixed lexical queries from the prior manifest and ten existing
document vectors as vector-search controls. Those vectors are not new query
embeddings or fresh quality data. Two warmup pairs preceded ten observed pairs
per path; arm order alternated.

| Search | Original median ms | Narrow projection median ms | Median paired change ms |
| --- | ---: | ---: | ---: |
| Vector | 87.42 | 82.50 | -5.02 |
| Lexical | 99.57 | 93.91 | -4.35 |

All 24 pairs, including warmups, returned exactly the same 30 full rows in the
same order. Vector scores also matched exactly. The SQL-only timings are not
directly interchangeable with the instrumented service timings above.

The first probe stopped before its first timed ranking query because its harness
called an instance serializer as a static method. V2 used the same sqlite-vec serializer
directly, preserved the failed attempt and completed all 48 observations. No
product source or frozen database changed. The small gain does not justify
claiming that full payload selection explains the search cost.

## Evidence and next work

The diagnostic used candidate `feb6ac3b3eebd2e178ec584b3be1e72cca1228fc` and the
same installed wheel, dependencies and BGE cache as the complete comparison.
Its 10,000 accepted facts and vectors came from the pristine seed. The audit
verified 110 file bindings, 17 model assets, 12 worker dispatches, 18 memory-service
dispatches and six real prompt embeddings. All 12 captured source/job records
survived shutdown. All 12 jobs remained pending; no completed-fact claim follows.

The accepted cohort and original seed remained unchanged. Both owned processes
exited cleanly, their descriptors were removed, and the audit left the final
database bytes unchanged. Effective settings were recorded only as a digest;
provider credential values were not written to the diagnostic records. Live host
configuration and pilot material were untouched.

The verified private archive contains 143 members. SHA-256:
`5396507567b0180f28b021890988660c61195bcec0ecddbedc6d244ad305127c`.

The next candidates are the repeated scope/JSON work in storage, Git identity
resolution at each event, and the command's interpreter/import boundary. Any
change must preserve current scope membership, source identity, durable capture,
retrieval and checked fallback. A faster component still needs a complete
installed comparison and native-host acceptance. The pilot remains inactive.

- [Diagnostic protocol](../../plans/2026-11-07-resident-hook-diagnostics-v1.md)
- [SQL probe protocol](../../plans/2026-11-07-sql-ranking-probe-v1.md)
- [Serialization repair](../../plans/2026-11-07-sql-ranking-probe-v2.md)
- [Verification receipt](resident-hook-diagnostics-v1-receipt.json)
