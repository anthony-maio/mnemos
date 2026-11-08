# Controlled legacy chronology result

The installed 0.6.2 candidate passes the prospectively declared access-invariance
check. Reading stale content no longer makes it the newest content in this
comparison. On both stores and all three repetitions, the repaired engine returns
all eight expected facts when source chronology is known, before and after prior
access to old facts. The 0.6.1 engine returns eight before prior access and zero
after it. Direct vector retrieval returns three in either case.

This is development evidence for the specific recency repair. The questions were
already known, and the comparison does not qualify general answer support,
abstention, native-host behavior or release readiness. The original failed
benchmark, fixture, gold labels and production gates remain unchanged.

The [protocol](../../plans/2026-11-07-legacy-chronology-v1.md), worker, accepted
records and four harness tests were committed at `a4654e70` before measurement.
One run produced 1,152 observations, 384 matched route triples and 576 matched
access pairs. It used the existing installed packages: 0.6.1 at `9ea3852a` and
0.6.2 at `ace62950`. Neither was rebuilt. Both installations match their respective
35-file wheels; their embedder implementation and 40 common dependency entries
match. The additional pytest-asyncio dependency in 0.6.2 is recorded separately.

Each request starts from the same accepted snapshot for its condition. The eight
original facts share scope with an unrelated global control; eight foreign-project
copies and a workspace control are present but excluded. All routes receive the
same nine eligible candidates, actual query vector, twenty-candidate lookup,
top-one output limit and 2,048-byte budget. All common stored fields match before
each route comparison. The intentional representation difference is that 0.6.1
cannot retain the new content-revision field. Both 0.6.2 routes retain it exactly.

The two determinate conditions keep the original expected IDs. Each table cell
contains 48 observations: eight questions, two stores and three repetitions.
Those repetitions do not provide 48 independent questions.

| Source chronology | Prior stale access | 0.6.1 engine | 0.6.2 engine | Vector retrieval |
| --- | --- | --- | --- | --- |
| Ordered creation | No | 48/48 | 48/48 | 18/48 |
| Ordered creation | Yes | 0/48 | 48/48 | 18/48 |
| Explicit later content rewrite | No | 48/48 | 48/48 | 18/48 |
| Explicit later content rewrite | Yes | 0/48 | 48/48 | 18/48 |

Recall@1 and MRR are equal here because only one record is returned. Every
repetition has the same scores: 1.0 or 0.0 for 0.6.1 as shown, 1.0 for 0.6.2 and
0.375 for vector retrieval. The repaired engine loses none of the expected
answers that 0.6.1 returned in untouched determinate conditions. Memory and
SQLite selections agree across all 576 corresponding store pairs.

Prior access changes 162 of 192 paired selections in 0.6.1: 48 in ordered
creation, 48 with a known rewrite, 18 with tied chronology and 48 with an unknown
rewrite. The repaired engine and vector route each have zero changes in 192
pairs. For the repaired engine, this confirms the declared mechanism check
across the four conditions; it does not make an indeterminate history known.

All three routes return a record for every tied or unknown-rewrite request:
192/192 indeterminate observations per route. The protocol withholds temporal
correctness scores for those conditions because their visible history cannot
establish a current fact. These returns are not abstention. In particular,
creation-time fallback cannot recover an unknown content-revision time.

The audit finds no execution errors, scope failures, budget failures, unexpected
record mutations or input/candidate mismatches. Context never exceeds 78 bytes.
Retrieval changes only the selected engine record's access count and activity
time; content, vectors, source metadata and known or unknown revision times stay
intact. Read-only checks of all 72 closed SQLite artifacts match their final
captured states, with 18 rows and vectors each, valid schema, integrity and
foreign keys, and no remaining WAL frames.

Instrumented timing includes actual query embedding, candidate retrieval and
context serialization. Fixture construction is excluded. Each row below contains
192 requests; the range covers the three repetition-level p95 values.

| Route | Store | Median ms | Overall p95 ms | Repetition p95 range ms |
| --- | --- | --- | --- | --- |
| 0.6.1 engine | Memory | 1.00 | 1.42 | 1.35-1.51 |
| 0.6.1 engine | SQLite | 7.53 | 12.51 | 11.02-13.00 |
| 0.6.2 engine | Memory | 1.02 | 1.54 | 1.40-1.58 |
| 0.6.2 engine | SQLite | 7.56 | 13.11 | 11.00-13.19 |
| Vector retrieval | Memory | 0.90 | 1.37 | 1.25-1.41 |
| Vector retrieval | SQLite | 6.91 | 14.33 | 12.10-14.33 |

The largest request is 418.36 ms on the 0.6.2 SQLite route and remains in the
results. This eighteen-record, instrumented comparison is not evidence for the
10,000-fact latency target, host-command acknowledgment or production speedup.
The frozen functional candidate's previous timing failures remain unresolved.

The first read-only audit made a numeric mistake: it compared production float32
similarities with double-precision arithmetic at a tolerance of 1e-12. The
corrected audit recomputes all 10,368 values in float32 and matches them exactly.
The largest double-precision difference was about 8.14e-8. Both audits and the
initial discrepancy report are retained; the measured run was not repeated and
none of its results changed. Earlier fixture-preparation issues were resolved
before the freeze and produced no cohort results.

The [receipt](legacy-chronology-v1-receipt.json) records exact revisions, artifact
hashes, scores and the audit correction. The 133-member private archive has
SHA-256 `c2608dbe52d593ed087fe6fab0a78e4f65b0949169b3297cd0b8e44faaa70a95`.
Four focused harness tests, strict typing and formatting pass. The frozen f5
product, original wheel artifacts and fourteen watched live paths are unchanged.

This bounded comparison is complete. No hosted or extraction model was used.
Native acceptance, the supported platform matrix, release qualification and
actual pilot use remain open. Both pilots remain at zero days, and nothing was
published or activated.
