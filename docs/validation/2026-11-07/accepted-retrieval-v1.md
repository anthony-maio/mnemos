# Retrieval from declared accepted state

The new benchmark path compares vector, lexical and hybrid retrieval without
running extraction. Its development screen completed 63 observations on source
`490317feab4c213ec187cdbbd56f697bbaf4529e`. Every strategy found the expected fact
in all answerable checks, but none met the empty-result abstention criterion
for the unknown-region, conflicting-reviewer or foreign-project questions.

| Strategy | Answerable hits | Correct empty-result abstentions | Nonrelevant returned IDs | Scope leaks | Operational failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector | 12/12 | 0/9 | 81 | 0 | 0 |
| Lexical | 12/12 | 0/9 | 75 | 0 | 0 |
| Hybrid baseline | 12/12 | 0/9 | 81 | 0 | 0 |

These counts repeat four answerable and three no-answer queries three times.
They are seven distinct authored development cases, not 21 independent cases
or fresh acceptance evidence. The conflicting facts retained their unresolved
conflict qualification. That is useful evidence, but it does not satisfy the
predeclared empty-result criterion. No generated answers were evaluated.

The highest conflict-query cosine was 0.89545, while the correct foreign-project
answer in its permitted scope scored 0.85588. Raising one shared cosine floor
cannot both remove that conflict result and retain that correct answer in this
fixture. No threshold, label or production default changed after observing
these results.

## Input and execution checks

The snapshot contains six declared current facts, six retained source events
and six literal evidence links. Four facts are asserted and two are conflicting.
Their identities, versions, scope partitions and timestamps are validated before
document embedding. The builder rejects unsupported facts, missing or foreign
sources, nonliteral excerpts, future timestamps and undeclared metadata fields.
Expected-answer labels stay outside the stored records and embedding inputs.
The declared snapshot is not evidence that extraction or correction produced it;
literal source validation alone does not prove entailment.

All modules and query-state inference were disabled. FastEmbed 0.9.1 used local
BGE-small, 384 dimensions, with the existing cache and offline mode. Six document
calls and 21 query calls were recorded. Vector and hybrid shared each query
vector. Each strategy began with the same committed SQLite snapshot and received
its own access updates. All 21 query/strategy groups returned the same ordered
IDs across repetitions. The original database retained six facts, six sources,
six evidence links, zero jobs and zero access counts. Integrity and foreign keys
passed. The 17 model-cache files retained their hashes.

The independent audit recomputed all 63 grades from the frozen labels and raw
results, verified every embedding input and vector binding, checked context
budgets, source links and versions, and confirmed the identical snapshot hashes.
It made no provider calls. The [receipt](accepted-retrieval-v1-receipt.json) binds
29 verified private archive members. Archive SHA-256:
`28a10c77191746e0b8c81f3e7264bb34de9f19ef5dc3f4bf60b6d1f546b874ea`.
The [prospective protocol](../../plans/2026-11-07-accepted-retrieval-development-v1.md)
retains the exact input and runner hashes.

Median retrieval dispatch times were 38.21 ms for vector, 37.42 ms for lexical
and 40.93 ms for hybrid. Vector and hybrid share a separately measured query
embedding median of 5.65 ms; provider construction took 775.56 ms. Snapshot
cloning and state validation are outside those dispatch measurements. This
six-fact screen does not establish native-host or 10,000-fact latency.

Thirty-six new checks cover state and evidence validation, scope visibility,
shared vectors, isolated access writes, changed-state refusal, invalid vectors,
destination preservation and failure scoring. All 65 focused and adjacent checks
passed, along with formatting and strict types for the new benchmark module.
Production package files did not change; the previous package checkpoint keeps
its original binding.

Answer support and abstention need further development before opening fresh
acceptance data. The fresh corpus, full-module comparison and persisted
extraction evaluation remain separate obligations. The private pilot is still
inactive with zero days. No native host, live memory or configuration was changed.
