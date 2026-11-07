# Accepted-state corpus runner validation

The retrieval-only corpus runner now binds each query to its declared state at
that time and produces paired statistics without treating repetitions as new
independent questions. Its real-embedding development screen completed all 36
planned observations. Every strategy found the expected fact at both answerable
points, with the correct version. All strategies still failed to return an empty
result for the two no-answer questions. This verifies the runner while preserving
the existing negative abstention result.

## What the runner covers

`AcceptedCorpus` validates all declared snapshots, query times, scopes, source
links, budgets and labels before constructing a provider. A query cannot use a
future state or an older snapshot when a newer declared state was already available.
Changed fact content requires a newer version. Labels remain outside stored rows
and embedding inputs. The runner creates a new owned output directory, retains
failed attempts, rejects reuse and closes stores on cancellation.

The scorer requires the complete case/query/repetition/strategy matrix. It checks
shared accepted state and document vectors across queries and repetitions, plus
shared query vectors within vector/hybrid pairs. It recomputes scores from raw
returned IDs, versions, content and scopes. Provider failures stay in the planned
denominator; an empty failed response cannot earn abstention credit.

Answer hits, clean hits and fractional recall are separate measures. Paired
intervals cover both hit/abstention differences and fractional-recall differences.
Whole sessions are resampled together, preserving all their questions and
repetitions. Wilson intervals use each repetition's unique question count; they
remain descriptive because questions within a session can be dependent.

Held-out declarations must meet the original structural minimums of 200 distinct
sessions, 100 declared critical facts, 100 answerable and 100 no-answer questions,
50 matched state pairs, three repetitions, all three host labels and multiple
queried projects. Counts and labels do not prove human acceptance, native-host
execution, semantic correctness or data freshness. State pairs are baseline
controls here: all three strategies ignore the declared state condition. The
full-module treatment and ablation remain separate work.

## Frozen development screen

The fixture declares one session with three snapshots: thirty-day production-log
retention, a version-2 correction to forty days, and removal of that fact. The
snapshots contain six, six and five facts, with corresponding retained sources
and evidence links. Questions ask for production retention before the correction,
after the correction and after removal, plus an unknown region after correction.
These are authored states, not measured outputs of correction or forgetting.

Used local FastEmbed 0.9.1 with `BAAI/bge-small-en-v1.5`, 384 dimensions, offline
loading and 17 verified cache-asset files. The run made 17 document and 12 query
embedding calls across three repetitions. No extraction or support model ran.

| Strategy | Answer hits | Correct empty abstentions | Stale returns | Scope leaks | Operational failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector | 6/6 | 0/6 | 0 | 0 | 0 |
| Lexical | 6/6 | 0/6 | 0 | 0 | 0 |
| Hybrid | 6/6 | 0/6 | 0 | 0 | 0 |

Each denominator represents two unique questions repeated three times. Each
strategy also returned 51 nonrelevant fact selections across its twelve query
observations, so its clean-hit count is zero. These are expected-ID relevance
judgments; they do not establish that the stored facts themselves are false.
Stored conflict qualifications remain intact and do not satisfy the frozen
empty-result abstention criterion.

The audit verified the expected thirty-day and forty-day versions at their query
times and absence of the removed fact in the final declared snapshot's results.
All twelve query/strategy groups were stable across repetitions. The three source
databases retained their declared contents, with zero jobs and access writes,
valid SQLite integrity and no foreign-key errors. This establishes snapshot
selection and isolation, not a new forgetting or resurrection workflow result.

Wilson intervals use two questions per repetition, not six pooled repeats. Paired
comparisons retain six observations in one session cluster. All measured strategy
differences were zero, making the one-cluster intervals degenerate. This small
screen checks the statistics machinery and provides no population uncertainty
claim. Separate tests cover unequal cluster sizes and partial recall with an
unchanged any-answer hit rate.

The run took 20.47 seconds including fixture preparation, provider loading,
binding checks, snapshot copies, retrieval and reporting. This is not a native-host
or 10,000-fact latency measurement. The independent audit made no model calls.

## Verification and remaining work

Ninety focused and adjacent checks passed, including 38 new corpus-runner cases;
twelve documentation checks also passed. Black and strict mypy passed for the new
runner. Initial validation caught a wrong foreign fact ID in the test fixture,
Python 3.10-incompatible tuple subscripting and missing type declarations; these
were corrected before the frozen model run. The completed model process exited.

Implementation and executed source:
`bd2c3e2c1884e63795d73d05a25a2898308b4930`.
Frozen generated fixture SHA-256:
`ff95cdda87ec6bcefdfb592685a9fa3a0d31914e5d76c119e275c889db89234a`.

The reusable Python API accepts explicit accepted-state corpora. Conversion of
the existing fresh corpus into reviewed query-time states is still unimplemented;
the original ingestion-based comparator is unchanged. The fresh corpus remains
unopened and unexecuted. Applicable support quality, the full module comparison,
native/scale checks and the actual two-week pilot remain open. Production package
sources, dependencies and live configuration are unchanged. The pilot remains
inactive at zero days.

- [Prospective protocol](../../plans/2026-11-07-accepted-corpus-development-v1.md)
- [Machine-readable receipt](accepted-corpus-v1-receipt.json)
- [Earlier accepted-state baseline](accepted-retrieval-v1.md)
- [Separate answer-support experiment](extractive-answer-support-v1-v2.md)
