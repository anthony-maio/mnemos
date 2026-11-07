# Fresh accepted-state retrieval result

The first retrieval-only evaluation on the v7 authored corpus completed all 5,220
planned observations. All three strategies meet the original 90% answer
Recall@5 threshold by point estimate. All fail the original 90% correct-abstention
threshold, scoring 22.73%. No overall quality or release acceptance follows.

## Ordinary questions

| Strategy | Any-answer hits | Clean hits | Mean fractional Recall@5 | Correct empty abstentions |
| --- | ---: | ---: | ---: | ---: |
| Vector | 600/600 | 300/600 | 100.0% | 150/660 |
| Lexical | 561/600 | 300/600 | 93.5% | 150/660 |
| Hybrid | 600/600 | 300/600 | 100.0% | 150/660 |

Answerable denominators represent 200 unique questions repeated three times;
no-answer denominators represent 220 repeated three times. A clean hit includes
an expected fact without extra IDs. Returned IDs outside the expected answer set
are nonrelevant selections, not proof that the stored statements themselves are
false. These are supplied synthetic labels; no human semantic acceptance is
claimed.

Across all ordinary and paired observations, the audit found zero operational
failures, scope leaks or stale returns. Selected IDs were stable in all 1,740
query/strategy groups across repetitions. Stability does not turn repetitions
into new independent questions.

## Candidate-pool and host strata

Whole-snapshot size and eligible-candidate size differ when scope excludes facts.
The following tables retain ordinary questions only; all scenario-tag, host and
pool strata, including both pair variants, are in the receipt. Tags overlap.
A zero denominator means that stratum contains no question of that kind.

| Facts in snapshot | Strategy | Answer hits | Clean answer hits | Empty abstentions |
| --- | --- | ---: | ---: | ---: |
| 1 | Vector | 300/300 | 300/300 | 150/330 |
| 1 | Lexical | 300/300 | 300/300 | 150/330 |
| 1 | Hybrid | 300/300 | 300/300 | 150/330 |
| 6 | Vector | 300/300 | 0/300 | 0/330 |
| 6 | Lexical | 261/300 | 0/300 | 0/330 |
| 6 | Hybrid | 300/300 | 0/300 | 0/330 |

| Eligible facts | Strategy | Answer hits | Clean answer hits | Empty abstentions |
| --- | --- | ---: | ---: | ---: |
| 0 | Vector | 0/0 | 0/0 | 150/150 |
| 0 | Lexical | 0/0 | 0/0 | 150/150 |
| 0 | Hybrid | 0/0 | 0/0 | 150/150 |
| 1 | Vector | 300/300 | 300/300 | 0/180 |
| 1 | Lexical | 300/300 | 300/300 | 0/180 |
| 1 | Hybrid | 300/300 | 300/300 | 0/180 |
| 6 | Vector | 300/300 | 0/300 | 0/330 |
| 6 | Lexical | 261/300 | 0/300 | 0/330 |
| 6 | Hybrid | 300/300 | 0/300 | 0/330 |

For no-answer observations with at least one eligible fact, correct empty
abstentions were 0/510 for every strategy: 170 unique questions, each repeated
three times. Every correct abstention had zero eligible facts. This demonstrates
scope exclusion without a semantic support decision. Every clean answer hit
came from a one-fact pool; none of the six-fact pools produced a clean hit.

| Source host label | Strategy | Answer hits | Clean answer hits | Empty abstentions |
| --- | --- | ---: | ---: | ---: |
| claude-code | Vector | 201/201 | 102/201 | 51/222 |
| claude-code | Lexical | 162/201 | 102/201 | 51/222 |
| claude-code | Hybrid | 201/201 | 102/201 | 51/222 |
| codex | Vector | 201/201 | 99/201 | 51/222 |
| codex | Lexical | 201/201 | 99/201 | 51/222 |
| codex | Hybrid | 201/201 | 99/201 | 51/222 |
| hermes | Vector | 198/198 | 99/198 | 48/216 |
| hermes | Lexical | 198/198 | 99/198 | 48/216 |
| hermes | Hybrid | 198/198 | 99/198 | 48/216 |

Host labels describe synthetic source sessions. No native host ran in this
comparison. Small pools and repeated question families limit generalization;
the preparation record already identified those limits before inference.

## State-pair wording

| Strategy | Semantic-wording hits | State-wording hits | Semantic extras | State-wording extras |
| --- | ---: | ---: | ---: | ---: |
| Vector | 240/240 | 240/240 | 960 | 960 |
| Lexical | 240/240 | 240/240 | 960 | 960 |
| Hybrid | 240/240 | 240/240 | 960 | 960 |

Each variant has eighty unique questions repeated three times. Both original
wordings were preserved. The audit found different vectors in all 240 recorded
wording pairs. All baselines ignore declared state labels, so these
results cannot isolate a state-module effect. Graph benefit and the full-module
treatment remain separate work.

## Uncertainty and context

Wilson intervals use unique questions within each repetition. The intervals
were identical in repetitions 1, 2 and 3; each is shown once below, with every
repetition retained in the receipt. Questions within a session can remain
dependent, and these intervals do not establish human review or semantic
independence.

| Strategy | Measure | Unique questions per repetition | Descriptive 95% interval |
| --- | --- | ---: | --- |
| Vector | Answer hit | 200 | 98.12% to 100.00% |
| Vector | Empty abstention | 220 | 17.69% to 28.71% |
| Lexical | Answer hit | 200 | 89.20% to 96.16% |
| Lexical | Empty abstention | 220 | 17.69% to 28.71% |
| Hybrid | Answer hit | 200 | 98.12% to 100.00% |
| Hybrid | Empty abstention | 220 | 17.69% to 28.71% |

The paired bootstrap resamples whole sessions with 10,000 draws and seed
20261107, retaining all questions and repetitions within each sampled session.
The audit independently recomputed it using session sums and sample counts.
Differences below are percentage points. Fractional-recall and both state-variant
intervals are also preserved in the receipt.

| Measure | Paired comparison | Difference | 95% interval | Session clusters |
| --- | --- | ---: | --- | ---: |
| Answer hit | Lexical minus vector | -6.50 | -10.00 to -3.50 | 200 |
| Answer hit | Hybrid minus vector | 0.00 | 0.00 to 0.00 | 200 |
| Empty abstention | Lexical minus vector | 0.00 | 0.00 to 0.00 | 200 |
| Empty abstention | Hybrid minus vector | 0.00 | 0.00 to 0.00 | 200 |

| Strategy | Nonrelevant selections, all observations | Total context bytes | Expected-ID matches per 1,000 context bytes |
| --- | ---: | ---: | ---: |
| Vector | 4950 | 766,839 | 1.408 |
| Lexical | 4989 | 763,266 | 1.364 |
| Hybrid | 4950 | 766,839 | 1.408 |

Context uses complete qualified facts and a 1,200-byte conservative token bound.
The density above is an expected-ID/byte measure, not supported facts per actual
host token. It does not include duplicated fields in full transport responses.
No answer-support model or host tokenizer ran.

## Timing and execution evidence

| Strategy | Median retrieval milliseconds | Interpolated p95 milliseconds |
| --- | ---: | ---: |
| Vector | 38.04 | 100.01 |
| Lexical | 37.84 | 98.63 |
| Hybrid | 40.40 | 101.91 |

These per-arm calls use small declared snapshots and exclude shared query
embedding and comparison preparation. The complete run took 4,068.64 seconds,
including conversion, model work, repeated source-binding checks, database
creation/copies and reporting. Queued disk writes were observed on the run's
storage volume; that observation does not isolate the cause of total runtime. Neither
timing establishes the 10,000-fact or native-hook latency requirements.

The run used FastEmbed 0.9.1 with offline `BAAI/bge-small-en-v1.5`, 384 dimensions
and seventeen unchanged cache assets. It made 700 document and 1,740 query
embedding calls. Vector and hybrid shared each query vector; lexical used none.
All arms shared accepted state, scope, top-k 5 and context budget. All cognition
modules were disabled, and the relevance floor remained 0.2.

The audit checked all 152 bound files, all 2,440 role vectors, all 5,220 raw
results and exact source-to-corpus fields. Two hundred source databases retained
700 facts, 700 source events and 700 evidence links, with zero jobs and access
writes. Integrity and foreign-key checks passed. A supplemental read-only audit
matched every stored document vector and every float32 SQLite vector-index entry
to its recorded provider output, without changing database bytes. Both audits
made zero model calls.

The main private archive has 765 verified members, SHA-256
`7a27ddf03bd095a52341bb88eba967710b9e88d1a968be23c42b4d750770e6ee`.
The stored-vector audit has a separate two-member archive, SHA-256
`345b03594547a4e935332d66603b6cce36b1dd81d25940c2ee5b8a8872ebca77`.
The evaluation and initial audit both exited successfully; the evaluation
process is stopped.

Executed source: `291c6fcc8fb206400a902e43579691242dd110f8`.
Original source SHA-256: `b129004eacc1364dce6b8ab069a0b2b9cdf5604543aae003b8ea9f1e42a74fec`.
Canonical converted corpus SHA-256: `d930e6c2e804074e98310b8d67efdc361c1f1db214d912309cce04e71d04f716`.

## Limits and remaining work

The adapter queried terminal declared gold states after each session's final
event. It did not run capture, extraction, correction, forgetting or replay.
Gold IDs begin at declared version 1. Excluding forgotten gold facts is not a
new forgetting-workflow result. The twenty development sessions were excluded
from model inputs.

This corpus is now evaluated input. Subsequent reuse must retain that history;
the word "fresh" cannot be reapplied after tuning on its results. Synthetic
families, short sessions, concentrated state pairs and absent human acceptance
remain material limits. Supported-fact precision and critical retention require
their separate extraction evaluation.

Answer-support quality, the full-module comparison, remaining native/scale
checks, final delivery alignment and the actual two-week pilot remain open.
Production source and live configuration are unchanged. The pilot is inactive
at zero days. No release or deployment occurred.

- [Frozen protocol](../../plans/2026-11-07-fresh-accepted-retrieval-v1.md)
- [Machine-readable receipt](fresh-accepted-v1-receipt.json)
- [Adapter validation](accepted-adapter-v1.md)
- [Separate support diagnostic](support-intersection-v1.md)
