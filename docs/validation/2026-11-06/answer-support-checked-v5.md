# Three-check answer-support comparison v5

The three-check candidate did not improve answer support. Both treatments
missed the 90% abstention target on the reused development set. The candidate
added unsupported facts, four provider-contract failures and latency. Keep
`checked_v4` experimental and outside production retrieval.

The run used source `570b93e3270a2fbdb403d81034679201d3272b0e`, the committed
[protocol](../../plans/2026-11-06-answer-support-checked-v5.md), Qwen3 14B and
the same 44 development questions used in v4. It completed 352 observations:
176 exact retrieval inputs, each evaluated by both treatments. There were 27
answerable questions and 17 without an answer, across four retrieval arms.
These observations are not 352 independent samples. Extraction was disabled.

## Paired results

An answerable hit means at least one expected fact was selected. A clean hit
requires the exact expected set with no unsupported additions. Abstention
requires a successful assessment and an empty result for a question with no
answer; failed requests receive no credit. Unsupported counts are facts,
not questions. Timing is the measured support operation in this one run.

| Treatment | Arm | Hits / 27 | Clean / 27 | Abstentions / 17 | Unsupported facts | Failures | Median ms | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint | Vector | 27 | 26 | 14 | 6 | 0 | 2,332 | 2,663 |
| Three-check | Vector | 26 | 17 | 13 | 14 | 1 | 5,817 | 6,761 |
| Joint | Lexical | 27 | 26 | 14 | 5 | 0 | 2,343 | 2,620 |
| Three-check | Lexical | 26 | 19 | 14 | 11 | 1 | 5,823 | 6,766 |
| Joint | Hybrid | 27 | 26 | 14 | 6 | 0 | 2,321 | 2,628 |
| Three-check | Hybrid | 27 | 19 | 13 | 12 | 1 | 5,819 | 6,684 |
| Joint | Full | 27 | 26 | 13 | 7 | 0 | 2,309 | 2,623 |
| Three-check | Full | 26 | 19 | 13 | 12 | 1 | 5,791 | 6,669 |

The hybrid result separates by dataset as follows. The receipt contains every
arm's per-dataset results, request counts and provider token accounting.

| Dataset | Treatment | Hits | Clean hits | Abstentions | Unsupported facts | Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original development | Joint | 10/10 | 10/10 | 10/10 | 0 | 0 |
| Original development | Three-check | 10/10 | 8/10 | 9/10 | 2 | 1 |
| Temporal development | Joint | 17/17 | 16/17 | 4/7 | 6 | 0 |
| Temporal development | Three-check | 17/17 | 11/17 | 4/7 | 10 | 0 |

In the hybrid arm, both treatments made 78 candidate requests. Joint used
39,321 provider-reported prompt tokens and 4,733 output tokens; three-check
used 71,361 and 12,398. Reported cached prompt counts were 21,583 and 37,435
respectively. These counters are provider accounting, not native-host context
measurements. Concurrent machine activity limits latency comparisons.

The three-check candidate still selected incompatible retention periods,
projected undated current facts into the past, and projected a dated current
backend into the future. It also added a scheduled transition to a current
answer, accepted an implemented-transition claim whose source described an
unapproved proposal, and mixed environment or attribute qualifiers. In each
arm, the quotation/owner question failed because the selected option did not
have the highest supplied probability. Those four failures remained empty
results and received no abstention credit. No threshold or label changed.

## Verification and evidence

All 130 bound source files, both datasets, the protocol, 176 input pairs and
22 accepted-store snapshots verified. Every store passed integrity and
foreign-key checks. All 620 requests have exactly one raw response. The audit
reconstructed all 348 assessed selections and checked exact fact versions and
evidence bindings, recomputed every recorded grade, and matched the original
summary. The four remaining observations are the contract failures above.

The first audit omitted the safety preamble when checking recorded instruction
prefixes. A corrected audit passed, but its prompt reconstruction added that
preamble a second time to string instructions. The final audit requires an
exact serialization roundtrip and measures a maximum of 4,473 bytes, matching
the offline preflight and below the frozen 5,120-byte cap. The prior 4,848-byte
audit estimate was an overcount. All other analysis fields and the failure
audit are identical. No inference or observation was repeated or changed.

The [original receipt](answer-support-checked-v5-receipt.json) retains the
initial audit estimate and binds 222 verified private archive members:
`de27069cbd6c6f0b145b68e5414fd2c2f4bec3010f48294ecbf8e1f54b86759e`.
The [audit correction](answer-support-checked-v5-audit-correction.json) binds
the final prompt measurement and six verified supplemental archive members:
`ad673bdf9fc8ceeaf72bf22545c7efe600fddc54ca64a524a4190120668110f3`.
Raw requests, replies, stores, helpers and failed audits remain private.

The model unloaded, the owned runtime stopped, port 11437 closed and all
recorded owned processes were absent. Later commits during the run changed
only pilot documentation; every study-bound source hash remained unchanged.

This is a negative development result, not fresh evaluation or a reliability
estimate. It does not revise earlier results, validate extraction, supply a
pilot day, change defaults or publish a release. The remaining work includes
selecting a better-supported candidate before fresh evaluation, separate
extraction and context-efficiency measurement, participating-host validation,
and the actual private pilot.
