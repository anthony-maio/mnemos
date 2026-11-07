# Contrastive answer-support comparison v6

The contrastive candidate improved the answerable results but still failed the
90% abstention target. In the hybrid arm it returned exactly the expected facts
for 27/27 answerable questions, then answered five of the 17 questions without
an answer. The paired joint baseline answered four. Keep `contrast_v5`
experimental and outside production retrieval.

The [protocol](../../plans/2026-11-07-answer-support-contrast-v6.md) was bound to
source `5c22eab000fb124222f8f9702f43b9a965dffce4`. Qwen3 14B evaluated 176 frozen
retrieval inputs under two treatments, producing 352 observations on 44 reused
development questions: 27 answerable and 17 without an answer. Four retrieval
arms shared accepted-fact snapshots, embeddings, scope and retrieval budgets.
Extraction was disabled. These are not 352 independent samples.

The candidate changed wording, shared comparison context and batching together.
It did not isolate the effect of any one change. The reference date stayed
November 6, 2026, with the original labels and 0.8 probability threshold.

An answerable hit requires an expected fact. A clean hit requires the exact
expected set. Correct abstention requires a successful assessment and no
selected facts for a question with no answer. Unsupported counts are facts.
Timing covers the support operation; p95 uses nearest rank.

| Treatment | Arm | Hits / 27 | Clean / 27 | Abstentions / 17 | Unsupported facts | Failures | Median ms | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint | Vector | 27 | 26 | 13 | 7 | 0 | 2,462 | 2,662 |
| Contrastive | Vector | 27 | 27 | 12 | 5 | 0 | 2,762 | 2,937 |
| Joint | Lexical | 27 | 26 | 13 | 6 | 0 | 2,443 | 2,630 |
| Contrastive | Lexical | 27 | 27 | 12 | 5 | 0 | 2,750 | 3,191 |
| Joint | Hybrid | 27 | 26 | 13 | 7 | 0 | 2,401 | 2,642 |
| Contrastive | Hybrid | 27 | 27 | 12 | 5 | 0 | 2,736 | 3,047 |
| Joint | Full | 27 | 26 | 13 | 7 | 0 | 2,461 | 2,627 |
| Contrastive | Full | 27 | 27 | 12 | 5 | 0 | 2,736 | 3,223 |

Hybrid results by dataset:

| Dataset | Treatment | Hits | Clean hits | Abstentions | Unsupported facts |
| --- | --- | ---: | ---: | ---: | ---: |
| Original development | Joint | 10/10 | 10/10 | 9/10 | 1 |
| Original development | Contrastive | 10/10 | 10/10 | 7/10 | 3 |
| Temporal development | Joint | 17/17 | 16/17 | 4/7 | 6 |
| Temporal development | Contrastive | 17/17 | 17/17 | 5/7 | 2 |

The hybrid candidate made 44 requests versus 78 for joint. Provider-reported
prompt counts fell from 39,321 to 32,287, while output counts rose from 4,758
to 5,630. Reported cached prompt counts were 25,663 and 19,700. Median support
latency rose about 14%. These counters describe provider accounting, not the
returned memory context. Concurrent machine activity limits timing conclusions.

The candidate still supplied a queue technology for a missing cloud region,
a prohibition for an unknown approver, and a verification requirement for a
missing record identifier. It selected one side of an unresolved retention
conflict and projected an undated handbook location into 2020. It stopped
adding the wrong attribute to the backup answer and stopped accepting the
quoted future transition. Those gains did not resolve the abstention problem.

The earlier v5 run recorded 14/17 hybrid abstentions for the same joint protocol;
this run recorded 13/17. Both records remain intact. Temperature zero and one
repetition do not establish repeatability or explain that difference.

The independent audit verified all 130 bound source files, both datasets, the
protocol, all 176 pairs and 22 accepted-store snapshot hashes. Every store passed
integrity and foreign-key checks. All 486 requests have one raw response, and
all 620 choice distributions satisfy the provider contract. All 352 selections,
fact versions, evidence bindings, grades and summary totals were reconstructed.
The largest exact rendered prompt was 4,861 bytes, matching preflight and below
the frozen 5,120-byte cap. No observation or inference was repeated.

The [receipt](answer-support-contrast-v6-receipt.json) binds 215 verified private
archive members, including original stores, requests, responses and runtime logs:
`62db875c5cabf0c5c3aa36fb97109944661e08967077e89e79d51bea1b8d6dc9`.
The model unloaded, the owned runtime stopped and its port closed. A later
commit added the separate context measurement outside the 130 bound files;
every comparison source hash remained unchanged.

This is a negative development decision. It supplies no fresh evaluation,
extraction result, pilot day, installed policy, default change or publication.
The [separate context measurement](context-density-v1.md) reports memory size
and recall without changing these original observations.
