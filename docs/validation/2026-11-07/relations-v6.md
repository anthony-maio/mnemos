# Referent relation and consistency diagnostic v6

The stronger diagnostic rejects the apparent reliability of the earlier 19/20
decision result. On 72 matched development views, the unchanged comparator
produced 49 correct operations, five wrong corrections and eleven wrong
additions. Only 110/198 intermediate relations were correct. Three of twelve
seed scenarios stayed consistent across all six views, and only two were
correct at both layers in every view. Keep this comparator experimental.

The [protocol](../../plans/2026-11-07-referent-relations-v6.md), corpus and
source were frozen at `db7d182f51079b012e3c18fc8dadfdf9939e3938`. Twelve
authored seed scenarios in five clusters each have six views that vary target
IDs, order and unrelated distractors. Each view preserves the source and
classification. Logical target identities preserve the target text, version and
expected relation. All relation labels were fixed before inference.

These 72 views are dependent development observations, not 72 independent
scenarios or a fresh held-out evaluation. The earlier v5 scores and its post-run
diagnosis remain unchanged. This run changes the evaluation cases and scoring,
while retaining the same comparator prompt and model configuration.

| Measure | Result |
| --- | ---: |
| Correct operation and target | 49/72 |
| Correct unresolved controls | 22/30 |
| Wrong staged corrections | 5 |
| Wrong staged additions | 11 |
| Operational failures | 0 |
| Correct intermediate relations | 110/198 |
| Views with every relation correct | 32/72 |
| Views correct at both layers | 32/72 |
| Seeds with all six operations correct | 3/12 |
| Seeds correct at both layers in all six views | 2/12 |
| Seeds with stable operations and relations | 3/12 |

The five wrong corrections included three public-repository requirements that
selected a private-repository rule, one correction with a missing environment,
and one with a conflicting environment. The eleven wrong additions comprised
five requirement reversals that should have corrected the existing rule, two
requirements missing an environment, and four requirements with an unidentified
count. The last six should have remained under review. These are staged
reconciliation results; the run wrote no facts.

The vague-count correction illustrates why both scores are needed. Its review
operation was correct in all six views, but its intermediate relations were
wrong in all six. Consistency also does not imply correctness: that scenario
was one of the three stable groups. Only the explicitly qualified schedule
and the independent report requirement were correct at both layers in all views.

| Matched presentation change | Pairs | Operation changed | Shared-target relation changed |
| --- | ---: | ---: | ---: |
| Rename IDs only | 36 | 7 | 9 |
| Reverse target order only | 24 | 6 | 7 |
| Add unrelated distractors | 24 | 9 | 9 |

These pair counts reuse observations and are not independent samples. Each
view ran once, so the study reports observed instability without separating
all possible runtime variability from presentation sensitivity. The source,
labels and prompt were fixed throughout.

All 72 logical calls returned one response with a normal stop reason and no
thinking text. All 189 selected value spans were ordered and source-bound;
81 value fields were null. Valid span locations did not establish correct
relations. The independent audit reconstructed every operation and relation
score, all twelve consistency groups and all matched pair counts. It replayed
all 72 results through runtime validation without model calls and checked
138 source hashes, all input snapshots and 198 comparison records.

The run used Qwen3:14b Q4_K_M, Ollama 0.40.9, generate mode, thinking false,
temperature zero, context 8192 and one request slot. It retained the v5 limits:
4096 output tokens, 3584 counted input tokens, a 512-token template reserve,
a 90-second provider deadline and a 120-second observation deadline. Maximum
native prompt-plus-schema, reported prompt and generated counts were 2,866,
2,002 and 231. No input or output limit was exhausted.

Median observation latency was 2,573 ms; nearest-rank p95 was 3,351 ms. The
provider reported 110,859 prompt tokens, 59,729 cached prompt tokens and 7,671
generated tokens. Those are reconciliation costs, not retrieval latency or a
context-efficiency improvement.

All 183 focused tests, including 24 new scoring and runner checks, and 12
documentation checks passed. The integration check specifically preserves a
correct review operation while rejecting the wrong intermediate relation.
Preflight mechanically exercised all 72 labeled views and verified that labels
and logical aliases were absent from provider payloads. Black and strict mypy
passed. The earlier full-suite result retains its own source binding.

The [receipt](relations-v6-receipt.json) binds 175 verified private archive
members under SHA-256
`5afd4ea5b2393a28df3d5b9232d96e481cc5448d063d9392737d3f29c463152c`.
The model unloaded, all listed owned processes were absent and both endpoints
closed. The original tree-stop warning is retained, with no extra kill or
inference rerun.

A [Qwen3.5 candidate](qwen35-local-setup.md) has separately passed empty-load
setup. It supplied no response in this diagnostic. A comparison must bind its
different package defaults, native tokenizer and model digest before inference.
The present result neither promotes a model nor closes extraction, fresh quality,
persisted-outcome or daily-use requirements. Live settings, installed pilot
artifacts, production defaults and publication state remain unchanged.
