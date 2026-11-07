# Target identity comparison v2

The identity comparator does not qualify for a full-pipeline regression. It
avoided the direct prompt's wrong-target corrections, but introduced six quote
validation failures, missed valid targets among distractors and treated a
requirement reversal as new information. Its exact score fell to 45/60, below
the legacy arm's 48/60. Keep it experimental and retain the existing default.

The [protocol](../../plans/2026-11-07-reconciliation-identity-v2.md) and source
were frozen at `ec3e5f19b38a860ac36211a5c52b3a934bc61228`. The study reused
the same 20 preclassified development cases in six clusters, three repetitions
per arm. All three arms included the separately tested
[duplicate-identity repair](duplicate-identity-v1.md). Their original events,
candidates, target snapshots and expected outcomes were identical. This is a
reconciliation comparison, not extraction quality or fresh evaluation.

The new comparator asks separately about subject, attribute and identifying
qualifiers, with literal quotes from the candidate and each target. Code permits
a correction only when exactly one target matches every dimension and all
others are ruled out. A second plausible target requires review. The source
role and kind still determine correction authority. Other kinds use the legacy
relationship path. Empty targets and exact duplicate user facts have mechanical
outcomes. This changes decomposition, instructions and the decision rule
together, so the experiment does not isolate one causal contributor.

| Measure | Legacy/full | Direct/full | Identity |
| --- | ---: | ---: | ---: |
| Exact operation and target observations | 48/60 | 51/60 | 45/60 |
| Cases correct in all three repetitions | 16/20 | 17/20 | 15/20 |
| Matching correction trials recovered | 0/12 | 12/12 | 6/12 |
| Correct unresolved controls | 27/27 | 18/27 | 21/27 |
| Wrong support or correction target choices | 0 | 9 | 0 |
| Operational failures | 0 | 0 | 6 |
| Median observation latency | 1,795 ms | 1,411 ms | 2,477 ms |
| p95 latency, nearest rank | 2,493 ms | 1,829 ms | 5,424 ms |
| Logical calls and recorded responses | 60 | 60 | 51 |

The absence of wrong-target corrections is not sufficient for acceptance.
Identity recovered the original single-target correction and the numeric
correction, but missed the target-among-distractors and explicitly qualified
cases. In the latter, it correctly identified the production subject and then
called the changed day a different qualifier, which blocked the correction.
It also classified the requirement reversal as a new fact in all three trials,
leaving the observed prior rule uncorrected. Those three wrong additions are
not counted by the narrower wrong-target-choice metric.

Two cases failed response validation in all repetitions. For the ambiguous
numeric setting, the model quoted `count` from targets that instead described
workers or fixtures; that word was not present. It also compared changed
numbers as subjects. For the ambiguous environment, it called qualifiers
different while providing a target qualifier and null candidate quote. The
contract required an unspecified comparison for missing wording. Both failures
were retained as failures and received no abstention credit. Literal evidence
guards prevented those responses from becoming decisions, but did not turn
them into successful model judgments.

The duplicate-identity repair worked in the new real-model run: direct/full
returned unresolved outcomes for all three duplicate-support trials that were
disagreements in v1. Its three ambiguous or mismatched target cases still
produced nine wrong corrections. The earlier v1 scores remain bound to their
original source and were not revised.

| Provider-reported tokens | Legacy/full | Direct/full | Identity |
| --- | ---: | ---: | ---: |
| Prompt | 39,741 | 44,241 | 30,750 |
| Cached prompt | 31,879 | 35,430 | 25,262 |
| Output | 5,140 | 3,394 | 8,062 |

All 171 logical calls produced one response with a normal stop reason. The
maximum native prompt-plus-schema count was 1,669, below the 5,632 allowance;
the largest reported prompt and output counts were 801 and 299. The local
Qwen3:14b Q4_K_M model used thinking disabled, temperature zero, context 8192
and output limit 2048. Nine identity observations needed no provider call.
Fewer requests did not produce lower observed latency or output-token use.

The independent audit rebuilt all 180 grades, including the six failures,
from preserved inputs and responses. It verified 134 source hashes, all 20
paired input groups, exact preflight prompts and schemas, 51 target comparisons
and 234 literal quoted strings. A quote's presence does not establish its
semantic interpretation. All 132 focused tests passed, including 24 new
identity checks; Black and strict mypy passed. The full-suite result remains
bound to its earlier source.

The [receipt](identity-v2-receipt.json) binds 164 verified private archive
members under SHA-256
`4e75c94c73a5fc776593880a9c221253ae4fb63bdb3a440b3eebe2f337b6e49a`.
The model unloaded, all listed owned processes were absent and both endpoints
closed. The original tree-stop warning is retained, with no extra kill or
inference rerun. No database writes, retrieval, embeddings, native host, live
configuration, installed pilot artifact or published package were involved.

The next decision must account for both semantic mistakes and quote-contract
failures. Merely making quotes structurally valid would leave wrong attribute,
qualifier and requirement judgments unresolved. Further development, fresh
evaluation, persisted extraction and the actual daily-use pilot remain open.
