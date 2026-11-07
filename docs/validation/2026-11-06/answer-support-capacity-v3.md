# Answer-support capacity comparison v3

The 14B checkpoint improved answer selection and abstention on this small
development set. Both 14B treatments correctly abstained on all ten no-answer
questions in every retrieval arm. Both still returned a historical queue as an
answer about the current queue. Each also timed out once in the hybrid arm.
Answer support remains experimental and outside production retrieval.

The joint 14B treatment is the leading candidate for further development. The
separate treatment returned the same selections on completed observations and
took roughly twice as long. This result supports testing the larger checkpoint;
it does not establish that model size alone caused the difference or that the
behavior generalizes beyond these reused questions.

## Fixed comparison

Source `356791aec754a20602257426ed32c6b9f06f150f` and the
[protocol](../../plans/2026-11-06-answer-support-capacity-v3.md) were committed
before inference. The study used Qwen3 4B and 14B Q4_K_M artifacts with recorded
digests, Ollama 0.40.9, BGE small English v1.5 embeddings, temperature zero,
thinking disabled and an 8,192-token context. The threshold stayed at 0.8 and
the shared support-operation deadline stayed at 15 seconds.

All 320 observations completed: twenty reused development questions across four
retrieval arms, two models and two support treatments. There are ten answerable
and ten no-answer questions, not 320 independent samples. Retrieval ran once per
query and strategy before either model. All four treatments used the same frozen
IDs, facts and original accepted store for each input. Extraction was disabled.

Every retrieval baseline found the expected fact on all ten answerable queries.
None abstained on the ten no-answer queries. Baselines returned 30 unsupported
facts across the twenty queries, except lexical with 28. Candidate coverage was
therefore sufficient on this fixture; selecting only supported answers remained
the problem under test.

## Results

A clean answerable hit returns exactly the expected fact IDs. An ordinary hit
may include unsupported extras. Unsupported facts are counted across all twenty
queries. Failures receive no hit or abstention credit. Latency is median support
time, excluding retrieval and model loading.

| Model / treatment | Retrieval | Hits /10 | Clean hits /10 | Abstentions /10 | Unsupported facts | Failures /20 | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4B joint | vector | 9 | 5 | 5 | 11 | 0 | 1,142 |
| 4B joint | lexical | 9 | 5 | 5 | 11 | 0 | 1,138 |
| 4B joint | hybrid | 9 | 5 | 5 | 11 | 0 | 1,161 |
| 4B joint | full | 9 | 5 | 5 | 11 | 0 | 1,138 |
| 4B separate | vector | 10 | 3 | 2 | 18 | 0 | 1,901 |
| 4B separate | lexical | 10 | 4 | 2 | 17 | 0 | 1,911 |
| 4B separate | hybrid | 10 | 3 | 2 | 18 | 0 | 1,913 |
| 4B separate | full | 10 | 3 | 2 | 18 | 0 | 1,895 |
| 14B joint | vector | 10 | 9 | 10 | 1 | 0 | 2,631 |
| 14B joint | lexical | 10 | 9 | 10 | 1 | 0 | 2,563 |
| 14B joint | hybrid | 9 | 8 | 10 | 1 | 1 | 2,576 |
| 14B joint | full | 10 | 9 | 10 | 1 | 0 | 2,525 |
| 14B separate | vector | 10 | 9 | 10 | 1 | 0 | 5,121 |
| 14B separate | lexical | 10 | 9 | 10 | 1 | 0 | 4,921 |
| 14B separate | hybrid | 9 | 8 | 10 | 1 | 1 | 4,980 |
| 14B separate | full | 10 | 9 | 10 | 1 | 0 | 4,951 |

The 14B treatments meet the 90% answerable-hit and 90% abstention targets on
this development set. Their hybrid clean-hit rate is only 8/10. The remaining
unsupported fact and operational failures remain part of the result; these
development targets do not promote a default or replace fresh evaluation.

## Remaining errors

The temporal case asks which queue Juniper uses now. One retained fact says it
used Redis in 2024; another says PostgreSQL replaced Redis in October 2026.
Both 14B treatments return both facts, assigning the historical fact probability
1.0. Its source supports the historical statement, but that statement does not
answer the current-time question. Raising the threshold would not remove it.
Candidate-at-a-time checks also lack a comparison with the replacement fact.

The 4B joint treatment still loses the explicit negative answer about an
unapproved migration and accepts several wrong attributes, environments and
quoted instructions. The 14B treatments reject those unsupported answers on
this fixture. Their remaining semantic failure is temporal relevance.

Both timeouts occurred on the first hybrid release-path question, one per 14B
treatment. The empty-prompt model load completed before measurement. These
requests still exceeded the fixed operation deadline. They were not retried,
removed or credited as abstentions. Later successful observations do not repair
those failures or establish their cause.

## Integrity and limitations

Independent checks verified all 128 source hashes, eighty paired input groups
and ten reconstructed SQLite snapshot hashes. All ten databases passed integrity
and foreign-key checks. Foreign-project facts were excluded. Gold IDs remained
in grading records and never entered judgment state.

There were 567 judgment requests and 565 retained raw replies. The two missing
replies map to the two timeout observations. Every returned distribution supplied
all choices, summed to one within tolerance, and selected a highest-probability
option. No probability was filled or normalized. The largest rendered prompt
was 3,490 UTF-8 bytes; the largest returned context sequence was 889 tokens and
the largest output was 135 tokens. All retained replies ended with `stop`, and
the server's truncation markers were zero. Token totals cover retained replies
only, not unobserved work on the timed-out requests.

An initial analysis helper inserted empty entries into its response lookup while
aggregating metrics, then incorrectly printed an empty missing-response list.
The corrected helper uses non-mutating lookups and verifies request accounting
before aggregation. The original helper and output are retained. Model calls,
raw observations and every selection metric are unchanged.

Both models reported full GPU residency on an RTX 4080. Other applications used
the device, and the fixed model order was 14B then 4B. Model-block order, prompt
caching and background activity limit latency interpretation. The new context
setting differs from v2, so the paired 4B observations here are the comparison
baseline. This is one repetition, not a performance benchmark.

The model was unloaded, the owned server tree stopped, and the listening port
verified closed. The [receipt](answer-support-capacity-v3-receipt.json) binds the
raw evidence, accepted databases, source files, runtime logs and analysis in a
private archive with 179 verified members. Its SHA-256 is
`fc4a409f77bcbb5527839534f0a237315a023d845fa3c235890225f54dc27b7f`.
The harness passed 49 focused tests, strict mypy and Black before inference.

Next, test temporal applicability and explicit replacement evidence under a new
development protocol, then freeze the candidate before constructing fresh
evaluation data. Do not infer fact validity from its ingestion timestamp. At
least 200 fresh scenarios with three repetitions, separate extraction results,
supported context efficiency, actual pilot use and consistent delivery remain
open. No historical result, default, live host configuration or release changed.
