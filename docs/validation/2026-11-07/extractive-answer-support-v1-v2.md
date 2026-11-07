# Extractive answer-support development result

The dedicated QA model improved abstention on the existing development inputs,
but it is not ready to select production context. After a source-offset decoder
correction, it found 25 of 27 expected answers and correctly abstained on 16 of
17 no-answer questions in the larger fixture. Four unsupported fact selections
remain. The small seven-query fixture passes. Neither result clears the fresh
evaluation or daily-use pilot gates.

## Results on the same frozen hybrid candidates

These counts use unique questions. Each QA result was identical across three
repetitions. A clean hit finds the expected fact without adding unsupported IDs.
The historical Qwen rows are existing measurements, not new inference runs.

| Larger development fixture, 44 questions | Answer hits | Clean hits | Correct abstentions | Unsupported fact selections |
| --- | ---: | ---: | ---: | ---: |
| Existing Qwen joint judgments | 27/27 | 26/27 | 13/17 | 7 |
| Existing Qwen contrast judgments | 27/27 | 27/27 | 12/17 | 5 |
| QA v1, original strict source boundary | 22/27 | 20/27 | 16/17 | 4 |
| QA v2, corrected boundary, post-hoc | 25/27 | 23/27 | 16/17 | 4 |

On the separate seven-query fixture, both QA decoders produced four clean hits
and three correct abstentions, with zero unsupported selections. Across three
repetitions this is 12/12 clean hits and 9/9 abstentions. The larger corrected
screen totals 75/81 answer hits, 69/81 clean hits, 48/51 correct abstentions and
12 unsupported selections. Repetition does not increase the number of independent
semantic cases.

## What changed in the decoder

The model's tokenizer included a preceding newline in the first source word's
offset. The original rule excluded any token not wholly inside the excerpt,
discarding three valid first-word answers. The correction clips only whitespace
outside the source boundary. It cannot import non-whitespace prefix or suffix
text, and selected spans still point to exact characters inside the excerpt.

All 291 original decisions reproduce under the unchanged default. Re-decoding
the same logits with the optional correction recovered three answers, or nine
selections across repetitions, without changing any other selection. No new model
call or threshold change was used. The original decoder, run and scores remain
archived. The corrected result is explicitly post-hoc development evidence.

The model still omits two supported answers: a batch-size question inside an
explicit date interval and an audit-log reviewer. It also selects a scheduled
future bucket for a current-state question, adds a replacement-owner fact to a
historical question, and returns both sides of an unmarked retention conflict.
Those last three queries account for four unsupported fact selections. Literal
source spans do not resolve temporal applicability or conflicts between facts.

## Execution and audit

Used `deepset/deberta-v3-base-squad2` at revision
`eea39c60cc305c2e4a9504f5ff117294bebb42db`, with verified safetensors weights and
local-only loading in a separate CPU environment. The publisher identifies
SQuAD2 training, including unanswerable questions, and a CC BY 4.0 license.
[Model card](https://huggingface.co/deepset/deberta-v3-base-squad2)

The frozen screen used 51 unique questions, 97 eligible query/excerpt pairs per
repetition, three repetitions, 291 model calls and 153 query observations. It
reused existing candidates and performed no retrieval or extraction. The claimed
fact and expected labels were withheld from model input. A selected source span
had to occur literally in the complete fact; retained qualifications were not
replaced with a bare answer span.

One bootstrap attempt failed before inference because the benchmark package
initializer imported an unrelated dependency absent from the isolated environment.
The runner was changed to load the standalone decoder directly. That failed
script and log are preserved separately. The completed model screen had zero
operational failures. None were converted into successful abstentions.

An independent audit regenerated every tokenization and reconstructed all 291
span/null decisions from raw logits. It then independently checked all 291
corrected decisions. All 97 pair groups matched across repetitions. Model assets,
original fixtures and frozen candidate inputs were unchanged. The maximum input
was 68 tokens; the frozen limits were 64 question tokens, 512 input tokens and
30 answer tokens, with null margin zero and no truncation.

The CPU run took 42.32 seconds, including 949 ms for model loading. Median pair
time was 112 ms. Median query time was 323 ms for the seven-query fixture and
224 ms for the larger fixture; nearest-rank p95 values were 349 and 306 ms.
These are local QA screen timings with source checks and logging. They exclude
retrieval and do not establish native-host, combined-pipeline or 10,000-fact latency.

Validation: 47 focused and adjacent checks, including 33 decoder cases, plus
12 documentation checks. Black and strict mypy pass for the decoder. The completed
model process exited. Production package sources and dependencies remain identical
to the previously verified package checkpoint. No live host settings, pilot
installation or memory database changed; the pilot remains inactive at zero days.

The original implementation is `2688dee5c9386549c487ebdd4ad5e275e67b958f`; the
boundary correction is `4f45720dbc286414f77b24e691fb0322b70ac691`. The original
private archive contains 30 verified members. The combined correction archive
contains 11 verified members, including that original archive. Exact asset,
input, output and archive hashes are recorded in the receipt.

## Evidence

- [Original prospective protocol](../../plans/2026-11-07-extractive-support-development-v1.md)
- [Boundary-correction protocol](../../plans/2026-11-07-extractive-support-boundary-redecode-v2.md)
- [Machine-readable receipt](extractive-answer-support-v1-v2-receipt.json)
- [Accepted-state retrieval baseline](accepted-retrieval-v1.md)
- [Earlier contrast support result](answer-support-contrast-v6.md)

The fresh corpus remains unused. No quality threshold, production default or
pilot entry requirement was relaxed. This method needs a separately evaluated
way to handle applicability and conflicting claims before promotion.
