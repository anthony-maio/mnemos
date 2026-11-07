# Temporal answer-support comparison v4

The qualified temporal option did not improve abstention in this development
comparison. Keep it experimental. Both treatments used the same 176 frozen
inputs and explicit reference date; the run completed all 352 planned
observations across 44 questions in one repetition. Extraction was disabled.

For hybrid retrieval, both treatments returned an expected answer on 24/27
answerable questions. The baseline abstained correctly on 14/17 no-answer
questions; the qualified option abstained on 11/17. Clean answerable returns
were 23/27 and 24/27 respectively. Each had three timeouts. Failed operations
receive no abstention credit. These small, observed development sets are not
fresh evaluation and do not justify default promotion.

## All retrieval arms

Pooled results retain all attempted observations. Each arm has 27 answerable
and 17 no-answer questions. Clean means the selected IDs exactly match the
frozen expected set, with a successful assessment.

| Treatment | Arm | Answer hits /27 | Clean hits /27 | Abstentions /17 | Unsupported facts | Failures | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| joint | vector | 27 | 26 | 14 | 6 | 0 | 2407.05 |
| joint | lexical | 26 | 25 | 14 | 5 | 1 | 2343.11 |
| joint | hybrid | 24 | 23 | 14 | 6 | 3 | 2338.74 |
| joint | full | 26 | 25 | 13 | 7 | 1 | 2343.88 |
| qualified | vector | 27 | 27 | 11 | 7 | 0 | 2527.72 |
| qualified | lexical | 26 | 26 | 12 | 6 | 1 | 2489.94 |
| qualified | hybrid | 24 | 24 | 11 | 7 | 3 | 2451.8 |
| qualified | full | 26 | 26 | 11 | 6 | 2 | 2484.97 |

## Separate datasets

Hybrid results separate the reused twenty regression questions from the
twenty-four temporal development questions designed after v3.

| Dataset | Treatment | Answer hits | Clean hits | Correct abstentions | Failures |
| --- | --- | --- | --- | --- | --- |
| Original regression | joint | 8/10 | 8/10 | 10/10 | 2 |
| Original regression | qualified | 8/10 | 8/10 | 7/10 | 2 |
| Temporal development | joint | 16/17 | 15/17 | 4/7 | 1 |
| Temporal development | qualified | 16/17 | 16/17 | 4/7 | 1 |

## Evidence and limits

Source `c320c3ef` and all 129 bound source-file hashes verified. Both dataset
hashes, all 176 paired input groups, every request state, all 22 original
stores and reconstructed snapshot hashes verified. The audit retains exact
control outcomes and nineteen paired selection differences for later review.

There were 620 judgment requests and 609 raw responses. Eleven requests without
responses are bound to failed observations. All 609 replies supplied complete
valid distributions and ended with `stop`; no probabilities were filled or
normalized. The maximum rendered prompt was 4,445 UTF-8 bytes. Provider token
totals cover retained replies and do not measure the unobserved work of timed-out
calls or the native host context.

The private runtime used the frozen 14B model, 8,192 context, 15-second operation
deadline and unchanged 0.8 selection threshold. Early throughput was slow and
other applications were using the device. The original timeouts remain in the
results. One repetition under shared device activity does not establish a
latency guarantee.

Cleanup passed: the model was unloaded, the private server tree stopped, its
port closed, and owned processes were absent. The [receipt](answer-support-temporal-v4-receipt.json)
binds a private archive with 213 verified members, including raw requests and
replies, failure/control audits, sources, stores and runtime logs.

Detailed semantic failure review is the next step after the requested break.
Do not retune on these results and call the next run held out. Fresh evaluation,
separate extraction, context efficiency, actual pilot use and delivery remain
open. No live configuration, published package or hosted page changed.
