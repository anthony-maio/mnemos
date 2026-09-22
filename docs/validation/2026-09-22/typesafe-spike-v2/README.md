# TypeSafe judgment spike v2: quote the text in the question

Same fixture, backend, and runner as [v1](../typesafe-spike-v1/README.md),
after one design change and two description changes, TypeSafe only.

The design change came from the first live capture through the service, not
from the fixture. A two-sentence prompt ("For this project, release notes go
in CHANGELOG.md under the next version header. Also, for this reply only,
keep it to two lines.") came back as two temporary instructions at 0.67 and
0.65. Asked about each sentence alone, the judge gave 1.0 to
continuing_requirement and 1.0 to temporary_instruction. The ranges had sat
side by side in the shared state and the judge blurred them, exactly the
"large irrelevant state" jaggedness the vendor documents. Every question now
quotes the text it is about inside its own instructions (range, unit pair,
candidate and target), with the whole source kept in state as context. The
same two sentences then come back at 1.0 and 1.0 in one request. The
description changes: continuing_requirement and temporary_instruction now
name scope as the deciding signal ("for this project" is a requirement, "for
this reply" is temporary), and asserted_fact excludes a passing status such
as "the service is down right now".

| backend | ok / cases | presence agreement | twin pairs | latency median / p95 | cost |
|---|---|---|---|---|---|
| baseline, Gemma3 27B, prompt-driven | 43 / 46 | 0.860 (43) | | | |
| v1 TypeSafe jev-1.13.0 | 46 / 46 | 0.891 (46) | 11 / 13 | 388 ms / 468 ms | $0.0040 |
| v2 TypeSafe jev-1.13.0 | 46 / 46 | 0.870 (46) | 10 / 13 | 393 ms / 463 ms | $0.0040 |

One case either way on 46 is noise; the fixture did not move. The live case
did. The six v2 misses are the same hard cases as before: two dated reports
phrased as instructions that the rubric wants kept as history and the judge
reads as one-turn work; a quoted benchmark date the user calls test data,
labeled a requirement at 0.81; a rejected proposal at 0.36, which the
confidence gate routes to review; a present-moment outage status kept as a
fact at 0.70; and a quoted old proposal read as a rejection at 0.62.

Tuning stops here. These are development fixtures the prompts were written
against, and the next number that matters comes from the held-out corpus.
Zero contract errors; nothing else about the limits in v1 changes.
