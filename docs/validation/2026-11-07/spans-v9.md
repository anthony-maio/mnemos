# Literal identity-dimension evidence v9

Reject this candidate. Splitting the judgment into subject, attribute and
qualifiers reduced Qwen3.5 from 66/72 to 30/72 correct operations. It made no
corrections at all, including the thirty cases that required one. Zero wrong
corrections therefore does not establish a useful safety improvement. The run
also produced ten wrong additions and twelve evidence-validation failures.

Source: `ee4f469bcf12b35c9d9fb47d081c6c9dbfe64c6a`.
The [frozen protocol](../../plans/2026-11-07-identity-spans-v9.md) uses the same
72 development views, labels, Qwen3.5 package, sampling controls and resource
budgets as [v8](pairwise-v8.md). V8 is a retained historical control, not an
interleaved rerun. These are views of twelve seed scenarios, not fresh acceptance.

The opt-in contract requests three identity dimensions with a literal word span
on each side. Runtime validation rejects a same/different subject or attribute
without evidence on both sides. Same qualifiers require evidence on both sides
or neither; different qualifiers require both. One absent side requires
unspecified. The runtime derives the overall relation and restores storage IDs
and versions. The existing comparator paths and installed package remain unchanged.

| Metric | Qwen3.5 pairwise v8 | Identity spans v9 |
| --- | ---: | ---: |
| Correct operations | 66/72 | 30/72 |
| Correct relations | 167/198 | 76/198 |
| Correct operation and every relation | 53/72 | 14/72 |
| Correct reviews | 24/30 | 18/30 |
| Wrong accepted targets | 6 | 0 |
| Wrong additions | 0 | 10 |
| Operational failures | 0 | 12 |
| Seeds correct in every view, both layers | 8/12 | 2/12 |
| Seeds stable in both layers | 11/12 | 8/12 |

The new path recovered no operation or combined success that v8 missed. It lost
36 previously correct operations and 39 combined successes. Its actual operations
were 38 reviews, 22 additions and twelve failures. Eighteen reviews and twelve
additions were correct; none of the thirty expected corrections succeeded.

All twelve runtime failures claimed a qualifier relation without the required
evidence on both sides. The runtime correctly rejected those claims. Valid
responses still confused a changed value with a changed property: one ordinary
schedule correction labeled Tuesday versus Monday as different attributes.
The missing-environment requirement also sometimes became an addition instead
of review, using whole-statement spans as incompatible attributes or conditions.
The structural checks cannot make those semantic assignments correct.

There were 198 planned pair comparisons. Failures stopped twelve cases before
their remaining targets, leaving 26 pairs unrequested. All 198 relation labels
remain in the denominator, with no partial credit for a failed case. The model
made 172 quality requests; all responses stopped normally with no emitted
thinking. The failure was evidence validation, not malformed JSON, reversed
spans, token exhaustion or transport failure.

The preflight verified all 198 planned requests across 34 distinct literal pairs.
IDs, target order and distractors did not change a pair's prompt or schema. It
also replayed all 198 v8 requests with their original responses and recovered
the original decisions. The new run observed all 34 literal pairs. Their valid
overall relations did not vary, but that consistency included incorrect answers.

| Cost across the planned 72 cases | Pairwise v8 | Identity spans v9 |
| --- | ---: | ---: |
| Actual model requests | 198 | 172 |
| Provider input tokens | 222,336 | 191,244 |
| Provider cached input tokens | 56,292 | 48,920 |
| Generated tokens | 9,257 | 28,727 |
| Median milliseconds per case | 2953 | 7074 |
| p95 milliseconds per case | 4147 | 10,395 |

The smaller request count and input total reflect unrequested comparisons after
failure. They are not an efficiency gain. Output increased substantially, and
the measured per-case latency was worse. Cache state and the historical control
limit causal timing claims. The largest native prompt-plus-schema count was
1954 tokens, below the 3584-token cap; no response exceeded 220 generated tokens.

The independent audit verified 141 source hashes, original snapshots, exact
requests and schemas, native token records, 1,032 span fields and 516 dimension
judgments. All spans were structurally valid; twelve qualifier judgments lacked
the required witnesses. It replayed all 72 runtime outcomes and reconstructed
operation, relation, combined and seed-stability scores without model calls.
The separate smoke response, five tokenizer repeat/roundtrip probes and all
173 native sampling log records matched the frozen setup.

Validation passed 207 focused tests, including 28 new checks, twelve documentation
checks, strict mypy on five source files and Black on seven files. The tests prove
the structural contract and runtime guards; they do not override the negative
model result.

Session 5382 exited zero. The model was unloaded, six recorded process IDs were
absent and both owned ports were closed. The original tree-stop command returned
255 with a child-process warning; it is retained. No extra inference or termination
was needed. The [receipt](spans-v9-receipt.json) binds the 174-member private
archive, with every member verified, SHA-256
`e64dd51e621a195117873b92fe897172c0a232fe119d8752b8fb9eff4b159729`.

V8 remains the stronger experimental baseline and still fails its safety gate.
Fixing v9's output-schema witness combinations alone would not repair its valid
but incorrect semantic judgments. No confirmation, promotion, fresh-quality pass,
installed change, pilot day or publication follows from this run.
