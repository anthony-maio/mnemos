# Isolated referent-pair comparison v8

Isolating each literal pair improved both models on the matched development
diagnostic, but neither passed the gate. Qwen3.5 reached 66/72 correct operations
with no wrong additions. Its remaining six wrong corrections all selected a
production schedule from a statement that did not specify an environment.
That remains an unsafe target choice, even though it is consistent across views.

Source: `0bab5c0c4210a78ff5ec970e2d2255804a6caa06`.
The [frozen protocol](../../plans/2026-11-07-referent-pairwise-v8.md) keeps the
dataset, semantic labels, sampling controls, models and budgets from the
[v7 comparison](model-v7.md). V7 is a retained historical control; it was not
rerun or interleaved here. These are 72 matched views of twelve seeds, not 72
independent fresh scenarios. The original 20-case screen was not rerun.

The new opt-in path sends one candidate and one target per request. Each request
uses a constant local target ID and version. The runtime retains the actual
storage ID, version and scope and restores them only after validating the reply.
The final operation still considers every observed target; a second plausible
identity requires review. Source, scope, duplicate-identity and concurrent-change
guards remain in place. Failure of a later comparison cannot return an earlier
partial correction. The default comparator and installed package are unchanged.

| Metric | Qwen3 batched v7 | Qwen3 pairwise v8 | Qwen3.5 batched v7 | Qwen3.5 pairwise v8 |
| --- | ---: | ---: | ---: | ---: |
| Correct operations | 49/72 | 57/72 | 52/72 | 66/72 |
| Correct relations | 109/198 | 144/198 | 142/198 | 167/198 |
| Correct operation and every relation | 33/72 | 42/72 | 40/72 | 53/72 |
| Correct reviews | 21/30 | 21/30 | 14/30 | 24/30 |
| Wrong accepted targets | 5 | 3 | 11 | 6 |
| Wrong additions | 9 | 12 | 9 | 0 |
| Operational failures | 1 | 0 | 0 | 0 |
| Seeds correct in every view, both layers | 2/12 | 7/12 | 6/12 | 8/12 |
| Seeds stable in both layers | 3/12 | 11/12 | 7/12 | 11/12 |

Qwen3's wrong additions consist of six vague-count requirements and six polarity
reversals of an existing report rule. Its three wrong corrections also come from
the missing-environment requirement. Qwen3.5 recovered all operations outside
that family, but 31 relation errors remain, including errors hidden by otherwise
correct review outcomes. These results do not support promotion or confirmation.

The preflight verified all 198 planned pair requests and schemas. For each of
the 34 distinct literal pairs, renaming IDs, reversing targets and adding unrelated
facts produced identical requests. The existing 72 batched prompts and schemas
were unchanged. In v7, 15 Qwen3 pairs and eight Qwen3.5 pairs received different
valid relations across views. In v8, one pair varied for each model despite
identical requests. Request invariance does not guarantee model determinism or
semantic correctness.

Against its own v7 control, Qwen3 gained fourteen correct operations and lost
six; Qwen3.5 gained fifteen and lost one. For combined operation/relation
correctness, Qwen3 gained twelve and lost three; Qwen3.5 gained thirteen and
lost none. These are descriptive matched development comparisons. They do not
establish performance on new cases.

The extra calls have a measurable cost:

| Cost across 72 cases | Qwen3 batched | Qwen3 pairwise | Qwen3.5 batched | Qwen3.5 pairwise |
| --- | ---: | ---: | ---: | ---: |
| Model requests | 72 | 198 | 72 | 198 |
| Provider input tokens | 110,859 | 219,564 | 111,867 | 222,336 |
| Provider cached input tokens | 60,782 | 171,069 | 41,832 | 56,292 |
| Generated tokens | 7,733 | 10,707 | 6,471 | 9,257 |
| Median milliseconds per case | 2572 | 3575 | 1649 | 2953 |
| p95 milliseconds per case | 3372 | 6179 | 2035 | 4147 |

Sequential loading and different cache states limit timing comparisons. The
largest independently counted prompt-plus-schema was 1554 tokens for Qwen3 and
1572 for Qwen3.5, below the 3584-token cap. A smaller individual request did not
produce a smaller workload: total input almost doubled for both models.

All 396 quality responses stopped normally with no emitted thinking. The two
structured smoke checks and five tokenizer repeat/roundtrip checks per model
passed separately. Native logs confirm the frozen sampling controls for all 398
generations. The audit checked 140 bound source hashes, original request/schema
identity, input snapshots, every token-count record, 792 value-span fields,
all 144 runtime replays, operation/relation grades and seed stability. It made
no new model calls. Earlier scores and raw responses remain unchanged.

Validation passed 175 focused tests, including seventeen new checks, twelve
documentation checks, strict mypy on four source files and Black on six files.
The scope test confirms that foreign targets are rejected before any pairwise
request. Other checks cover ID/version restoration, changed targets, incomplete
responses, malformed spans and failure of a later request.

Session 98572 exited zero. Both models were unloaded, eight recorded process IDs
were absent and all three owned ports were closed. The original tree-stop command
returned 255 with a child-process warning; that warning is retained. No extra
inference or unowned-process termination was needed.

The [receipt](pairwise-v8-receipt.json) binds the private archive of 191 members,
all verified on readback, SHA-256
`3d38c40e75f0cfa7eefb8b0e27fa3578faf9569aac15384a64db5a1f57ebbcee`.
Raw requests, responses and runtime logs remain private. This is a negative
acceptance result with useful development gains. The original-screen gate,
three-repetition confirmation, fresh quality, persisted extraction and daily-use
pilot remain open. Live configuration, publication and pilot days are unchanged.
