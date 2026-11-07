# Controlled referent model comparison v7

Neither model passed the development gate. Qwen3.5 improved relation accuracy
and consistency, but accepted more wrong targets. These results do not justify
changing the production model or comparator.

Source: `1393ef21d7431e13ac32c37b9ba826a9258846c3`.
The [frozen protocol](../../plans/2026-11-07-referent-model-comparison-v7.md)
binds model digests, inputs, settings and scoring. Both models used the same
89 model-bound prompts and schemas, generate route, thinking disabled,
temperature zero, explicit neutral penalties and identical context allowances.
Each retained its own template and tokenizer. Qwen3 was rerun under these
settings; older scores remain unchanged.

The original screen contains 20 development cases. Five of its 17 requests use
the existing relationship fallback; twelve use referent comparisons. Three
cases make no model call.

| Original screen | Qwen3:14b | Qwen3.5:9b-q4_K_M |
| --- | ---: | ---: |
| Correct operations | 18/20 | 17/20 |
| Correct reviews | 9/9 | 8/9 |
| Wrong accepted targets | 0 | 1 |
| Wrong additions | 1 | 0 |
| Operational failures | 0 | 1 |
| Median / p95 milliseconds | 1579 / 2942 | 1099 / 2212 |
| Maximum milliseconds | 3508 | 48787 |

Qwen3 missed the known correction and added an observed requirement instead of
correcting it. Qwen3.5 accepted a correction with an ambiguous environment,
treated an independent assistant statement as a conflict, and exhausted its
4096-token allowance on the unrelated-addition case. That truncated JSON remains
an operational failure in the denominator, with its original response preserved.
That failure took 48.8 seconds and is not visible in the p95 of twenty cases.

The second dataset contains 72 matched views of twelve seeds in five clusters,
with 198 labeled relations. It extends the development material and is reported
separately; its views are not independent fresh scenarios.

| Matched-view diagnostic | Qwen3:14b | Qwen3.5:9b-q4_K_M |
| --- | ---: | ---: |
| Correct operations | 49/72 | 52/72 |
| Correct relations | 109/198 | 142/198 |
| Correct operation and every relation | 33/72 | 40/72 |
| Correct reviews | 21/30 | 14/30 |
| Wrong accepted targets | 5 | 11 |
| Wrong additions | 9 | 9 |
| Operational failures | 1 | 0 |
| Seeds correct in every view, both layers | 2/12 | 6/12 |
| Seeds stable in both layers | 3/12 | 7/12 |
| Median / p95 milliseconds | 2572 / 3372 | 1649 / 2035 |

Qwen3's operational failure contained reversed word spans; runtime validation
rejected it despite a normal generation stop. Qwen3.5's eleven wrong corrections
covered incompatible audiences (four), incompatible environments (four), and
missing environments (three). Its vague-count requirement was consistently
wrong across all six views. Stability alone does not establish correctness.

In matched diagnostic views, both models chose the correct operation 39 times;
Qwen3 alone did so ten times and Qwen3.5 alone thirteen times. Both were wrong
ten times. Requiring correct relations as well gives 29 shared successes, four
Qwen3-only successes, eleven Qwen3.5-only successes and 28 shared failures.
Sequential loading, cache differences and one repetition limit timing claims.

The provider now accepts validated optional sampling controls and retains its
previous payload when those controls are omitted. Benchmark model selection
binds the manifest, loaded model and native tokenizer; per-call model changes
that would invalidate token counting are rejected. Verification passed 188
focused tests, 28 general provider regressions, twelve documentation checks,
strict type checking on four source files and formatting on seven files.

Both models passed a synthetic structured-response check with no emitted
thinking, five repeated-token and literal-roundtrip checks, and native counting
of all 89 prompts. Maximum counted input was 2866 tokens for Qwen3 and 2884 for
Qwen3.5, below the 3584-token cap. Native logs confirm the intended sampling
cutoffs and neutral penalties for all 180 generations, including two smoke
checks. Quality measurement produced 178 responses for 184 observations: 177
normal stops and one output-limit stop. No thinking was emitted.

The offline audit replayed every observation, rebuilt operation and relation
scores and seed stability, checked all 139 bound source hashes, verified exact
inputs and compared matched cases. Two audit-helper assumptions were corrected
without changing any observation or score: fallback requests do not have
referent word spans, and malformed final JSON must remain a replayable failure.
Earlier helper versions are archived. The preflight helper's initial case-ID
split error also occurred before inference and is recorded in the receipt.

Session 18896 exited zero. Both models were unloaded, ten recorded process IDs
were absent, and all three owned ports were closed. The original cleanup
command returned 255 for an already disappearing child; its warning is retained.
No extra termination or inference was needed.

The [receipt](model-v7-receipt.json) binds the 204-member private archive,
SHA-256 `4b8e55b6e6f3e18ce61be75ca51926be721be6d4667649bcc85507fb372a569b`.
This is a negative development comparison, not fresh acceptance, extraction
quality, retrieval quality or persisted-memory validation. The three-repetition
confirmation remains unrun. Live configuration, installed defaults, publication
and the inactive pilot are unchanged; pilot days remain zero.
