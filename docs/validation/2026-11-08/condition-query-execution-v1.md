# Condition-query development result

The frozen 16-case probe completed once and failed semantic and timing checks. All replies passed structural validation, but only 26 of 30 predicate states were correct. Final applicability was correct in 15 of 16 cases; that aggregate hides two false establishments because another unknown prerequisite kept the conjunction unresolved. This result does not qualify for production or fresh acceptance.

| Measure | Result |
| --- | --- |
| Completed / protocol-valid replies | 16 / 16 |
| Correct predicate states | 26 / 30 |
| Correct final applicability | 15 / 16 |
| Semantically adequate returned witnesses | 9 / 11 |
| Established circumstances retained | 9 / 11 |
| Expected unknown circumstances preserved | 17 / 19 |
| Applicable cases retained | 3 / 4 |
| Provider median / p95 | 1285.17 / 1781.68 ms |
| Runner processing median / p95 | 1382.68 / 2213.31 ms |
| Generated tokens median / maximum | 51 / 58 |
| Prompt tokens median / maximum | 429.5 / 437 |

Case 04 promoted a quoted phrase from a guide to an established maintenance condition. Case 05 promoted a reported log claim, contrary to the frozen policy requiring an adopted constraint on the requested case. Both quotations were exact unique substrings, demonstrating why binding alone does not establish semantic support. Case 07 missed both explicit constraints in a requested hypothetical and returned unresolved instead of applies.

All 11 returned witnesses were inspected against the full questions and original policy. Nine were adequate, including four longer quotes in cases 11, 12, 14 and 16 that were valid alternatives to the authored references. Exact reference-object agreement was only 9/16, but these four alternatives must not be counted as semantic errors. No expected states changed. Establishment precision and retention were each 81.82%; unknown preservation was 89.47%. These are component measurements, not end-to-end answer metrics. Returning applies in three correct cases with no false applies does not rescue the missed applicable case or the hidden predicate errors.

The native token-cost check measured a 28-token median for minified authored replies. Actual generated replies had a 51-token median, including formatting and different quote lengths. The 192-token output limit was not reached; every provider reply reported `done_reason=stop`. Provider latency alone exceeded the original complete 500 ms requirement. Processing includes local request/revision checks after dispatch and decoding, but excludes initial retrieval, source extraction and native-host dispatch. No causal comparison to earlier cohorts or prompts is supported.

The wrapper reconstructed every saved owner request without changing identifiers, verified exact prompts and schemas, checked model/runtime/code hashes, and tokenized all prompts before the single execution. There were 16 semantic calls, no retries and no semantic warmups. The model unloaded, both owned processes disappeared, both ports closed, and all bound files remained unchanged. Wrapper and audit processes exited 0. Server exit 1 resulted from the verified owned-process stop.

Raw evidence, prepared inputs and labels, runtime bindings, ownership, audit and the subsequent witness review are retained in `.venv/condition-query-probe-execution-v1-evidence.zip`, SHA256 `cc4f59c491e735d10cf0ece437458b96743d365a4347c8b95d8fb11916f4c1d7`. The raw event journal SHA256 is `fa1f498781f1fb25951ca11453fa5860169ad41bd27a4dd860c468da4638f3b6`. The witness review SHA256 is `67192d20c9a23d97758972481e224aff7385dc13cb894ca70a887a69b7d65ebe`. The initial audit records witness review as pending; the separately retained later review resolves that step without rewriting the initial audit.

This cohort is closed. Do not tune and rerun it as acceptance or integrate this sequential Qwen stage into the fast path. Source-condition extraction remains supplied. The next design decision must address semantic scope and the complete latency budget together; a smaller output format alone has not solved either. Original quality, fresh-evaluation, delivery and two-week pilot obligations remain open.
