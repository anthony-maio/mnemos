# Extractive source-selection v7 semantic review

All 46 original development cases have descriptive review entries. Runtime
outcomes: {'runtime_valid': 44, 'error': 2}. This candidate is not adopted. Provider failures and
validation rejections remain separate from successful abstention; inspect
runtime-errors.json for the final-response counters, without treating a rejected
inventory as a retained memory. Original rubric ambiguities remain attached.

The unchanged extractive contract uses Qwen2.5:14b instead of Qwen3.5:9b.
It recovers the release-format rule, benchmark policy, explicit PDF correction,
one-time report, one-turn mixed-message marker and manuscript requirement omitted in v6.
It also drops the independent marker in mixed-marker-13-policy, a regression.
Selected passages retain the original wording and qualifications.

The additional recall comes with unwanted retention: inspection-only deployment
and date fixtures, an options-only question, fictional material and a hypothetical
test report. Their original negations and attribution remain, so unwanted
retention must not be mislabeled adoption of the embedded claims. Two non-user
selections fail authority validation; neither is successful abstention.

This is a single reused-development repetition with all46 original rubrics and
ambiguity notes unchanged. Runtime counts and exact source slices do not prove
semantic accuracy. Neither model configuration is adopted. Reconciliation,
canonical-fact deduplication, durable writes, fresh holdout and full-module/native
acceptance remain unmeasured. Model version, tokenizer and weights change together;
no isolated parameter-count or general superiority claim is established.
