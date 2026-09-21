# Gemma unchanged-classifier v11 semantic review

All 46 original development cases have descriptive review entries. Runtime
outcomes: {'runtime_valid': 43, 'error': 3}. This candidate is not adopted. Provider failures and
validation rejections remain separate from successful abstention; inspect
runtime-errors.json for the final-response counters, without treating a rejected
inventory as a retained memory. Original rubric ambiguities remain attached.

Only the requested model changes from v9 Qwen2.5:14b to installed Gemma3:12B.
Prompts, schema, provider, settings, cases and rubrics are otherwise identical.
Some unsupported request references disappear, and several negative examples are
correctly discarded. But temporary tool instructions and a date fixture are
retained, the one-turn marker includes unrelated reply instructions, and the
mixed-policy case fails request-range validation. P07 fails authority validation;
N06 fails range validation. These are failures, not successful abstentions.

This reused development comparison does not justify adoption. Exact source slices
are not semantic accuracy. No target reconciliation, durable writes, fresh holdout
or full-module/native acceptance is established. Weight, architecture and tokenizer
changes are not an isolated parameter-count comparison.
