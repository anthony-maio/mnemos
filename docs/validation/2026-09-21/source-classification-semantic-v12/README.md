# Gemma unchanged-classifier v12 semantic review

All 46 original development cases have descriptive review entries. Runtime
outcomes: {'error': 3, 'runtime_valid': 43}. This candidate is not adopted. Provider failures and
validation rejections remain separate from successful abstention; inspect
runtime-errors.json for the final-response counters, without treating a rejected
inventory as a retained memory. Original rubric ambiguities remain attached.

Only the requested model changes from v9 Qwen2.5:14b to installed Gemma3:27B.
Prompts, schema, provider, settings, cases and rubrics are otherwise identical.
The larger model recovers the mixed-policy marker omitted in v9, but duplicates
its policy meaning. It retains inspection-only deployment material, one-reply
formatting and temporary tool instructions. The temporary formatting instruction
is falsely cited as a remember request. Some earlier unsupported request
references disappear. Runtime failures are one exhausted provider retry case,
one duplicate-range rejection and the existing P07 authority rejection.
No selective retry removes the initial cold-loading failure.

This reused development comparison does not justify adoption. Exact source slices
are not semantic accuracy. No target reconciliation, durable writes, fresh holdout
or full-module/native acceptance is established. Weight, architecture and tokenizer
changes are not an isolated parameter-count comparison.
