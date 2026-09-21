# Explicit-discard classification v9 semantic review

All 46 original development cases have descriptive review entries. Runtime
outcomes: {'runtime_valid': 45, 'error': 1}. This candidate is not adopted. Provider failures and
validation rejections remain separate from successful abstention; inspect
runtime-errors.json for the final-response counters, without treating a rejected
inventory as a retained memory. Original rubric ambiguities remain attached.

The explicit-discard prompt/schema changes retain the same model, temperature,
46 development cases and original rubrics as v8. The options question, fictional
moon, hypothetical report and date fixture now emit no memories. The unaccepted
assistant suggestion now emits no memory instead of an authority rejection.

The independent marker in the continuing-policy case is still omitted. A
hypothetical fallback still receives a false retention-request reference, and
P01, P06 and P08 cite requirements/preferences as if they were remember requests.
P07 still fails authority validation. Some discarded-kind labels are imprecise.
Original ambiguity notes remain attached, including incident and rejection history.

This is one reused-development run, with prompt and schema changing together.
No aggregate semantic accuracy or isolated causal claim is made. Exact excerpts
and runtime validity do not establish complete selection, target reconciliation,
durable writes, fresh holdout or full-module/native acceptance. Candidate unadopted.
