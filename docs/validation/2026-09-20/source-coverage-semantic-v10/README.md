# Complete source-accounting v10 semantic review

All 46 original development cases have descriptive review entries. Runtime
outcomes: {'runtime_valid': 42, 'error': 4}. This candidate is not adopted. Provider failures and
validation rejections remain separate from successful abstention; inspect
runtime-errors.json for the final-response counters, without treating a rejected
inventory as a retained memory. Original rubric ambiguities remain attached.

The same v9 schema, model, cases and rubrics are retained. Prompt and post-check
now require full nonblank source-unit coverage. Coverage exposes the omitted
mixed-policy marker as a runtime error rather than recovering it. The model also
drops the benchmark-date boundary while retaining only its referential keep
instruction, and splits a hypothetical fallback from its approval qualification.
The one-turn mixed marker now retains an unwanted tool instruction.

Some unsupported optional request references disappear, while others remain or
appear elsewhere. Read the complete per-case review and preserved runtime errors.
No aggregate semantic accuracy or fresh held-out acceptance is claimed. Full
coverage and exact slices do not prove correct classification or independently
complete memories. This candidate remains unadopted.
