# Extractive source-selection preparation v6

Prepared, not executed in this artifact. Candidate 598c38d selects contiguous
source passages by numbered lossless punctuation/line units. Memory wording
comes from exact source slices, with source role, profile, scope, time and content
hash attached. Source changes during the model await invalidate the result.
This prevents generated paraphrases; it does not prove the selected passages
are appropriate, complete or authoritative. Overlapping context is permitted,
while exact duplicate ranges are rejected. No canonical-fact deduplication or
reconciliation claim is made. Non-user reports retain their source role and
still require semantic evaluation of authority and missing evidence.

The candidate passes 32 focused tests; the combined inventory/reviewer/selector
suite passes 102. Formatting and Python 3.10 mypy checks pass. All 46 source
unit roundtrips, schema bindings and both external-digest guards pass offline.

One call per unchanged source case is planned using the same frozen Qwen3.5:9b
model and provider source as v4, thinking disabled, temperature 1, 8192 output
tokens and timeout 180 seconds. New prompt, schema and representation change
together; this is not an isolated prompt-only comparison. Original rubrics and
ambiguities remain unchanged. Review retention, complete conditions, referents,
quoted material and scope for all cases. Reused development evidence cannot
establish held-out quality or full-module/native acceptance. No production
configuration or durable data changed. Prior drafts are not given to this model.
