# Structured chat extraction development comparison

All 36 calls completed and passed the existing extraction response validation.
Twelve previously authored cases ran three times with the history-only candidate
prompt, local Qwen3.5:9b, thinking enabled, temperature zero, an 8,192-token
budget, and a 180-second timeout. The frozen preparation records exact model,
source, inputs, and settings. The experimental provider uses the chat endpoint.

proposal-review.json contains the final proposals for semantic review. Runtime
validity is not a semantic accuracy score. These are previously observed
development cases, not an unseen holdout. No facts were committed and no live
configuration or production extraction prompt was changed. The separate
production-prompt comparison tests the managed production provider configuration.
Original failed generate-route runs remain archived without revised verdicts.
