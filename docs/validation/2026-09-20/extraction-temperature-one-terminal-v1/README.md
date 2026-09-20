# Temperature-only extraction comparison: terminal record

All 40 calls completed, preserving planned case/arm order and all failures.
Runtime outcomes: {"production": {"runtime_valid": 20}, "history_only": {"runtime_valid": 20}}. Only temperature changed from zero to
1.0 relative to the original authored comparison. Source 05c71d2, the original
production/history prompt arms, model digest, thinking mode, 8,192-token budget,
180-second timeout and one repetition are bound in preparation.json. This is
not the full sampling recipe recommended by the model author.

The earlier positive-case review is byte-matched to this terminal trace.
Raw requests/responses and a proposal-only projection are preserved for full
semantic review. The exact-output comparison measures changed parsed results;
it is not an accuracy score. The original rubric and P07/N10 ambiguities remain.

These are reused development cases. One stochastic repetition cannot establish
reliability, held-out quality, or an adoption decision. No production settings
or prompt changed and no facts were committed. This is not a five-module
ablation, native-host recovery result or completion of T18.
