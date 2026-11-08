# Token-support preparation checks

The development-only source-witness selector and its prospective policy are implemented. Twenty-five model-free cases passed across the decoder and selector suites. After a type-narrowing repair in the decoder, its thirteen cases passed again. These are 25 distinct cases, not 38. Strict typing passed for both development modules; formatting passed for the three changed Python files.

The selector assesses each fact against each of its retained sources while including the full eligible comparison cohort and explicit reference date. It preserves source IDs, checks the existing snapshot fence around calls, and does not stop after an early positive witness. Missing scores and provider failures are unavailable, not abstention. A later failure discards earlier positive selections. Tests cover these behaviors, out-of-scope exclusion, cancellation, stale evidence, fixed-rule boundaries, and complete-request budget checks before provider work.

The candidate rule requires generated A and reported A probability at least 0.9 without renormalization over the offered labels. This is an uncalibrated prospective development rule, not evidence of 90% correctness and not a reinterpretation of the previous generated-confidence threshold. No model inference or semantic evaluation was performed in this preparation step.

The [development protocol](../../plans/2026-11-08-token-support-development-v1.md) requires separately authored development data, separate gold labels, frozen runtime and model identities, complete raw observations, and the original external quality and latency requirements. Existing failed evaluations remain closed. Production retrieval, pilot activation, and release status are unchanged.
