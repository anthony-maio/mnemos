# Qwen2.5 14B model/configuration comparison: preparation

Prepared, not started. Local discovery found qwen2.5:14b already installed with
digest 7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6.
The recorded GPU is an RTX 4080 with 16,376 MiB; this is inventory, not proof of
GPU residency or exclusive access during a future run. No model was pulled.

All 46 reused development cases, both prompt arms, schemas and rubrics are
identical to the 9B v3 comparison. Temperature remains 1.0, token cap 8,192,
timeout 180 seconds, one repetition with alternating arm order. The model is
Qwen2.5 14B and the thinking option is omitted. This compares model/configuration
bundles, not an isolated effect of thinking or model weights. The candidate's
legacy arm label durability_v2 still identifies the exact frozen v3 clause.

The runner refuses to start until the 9B run is complete, its raw observations
are archived byte-identically, and its full semantic review exists. No parallel
model evaluation is launched. Offline preflight passes 92 case/arm checks with
zero generation calls. The preparation collector verifies that only model,
thinking option and launch-gate references differ from the previous inputs.

This does not change production defaults or establish improved quality. The
fresh synthetic corpus remains unexecuted. Keep runtime failures, omitted facts,
unwanted memories and preregistered rubric ambiguities in the eventual review.
