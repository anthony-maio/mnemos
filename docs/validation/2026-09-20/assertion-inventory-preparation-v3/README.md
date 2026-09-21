# Qwen3.5 qualified inventory v3 preparation

This freezes 46 unchanged reused development cases for local Qwen3.5:9b,
thinking enabled, temperature 1.0, 8192 output tokens and a 180-second timeout.
Exact model digest, local RTX 4080 inventory, Ollama version and candidate/provider
source hashes are in preparation.json. Provider/runtime code remains cc940ac;
the candidate adds a separate v3 contract without changing production defaults.

All 51 v1/v2/v3 candidate tests, formatting and Python 3.10 mypy pass. Offline
preflight validates all 46 source schemas, prompt isolation and source-ID
bindings with zero generation calls. Missing/wrong external preparation digests
fail before generation. Live launch must use this archived preparation digest.

Compare semantics with the v2 experiment and its separately recorded offline
v3 replay. Model, thinking mode and prompt differ, so this cannot isolate model
weights. Literal quotes are not semantic proof or evidence of retention intent.
Keep original rubrics, ambiguity notes and failures. No fresh evaluation corpus
is exposed, and no target reconciliation or durable writes are attempted.

At this preparation checkpoint the model run is NOT STARTED. Consult a later
terminal record for results. No quality acceptance or adoption is claimed.
