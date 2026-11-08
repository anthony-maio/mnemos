# One retained-model support screen

Evaluate the already retained Qwen3.5-9B Q4_K_M model against the exact 176
contrast-v5 requests from the completed v6 development comparison. These cover
44 known questions across four retrieval arms, with 27 answerable and 17
no-answer questions. Replay immutable question, candidate and source payloads;
keep labels outside inference. This is support-component development evidence,
not a new extraction/retrieval run or fresh acceptance. Do not rerun the old
model or regenerate accepted stores.

Use the existing typed OllamaJudge and unchanged contrast choices, full-source
content, reference date, complete probability validation and 0.8 selection rule.
Reconstructed typed questions must serialize exactly to the frozen originals.
Keep 8,192 context, 5,120 prompt bytes, 1,024 output tokens and a 15-second
operation deadline. No truncation, provider fallback or failed-request retry.
An unavailable operation receives no abstention credit. Preserve all original
denominators and mark unattempted requests explicitly if execution stops.

The model digest is
`56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`;
reverify the retained manifest, six blobs and the pinned Ollama executable before
loading. Own a private loopback runtime using the existing isolated model store,
one loaded model and one request slot. Do not change the normal runtime or pilot.
Use temperature zero, thinking false, seed 20261106, top_k zero, top_p one,
min_p zero, repeat penalty one, presence/frequency penalties zero. These explicit
controls override package defaults; this is a distinct configuration rather
than an isolated estimate of architecture effect.

Run one unrelated schema-validating warmup, then one fixed request-order pass,
with an overall 1,800-second process bound. Record exact outgoing prompts,
schemas, responses, selections and operation timing. Treat model probabilities
as estimates. Verify model/input/source hashes and stop owned processes and
ports afterward. No model download or training is needed.

Compare against the saved original results. Report per-arm quality, failures and
support-only latency, using unique questions as denominators. Precision and
critical retention remain 95%; recall and abstention remain 90%. A passing
component screen cannot establish critical retention: these frozen labels lack
a critical-fact registry, so report that requirement as unmeasured. A passing
development screen would still require integration checks, complete warm
retrieval at 10,000 accepted facts and independently fresh acceptance. Failure
closes this configuration without another prompt, threshold or model sweep.
