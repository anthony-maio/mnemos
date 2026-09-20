# Native Hermes request observation v1

This prospective development diagnostic follows the failed
[v8 cohort](../validation/2026-09-20/hermes-native-cohort-v8/README.md).
That cohort preserves three failed lifecycles. Two reached forgetting and
submitted invalid arguments. Its metadata-only SDK trace cannot establish
the complete serialized tool schemas or messages sent to Ollama.

A [read-only persistence diagnostic](../validation/2026-09-20/hermes-v8-preview-persistence-v1/README.md)
subsequently confirmed that attempts 2 and 3 saved complete previews matching
their native tool results. This narrows the unresolved boundary to outbound
request construction; persisted messages alone do not prove wire content.

Run one fresh lifecycle attempt with the v8 prompts, controls, model profile,
scope, capture, deadlines, and evaluator unchanged. Use
`hermes_lifecycle_observation_v1.py`, `hermes_native_bootstrap_v5.py`, and
`hermes_request_observer.py`. Commit their hashes and this protocol before
inference. Do not edit or rerun the v8 cohort and do not selectively repeat
failed phases. If this attempt stops before forgetting, report that limit.

Keep the verified runtime `e419c2aa615e037e639a74b14615889d03f4e47f`, wheel
SHA-256 `185f3df7720a32e5ed285d9c507e1e45e7f57370942740b53c4f8178bb97a97e`,
local Ollama, the Qwen3.5:9b greedy-v1 alias/digest, Nomic embedding digest,
and all five modules disabled. Preserve the isolated Hermes revision and
cwd-only patch. The request observer is benchmark instrumentation, not a
runtime-package change or a model-quality repair.

## Observation contract

Observe synchronous HTTPX sends only to the exact local
`http://127.0.0.1:11434/v1/chat/completions` endpoint. Save the eager request
body bytes before dispatch, their byte count and SHA-256, and the endpoint
and method. Do not log HTTP headers, authorization, or other endpoints.
Reject a matching streamed request rather than consuming it and claiming
unchanged observation. Preserve the original response object and network
exceptions; restore the send method when the observation context closes.
Async requests are outside this observer's coverage.

The synthetic disposable host contains only fixture content and the existing
isolated configuration. Never install the observer in a live conversation.
Keep the original metadata trace and add the model response finish reason.
Do not set tool choice, change generation limits, reconstruct omitted
arguments, or alter tool results. No inference occurs during import preflight.

## Questions and reporting

1. Does the sent `mnemos_forget_related` schema include both selection arrays,
   their types, defaults, and required-field list? Record actual tool choice.
2. For each reached preview, which subsequent requests contain its complete
   digest, candidates, and required source IDs? Distinguish saved host messages
   from the exact outbound body. Report any persistence, pruning, or wrapping.
3. What response finish reasons accompany announced actions or missing
   arguments? Preserve host-generated continuations and their request bodies.
4. Which unchanged lifecycle gates pass, fail, or remain unreached? Retain
   all outcomes, cleanup evidence, and input/package/model bindings.

A successful instrumented phase demonstrates that phase only. This single
development attempt does not replace the fixed cohort or establish reliable
native behavior, three-host continuity, complete capture, biomimetic quality,
Desktop acceptance, or a daily-use pilot. A missing observation is missing
evidence, not proof that a schema or result reached the model intact.
