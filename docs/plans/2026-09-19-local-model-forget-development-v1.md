# Local model development comparison for reviewed forgetting

This is a prospective development diagnostic, not a native-host lifecycle
result or a replacement for T18. The fixed native Hermes v6 cohort remains
zero of three complete lifecycles. Two attempts excluded captured
conversational copies; the third omitted required source selections. The
subsequent typed validation repair is verified separately on `e454a85`.

## Purpose and runtime

Compare the locally available Qwen3.5:9b and a refreshed GPT-OSS:20b on the
same three new synthetic forgetting cases. This evaluates whether a model
can use the existing reviewed-copy workflow before committing to another
native-host cohort. It cannot isolate model architecture, reasoning effort,
or quantization as a causal explanation, and three cases do not estimate
general reliability. Preserve every case, including errors and timeouts.

Use the installed `e454a85f7b4f79ce2b483f738d68d88a15bd72d6` wheel,
SHA-256 `94011300cda669ddcabefe9e9077af0f2aa01b6ed5f23bc13c8e6b7f2a5fbf35`,
from an external working directory with Python isolation enabled. Verify
imports originate in that installed environment. Do not put the repository
root on the import path. Record the diagnostic commit, exact helper and
protocol hashes, source/wheel bindings, installed module origins, actual
Ollama version, model digests, complete show responses, and loaded-model
memory observations. Commit and hash the final inputs before inference.

The local endpoint remains `http://127.0.0.1:11434`. The failed remote `.35`
machine is not part of this run. No live host configuration or live memory
is changed. The earlier GPT-OSS tag is preserved before a registry refresh:
its show request failed with a GGUF tensor-size error. A listing alone does
not establish that a model can load. The refreshed digest must pass show and
be frozen in a profile artifact before any development-case inference.

The [frozen profiles](2026-09-19-local-model-forget-profiles-v1.json)
bind Qwen digest `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`
and refreshed GPT-OSS digest
`17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7`.
The [readiness record](../validation/2026-09-19/local-model-readiness-v1/README.md)
preserves the pull client's timeout and a later successful server-side
refresh. A separate one-request load probe returned HTTP 200 and `READY.`;
its exact `READY` assertion failed on punctuation. That failed assertion
remains recorded. The response and subsequent loaded-model observation
establish successful loading, not tool proficiency. Ollama reported
12,765,573,938 bytes resident in VRAM at a 32,768-token context.

## Fixed setup and schedule

Run Qwen3.5:9b first, then GPT-OSS:20b, once each across all three cases.
Qwen uses `think: false`; GPT-OSS uses `think: low`. Send local `/api/chat`
requests with streaming disabled and explicit temperature 0, seed 42,
presence penalty 0, top_k 20, top_p 0.95, num_ctx 32768, and num_predict 2048.
Record the options as sent and the actual returned usage and durations.
Do not describe these two reasoning settings as equivalent internal compute.

Each case permits eight model response rounds and a 300-second total budget.
Each model HTTP request has a maximum 60-second timeout, reduced by the
remaining case budget. Apply the remaining deadline to tool dispatch too.
Bound the number of tool calls per response and report budget exhaustion;
do not silently drop calls or synthesize their results. A failed case does
not prevent the other scheduled fresh cases after cleanup. No selective
retries, added repetitions, or input edits are allowed during a model run.

Use actual SQLite storage, LocalMemoryService, an authenticated local
service, and the installed Hermes tool schemas and dispatch adapter.
The five cognition modules are disabled and SimpleEmbeddingProvider supplies
synthetic retrieval vectors. This isolates tool decisions from extraction
quality; it does not measure Nomic retrieval or end-to-end model latency.
The controller may seed each disposable fixture before model execution.
Once execution begins, it may dispatch only the model's supplied operations
and arguments, supply their actual results, and read state for evaluation.

The cases use distinct topics and markers from the native release-label
fixture: backup window (`olive-harbor-316` to `silver-marsh-529`), deployment
lane (`copper-meadow-617` to `violet-canal-283`), and archive bucket
(`amber-orbit-451` to `cobalt-field-628`). Each has an asserted target,
retained user-instruction and assistant-narration copies, an unrelated
own-project control, and a foreign-project control. Preserve historical
assertions where the fixture establishes them. Bind the final exact fixture
definitions through the committed helper hash before the first request.

The same user instruction structure asks both models to forget the topic
including retained conversational copies, retrieve and inspect the target,
preview the reviewed operation, inspect selected additional facts, select
their required sources, and preserve unrelated memory. Supply neither the
target's generated ID nor the selections in the user prompt. Do not change
the installed tool descriptions for this comparison.

## Evidence and scoring

Keep raw model requests/responses, exact tool calls/results, prompts,
fixture identities, state before and after, tool timings, and final statuses.
Ollama tool calls may need transport IDs derived from response and call
indices; label that normalization and retain the untouched raw response.
Such IDs do not establish native Hermes correlation.

A passing case requires model-observed inspection and preview before apply,
unchanged preview digest/limit, exact IDs and required source selections,
inspection of each selected additional fact, and successful correlated
results. Every rejected, degraded, or failed tool result fails the case even
if a later request recovers. Final state must establish removal and
suppression of the target, all known requested copies, and their selected
sources, while both control facts remain unchanged. Check persisted state
and retained label tokens, not only a bounded retrieval response. A claimed
deletion in the final answer is not a mutation.

This diagnostic does not automatically capture the model conversation.
It therefore does not test recapture during forgetting, source replay,
restart, WSL, native lifecycle callbacks, three-host continuity, or a pilot.
Those remain acceptance work even if all development cases pass. If both
models pass here but a native host fails, investigate the host's full prompt,
capture behavior, and transport rather than declaring the model sufficient.

Report all six scheduled outcomes and operational measurements. A model
chosen for subsequent native work needs a separate prospective profile and
protocol with the unchanged lifecycle and erasure requirements. Do not
rewrite the earlier cohorts or use this diagnostic to claim module quality.
