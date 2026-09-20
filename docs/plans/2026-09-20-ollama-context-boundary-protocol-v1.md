# Ollama context boundary diagnostic v1

The local loaded-model response after the instrumented Hermes run reports
16,384 context tokens. The captured request specified `options.num_ctx: 65536`,
and its earlier native-API readiness response loaded 65,536. The existing
greedy-v1 alias does not bind `num_ctx` in its parameters.

Ollama's [OpenAI compatibility documentation](https://docs.ollama.com/api/openai-compatibility#setting-the-local-context-size)
requires setting the local context size in a model definition. Verify the
observed endpoint difference with three scheduled tool-free requests, without
changing an existing alias or any host configuration:

1. Existing greedy-v1 alias through `/api/chat`, with `options.num_ctx: 65536`.
2. The same alias through `/v1/chat/completions`, with the same options field.
3. Create `mnemos-native-qwen35-9b:greedy-ctx64k-v2` from the old alias, adding
   only `num_ctx: 65536`; verify its weights, model information, template,
   capabilities, and inherited sampling parameters, then query it through
   `/v1/chat/completions`.

Each request asks only `Reply with READY.`, disables thinking, limits output
to 16 tokens using the endpoint's output-length field, and runs once with a
180-second deadline. Read `/api/ps` immediately after each call, recording
the model digest, loaded context, and positive GPU residency. Archive exact
request and response bytes, version, model metadata, elapsed times, errors,
and before/after tag digests. Stop on infrastructure failure; do not retry.
Keep all existing aliases unchanged. The new alias reuses local weights.

Commit the runner and protocol before inference. The endpoint comparison is
a configuration diagnostic, not a memory-quality or native lifecycle test.
Even if the new alias fixes context provisioning, it does not prove why
earlier models omitted actions or misselected deletion arguments. Preserve
all earlier failures and qualify their unverified effective-context claims.
