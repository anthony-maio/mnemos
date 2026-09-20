# Qwen reasoning configuration: empty final responses

All 36 proposal calls failed the production provider's final-response check.
The twelve previously authored inputs ran three repetitions with local
Qwen3.5:9b, thinking enabled, temperature zero, a 2,048-token output budget,
and the history-only candidate prompt. Inputs, model digest, source hashes,
and settings were frozen before execution. No facts were committed.

Every preserved API response has an empty response field, nonempty thinking,
done=true, and done_reason=stop. The token counts in summary.json are below
the configured budget. These are runtime failures, not abstentions or semantic
quality scores. The runner's terminal success means recording completed;
its status explicitly records 36 errors. Earlier results remain unchanged.

Ollama documents separate reasoning and final-answer fields for both routes:
[thinking contract](https://docs.ollama.com/capabilities/thinking). Reading
thinking as a substitute final answer would bypass that contract. The existing
provider rejected these responses correctly. No production transport, prompt,
model default, or live configuration was changed. A separate route diagnostic
will compare chat and generate with thinking enabled and disabled.

This is development evidence using previously observed cases. It establishes
neither general model quality nor the cause of the server behavior. The index
binds the original inputs, prompts, schemas, raw responses, source binding,
helpers, and terminal status; this README is outside the index.
