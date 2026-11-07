# Reconciliation reasoning development v3

Status: prospective. Freeze source, protocol and input binding before inference.

Recent non-thinking Qwen3:14b comparisons still miss valid revisions or choose
wrong targets. The [identity decomposition](../validation/2026-11-07/identity-v2.md)
added semantic mistakes and quote failures. This study holds the direct/full
prompt and schema constant while comparing execution settings before further
prompt changes.

Use the unchanged 20-case preclassified development corpus at
`benchmarks/local_memory/fixtures/reconciliation-development-v1.json`, SHA-256
`366ae35e01f517b88a0826dc2f109d51775065d552fcd0eb992033196776fef0`.
Compare three arms once per case, for 60 observations:

| Arm | API route | Thinking |
| --- | --- | --- |
| direct_generate_off | generate | false |
| direct_chat_off | chat | false |
| direct_chat_on | chat | true |

The chat/off versus chat/on pair isolates the requested thinking setting on
the same route. Generate/off versus chat/off measures the route difference.
All arms share exact source events, supplied classifications, target versions,
direct/full instructions, decoder schema, temperature, limits and deadlines.
They include the existing duplicate-identity guard. The independent checks
must compare actual prompts, schemas and input hashes rather than assume them
equal. Use the full schedule shuffled with seed 2026110701.

The existing provider automatically uses chat for structured reasoning. An
explicit Python `force_chat` option permits the non-thinking chat control;
its default preserves automatic routing. The final answer comes from
`message.content` for chat and `response` for generate. Reasoning remains a
separate recorded field. These field and thinking controls are documented in
the [Ollama chat API](https://docs.ollama.com/api/chat) and
[thinking reference](https://docs.ollama.com/capabilities/thinking).
Record the installed model's `/api/show` capability and thinking metadata;
require advertised thinking capability before running this experiment. Record
whether each thinking-enabled response actually includes a thinking trace.
Do not infer improved reasoning from a requested flag alone.

Use the existing local Qwen3:14b Q4_K_M model, digest
`bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`,
Ollama 0.40.9, temperature zero, context 8192 and one parallel request. Give
every arm the same 4096 generated-token allowance and 3584 counted input
allowance, reserving 512 tokens for the template. Count exact prompt plus
decoder schema with the verified model's native tokenizer before each request.
Reject oversized input or an output override that exceeds the reserved context.
Compared with earlier runs, these are larger output and smaller input limits;
the within-run controls share them. Keep a 90-second provider deadline and a
120-second observation deadline. The outer runtime allows three hours for all
bounded attempts and cleanup; it does not authorize retries of a failed study.

Retain all attempts, raw responses, failures, stop reasons, final and thinking
fields, provider token accounting and elapsed time. Score only final-channel
decisions. A missing final answer, malformed output, timeout or interrupted
request is a failure and receives no abstention credit. Do not publish raw
reasoning or conversation material. Report total generated tokens as supplied
by the provider, separately from final text and reasoning-text presence; do
not assume provider output counts describe final text alone.

This is a single-repetition development screen. Thinking/chat qualifies only
for a separately frozen three-repetition confirmation if all 20 cases pass,
with zero operational failures, wrong-target choices or unsupported additions,
and thinking emission is confirmed. A screen success would not qualify for
production use or replace the full-pipeline and fresh-evaluation requirements.
If it fails, preserve and diagnose the full run before choosing the next step.
Do not change prompts, labels, limits or source while inference is active.

Independently rebuild every score from original final responses and frozen
expectations. Verify route-specific response shapes, identical input groups,
source and artifact hashes, model identity and all request accounting. Unload
the model and stop only owned processes, preserving any cleanup warnings and
the independent process/port check. No database, retrieval, embedding, native
host, live configuration, installed pilot artifact or publication is involved.
