# Host context tokenizers

Python integrations can supply the tokenizer used by their host through
`MnemosEngine(context_tokenizer=...)`. The built-in service and installed host
adapters still use the UTF-8 byte bound by default. Mnemos does not infer an
encoding from a model name or download a tokenizer for this option.

```python
from mnemos.context_budget import HostTokenizer
from mnemos.engine import MnemosEngine

# These values and this function come from the host integration. The counter
# must count literal text, including any text resembling special token syntax.
tokenizer = HostTokenizer(
    name=host_tokenizer_name,
    revision=host_tokenizer_revision,
    count_tokens=host_count_tokens,
)
engine = MnemosEngine(
    config=config,
    embedder=embedder,
    store=store,
    context_tokenizer=tokenizer,
)
```

The counter must be synchronous, deterministic and side-effect free. It must
return a positive integer for nonempty rendered context. The caller owns the
compatibility between that counter, its recorded revision and the host model.
Changing the model may require changing the counter. Callable code comes from
the integration, never a retrieval request or stored memory.

Packing counts the complete concatenated context on each proposed addition,
including fact IDs, versions, qualifiers and newlines. It does not add separate
record counts, which can differ from tokenizing the concatenation. Oversized
records are skipped whole so a smaller later fact can still fit. Facts are not
summarized or clipped. A revision detected after access metadata is recorded is
omitted rather than substituted into the already-ranked result.

The budget covers the memory context string. The host must reserve its own
space for prompts, tool envelopes, other messages and output tokens.

If the counter raises or returns an invalid count, selection restarts from all
ranked candidates using UTF-8 bytes. This also revisits candidates skipped by
the failed counter. The fallback assumes byte-based text tokenization; a host
with a different contract needs a compatible explicit counter. No remote
counting or automatic model fallback occurs.

Built-in engine results report:

| Field | Meaning |
| --- | --- |
| `token_count_method` | `host_tokenizer` or `utf8_bytes_upper_bound` |
| `context_token_count` | Measured count with a supplied tokenizer; otherwise null |
| `context_token_upper_bound` | The quantity used to enforce the budget: measured count or byte bound |
| `context_byte_count` | UTF-8 size of the actual context string |
| `tokenizer_name`, `tokenizer_revision` | Identity supplied by the integration, including on fallback |
| `tokenizer_fallback_reason` | `tokenizer_error`, `invalid_token_count`, or null |

Per-fact retrieval scores retain their byte-bound accounting; the aggregate
fields above describe the final context. Legacy custom engines and early empty
results can retain the older byte-bound response without the additional fields.

The [local Qwen measurement](validation/2026-11-07/context-density-v1.md) checks
the packing contract against native token IDs and reports supported facts per
token, recall and latency on reused development data. The tested support policy
still fails its abstention target. Codex and Claude tokenizer compatibility,
deployed host integration, fresh quality evaluation and source-backed span
compression remain open.
