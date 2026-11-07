# Native local tokenizer and context density v1

Status: prospective measurement. Run only after the contrast v6 comparison and
its independent audit finish. Preserve that run's inputs, scores and latency.
Commit this protocol and the measurement runner before collecting token counts.

Use the completed contrast v6 development records: 22 accepted-fact scenarios,
44 reused questions and four retrieval strategies. Compare unfiltered retrieval,
`joint_v1` selections and `contrast_v5` selections without new judgments. This is
postprocessing of development data, not fresh answer-quality evaluation. Failed
support operations remain failures and receive no abstention credit.

Reconstruct complete context records from the retained fact IDs, versions,
contents, assertion status and retrieval trace. Require the unfiltered rendered
UTF-8 size to match every frozen retrieval record. Keep records whole and in the
original order. Do not summarize, clip or rewrite their content. Expected fact
sets stay frozen. Selection cannot add facts outside the original retrieval.

Start a separate owned loopback runtime using the same installed Ollama 0.40.9
and existing Qwen3 14B Q4_K_M model as contrast v6. Check the executable and
model digest; retain native runner identity, command line, binary hashes and
model blob hash. Empty-prompt loading is allowed. No answer generation, model
download, cloud request or existing application shutdown is part of this check.

Use the native llama-server tokenizer route with `add_special: false` and
`parse_special: false`. Count literal memory text, including strings that resemble
special tokens. Check repeated tokenization and exact detokenization for every
distinct text counted. Also check explicit ASCII, Unicode, code and special-token
syntax probes. Preserve requests, responses and timing. This is an internal
runner interface, not a supported public Ollama API. The [native server
documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
defines these tokenization options; the installed runner must pass the checks.

Build a fixed native count table before packing. Supply its pure lookup function
to the existing `HostTokenizer` interface. Count full concatenations, not sums of
individual record counts. Compare native token budgets and the UTF-8 byte
fallback at numeric budgets 32, 64, 128 and 1,200, top five. The same number means
different units for those two methods; report actual native tokens for both.
At 1,200 both should retain each original selection, whose source retrieval was
already capped at 1,200 bytes. Check an injected counter failure against the
ordinary byte fallback on every input and budget.

Report each method and selection separately for every retrieval strategy:
total context tokens and bytes, supported facts per 1,000 context tokens, fact
recall, answerable hits, clean answerable hits, correct no-answer abstentions,
unsupported facts and operational failures. Empty pooled context has undefined
density, not an infinite efficiency score. Report packing latency separately
from native tokenization and from the original retrieval and support latency.
Cached count lookup timing is not end-to-end native integration latency.

Bind the source files, protocol, completed comparison manifest, observations,
frozen inputs and independent audit by SHA-256. Keep the raw tokenization log
and all measurement rows. Verify their aggregates independently before reporting.
Unload the model and stop only the owned runtime when done.

This can establish local Qwen token accounting and the existing packing contract.
It cannot establish Codex or Claude tokenizer compatibility, a deployed host
integration, source-span compression, fresh quality results or pilot acceptance.
