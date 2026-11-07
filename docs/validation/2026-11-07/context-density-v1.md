# Native local tokenizer and context density v1

The existing host-tokenizer packing interface passed a real local Qwen tokenizer
check. The contrastive policy reduced returned hybrid memory from 1,978 to 767
tokens across 44 questions while retaining all 27 expected facts. It still
returned five unsupported facts. This measures context size; the policy still
fails its [answer-support quality gate](answer-support-contrast-v6.md).

The [protocol](../../plans/2026-11-07-context-density-v1.md) and runner were
committed at `66afe57209743cc156ee113aa257010685fcf56c` before token collection.
This postprocessed the completed comparison's 176 frozen retrieval inputs and
352 support results. It generated no new answers and performed no extraction.
All source facts, order, versions, labels and original latency values remained
unchanged. All 176 reconstructed unfiltered contexts matched their original
UTF-8 byte counts.

A separately owned Ollama 0.40.9 runtime exposed the installed Qwen3 14B model's
native llama-server tokenizer. The receipt binds the executable, native binary,
support libraries, model manifest digest and actual model blob hash. Literal
counting used `add_special: false` and `parse_special: false`, as described in
the [native server documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
This uses an internal native runner interface, not a public Ollama API.

All 88 distinct texts were tokenized twice and detokenized exactly, including
Unicode, code and special-token syntax probes. The 264 raw requests are retained.
A fixed table of those native counts then supplied the pure `HostTokenizer`
callback. Packing counted full concatenations and kept records whole, including
IDs, versions and qualifications. It did not summarize or clip sources.

At the 1,200-unit reference budget, both methods retained each complete original
selection. The following totals cover the hybrid arm's 44 questions; density
means expected facts per 1,000 returned context tokens.

| Selection | Native tokens | UTF-8 bytes | Expected facts retained | Facts / 1,000 tokens | Clean answers / 27 | Abstentions / 17 | Unsupported facts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unfiltered retrieval | 1,978 | 8,342 | 27/27 | 13.65 | 5 | 0 | 55 |
| Joint support | 855 | 3,535 | 27/27 | 31.58 | 26 | 13 | 7 |
| Contrastive support | 767 | 3,266 | 27/27 | 35.20 | 27 | 12 | 5 |

The contrastive selection used 61.2% fewer returned context tokens than
unfiltered retrieval and 10.3% fewer than joint support. Those savings exclude
selection overhead. Its original hybrid support calls reported 32,287 prompt
tokens and 5,630 output tokens, with median latency 2,736 ms. This measurement
does not establish a net token, cost or latency improvement for an application.

The numeric budget has different units with each method. At 64 native tokens,
the contrastive selection retained all 27 expected facts and the same five
unsupported facts. At 64 bytes, it retained only two expected facts and no
unsupported facts. That smaller context had higher apparent density and 17/17
abstentions, but recall collapsed to 2/27. Density or abstention alone would
therefore give a misleading result.

All four arms, three selections, two counting methods and budgets 32, 64, 128
and 1,200 produced 4,224 measurement rows and 96 aggregate groups. The runner
also checked 2,112 injected counter failures against ordinary byte fallback.
An independent audit reconstructed every selected ID, literal context, token
count, grade and aggregate from frozen facts and raw tokenizer replies.

Token collection took 462 ms in total, with median HTTP request time 0.568 ms
and p95 0.953 ms. These timings include tokenization and detokenization requests.
Hybrid reference-budget packing medians were roughly 0.003-0.004 ms using the
precollected table. That is lookup and packing time, not end-to-end live counter
integration latency. Seventeen existing context tests, formatting and strict
types passed. No production packing code changed for this measurement.

The [receipt](context-density-v1-receipt.json) binds 26 verified private archive
members under SHA-256
`dad307eded4f048fdae92b493e441d61e50af1ca1be520af12bdea70797988ee`.
Initial archive verification failed because ZIP normalized Windows separators
while member lookups retained them. The original archive and helper remain
preserved; corrected packaging passed without repeating the measurement.

The model unloaded and both owned endpoints closed. The original process-tree
stop reported one unsupported termination. Follow-up inspection found every
listed process absent; no additional kill or runtime retry was needed. Both
the original cleanup output and independent verification remain in the archive.

This establishes local Qwen token accounting on reused development data. Codex
and Claude tokenizer compatibility, deployed host integration, source-backed
span compression, fresh quality evaluation and actual pilot use remain open.
