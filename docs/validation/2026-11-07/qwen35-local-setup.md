# Local Qwen3.5 candidate setup

Qwen3.5-9B Q4_K_M downloaded into the private evaluation model directory and
loaded successfully with context 8192 and one native request slot. The runtime
reported 5,862,210,928 bytes of model residency on the RTX 4080. The empty load
generated no response tokens. This establishes local loading only; no quality
or structured-output result is claimed.

The candidate is the explicitly quantized
[Ollama package](https://ollama.com/library/qwen3.5:9b-q4_K_M) for the
[Qwen model](https://huggingface.co/Qwen/Qwen3.5-9B). Its package contains about
6.6 GB of original blobs, including a vision projector. No private source or
conversation material was sent to a remote inference service.

The registry manifest and runtime-reported model digest are
`56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`.
The installed manifest has SHA-256
`7dd7bb310c1f18ad132c7fd72e73d4f6cff2e916c596a818701d2a1269037c7f`.
Ollama appended a layer referencing the original manifest, which remains
stored verbatim by its hash. Every original layer and all other manifest
fields match exactly. Six original blobs passed independent size and SHA-256
checks. Existing private model manifests remained unchanged.

Two failed setup checks are preserved. The first downloaded successfully but
stopped before loading because it expected identical manifest bytes. The second
incorrectly stripped `from` fields already present in the original layers and
also stopped before loading. The third checked the exact original layer list
plus the added manifest reference and completed loading. Neither failure was a
model-quality result, and neither was rewritten as a pass.

The installed runtime is Ollama 0.40.9, binary SHA-256
`0957bdaa7b52e163cdc7feb1275653f27c6b32e8ba309e957bdbb83493e97a4b`.
The native runner is bound separately in the
[receipt](qwen35-local-setup-receipt.json). The model advertises completion,
vision, tools and boolean thinking controls. Actual structured responses,
thinking emission and native-tokenizer round trips remain untested.

The package defaults include presence penalty 1.5, temperature 1, top-p 0.95
and top-k 20. Those settings differ from the earlier model's package; the next
comparison must explicitly bind or control them. A model-name substitution
alone is insufficient to claim a controlled model comparison.

All setup models unloaded, listed owned processes were absent and their ports
closed. The final tree-stop warning remains preserved. The private archive has
24 verified members and SHA-256
`0cc44e3b5b9a769bc38e880a631ba4d18dc551e0fd3af6889b6a9f9f8bc4a6dc`.
Live host settings, installed pilot artifacts and production defaults were not
changed. The new model is not part of the Qwen3 relation diagnostic.
