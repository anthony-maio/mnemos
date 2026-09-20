# Complete role vectors for graph and lifecycle development

The preparation preserves all 184 document/control vectors and five queries
from each arm of the [paired prefix measurement](../embedding-prefix-development-v1/README.md).
It adds only the authored corrected-content document vector for each arm,
so the graph experiment can retain the original lifecycle controls.

Exactly two embedding calls and 11 total local Ollama API calls completed.
Each arm has 185 document vectors and five separate query vectors. Independent
verification matches the two new vectors to raw responses and confirms that
every earlier vector remains byte-for-value unchanged. Query and document
roles remain separate even when raw text is identical.

The protocol and preparation runner were committed at `1849942` before inference.
The pinned Nomic digest, Ollama version, and model configuration match the
earlier paired measurement. Both new embeddings have positive observed GPU
residency and serving context. Requests use `truncate=false` and no retry.
Raw requests, responses, configuration observations, and input hashes are retained.

Six preparation tests pass, with formatting and strict Python 3.10 type checks.
The red report records the initial missing-module collection failure, not a
production regression. The graph/lifecycle measurement is separate; vector
preparation is not a retrieval or lifecycle result. Live configuration and
memory were untouched. The index binds raw files; this README is outside it.
