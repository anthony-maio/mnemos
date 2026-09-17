# Embedding task contract follow-up

## Source finding

The frozen d94 Ollama embedding provider sends the supplied text unchanged to
both embedding endpoints. The engine embeds retrieval queries through the same
`embed_text_async` path used for stored content. The real-model mechanism
checkpoint also keys vectors by text without a query/document task distinction.
This is a verified source limitation; its effect on measured quality is unknown.

The recorded Nomic digest begins `0a109f422b47`, matching the current
[Ollama model listing](https://ollama.com/library/nomic-embed-text:latest), which
links to Nomic Embed Text v1.5. Nomic's
[model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#usage)
requires task prefixes and distinguishes document indexing (`search_document:`)
from retrieval queries (`search_query:`). Ollama's short usage examples omit
these prefixes. The provider follows those short examples today.

The active v4 queue remains a valid measurement of the frozen, unprefixed
configuration. It cannot establish how the recommended prefixed configuration
would perform. Do not rewrite its inputs, checkpoints, labels, or reports.

## Implementation and validation sequence

1. Add explicit query/document embedding roles with backward-compatible defaults
   for providers that expose only `embed`. Keep symmetric prediction comparisons
   in one consistent space. Map every call site before changing input text.
2. Bind configured prefixes and a contract version to the persistent embedding
   identity. Changing the contract must require a new verified index generation;
   never compare new query vectors against silently retained old document vectors.
   Do not infer a custom model's contract solely from a similar model name.
3. Carry the role through batch embedding, async offloading, evaluation caches,
   manifests, and raw request evidence. Text alone is no longer a sufficient
   cache key when the same text can have different embedding roles.
4. Test exact provider request payloads, compatibility fallback, distinct cache
   keys, restart identity rejection, and atomic reindex activation in disposable
   databases. Live data migration requires separate authorization.
5. After the existing model queue finishes, run a separately bound development
   comparison using the same cases and model digest. Change only the embedding
   task contract, keeping graph capacity and gates unchanged. Record negative
   results. Any subsequent full evaluation needs new outputs and fingerprints.

## Implementation checkpoint

The working implementation now distinguishes query, document, and symmetric
roles in the provider interface, async helpers, runtime, memory service,
modules, reindexing, and the basic retrieval benchmark. Prefixes are explicit
settings with empty defaults. Empty defaults preserve the existing persistent
identity; any nonempty prefix adds an `explicit-prefix-v1` task contract.
Changing that contract rejects the existing index identity until a new
generation is built and activated.

The provider foundation and initial runtime tests passed 19 checks. A subsequent
79-test runtime integration run passed, including saved TOML to actual mocked
Ollama request payloads, SQLite restart, identity mismatch rejection, atomic
reindex activation, and symmetric surprisal and recall-support comparisons.
Benchmark integration now preserves separate query/document vectors even for
identical text. Role-specific caches validate the task contract and dimensions
across roles and restart. Symmetric calls preserve the legacy flat cache.
The mechanism, held-out lifecycle, lifecycle preflight, and performance
launchers accept explicit prefix options and record the contract in their
run configuration. Traces retain role labels and actual prefixed provider
inputs; the existing frozen queue is unchanged.

The combined role, lifecycle, and conflict-safety set passed 35 tests. The
repository's strict mypy gate passed all 70 source files, and Black passed
164 files. These are deterministic correctness checks, not a full package
checkpoint for the new source. No prefixed real-model quality comparison has
run.
