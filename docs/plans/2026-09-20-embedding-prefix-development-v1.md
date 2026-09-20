# Embedding prefix development diagnostic v1

This protocol is a prospective, paired vector diagnostic for the fixed five
cases in `benchmarks/local_memory/datasets/source-association-development-v1.json`.
It is development-only. It does not use heldout cases, production SQLite, live
memory, graph traversal, lifecycle actions, or acceptance thresholds.

## Arms

Each arm uses a fresh provider instance and fresh output. The harness does not
create or resume checkpoint directories. Both arms embed the same 36
scope-eligible facts for each case and
the same authored query. The baseline is measured freshly with empty prefixes.
The comparison arm applies `search_query: ` to queries and
`search_document: ` to every fact. Symmetric prefix is empty in both arms.
Queries and documents remain separate roles even when their text is identical.

The harness is
`benchmarks/local_memory/embedding_prefix_development_v1.py`. It records the
dataset digest, exact case IDs, protocol/options, supplied model metadata, code
binding, role prefixes, deterministic ID-tie rankings, target/seed/bridge
ranks, scope-control presence, role-specific query/document vectors, and every
scored row. It writes no database and does not call a transport itself; a
caller must supply an embedding provider factory and, for live work, persist
the provider's raw requests/results, model metadata, and loaded-GPU/runtime
checks beside the report.

## Binding and resume rules

The binding covers the pinned authored dataset digest, protocol, options, model
metadata, code SHA, and both role-prefix configurations. An existing output is
always rejected. Keep both fresh arms in one new output directory, with raw
provider request/result and runtime metadata alongside the paired report.

## Measurements and limits

For each case, rank only scope-eligible authored facts by cosine similarity,
with fact ID as the deterministic tie-break. Record the target, seed, and
bridge ranks and confirm foreign scope controls are absent. These ranks are a
pure vector diagnostic. They do not measure graph utility, source association,
retrieval quality in general, lifecycle behavior, or product acceptance.

The local metadata does not establish which upstream Nomic revision is loaded.
The experiment must record the actual model/provider metadata and runtime
health evidence supplied by the live recording layer. It must not assume v1 or
v1.5 solely from a local model name. The primary model cards are
[Nomic v1](https://huggingface.co/nomic-ai/nomic-embed-text-v1) and
[Nomic v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/main/README.md).

## Live provider boundary

The live wrapper is responsible for one paired measurement, with a fresh output
path and no acceptance threshold or three-run stochastic claim. It must use the
loopback Ollama endpoint, the pinned Nomic model digest, `truncate=false`, and
batch size 16; persist exact request and response bytes plus SHA journal entries
without retries; and verify positive GPU/context and model digest after every
embedding batch. It must capture before/after model tags and show/config
metadata, and bind the checked source and runtime inputs before and after the
measurement. These raw transport and runtime artifacts belong beside the single
paired report; this protocol does not require per-arm directories.

Run only after the native model cohort has terminated with established cleanup:
`python -m benchmarks.local_memory.embedding_prefix_live_v1 --output <new-directory-outside-repository>`.
The model is `nomic-embed-text:latest`, digest
`0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.
The request timeout is 60 seconds. Each arm embeds all 180 eligible facts,
the four foreign controls, and five queries freshly; controls are scored for
inspection but excluded from the eligible ranking by the scope predicate.
Record observed serving context separately from configured context.
