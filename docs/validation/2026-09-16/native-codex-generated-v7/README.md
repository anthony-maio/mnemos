# Generated Codex config preflight failure

The installed wheel from source `20f271d` passed package provenance checks and
generated the isolated TOML configuration. Its direct SessionStart preflight
returned code zero with no context. The runner stopped before starting Codex's
native model turn. This is a failed preflight, not native capture evidence.

The seed existed as an accepted fact. Reproducing the seed and generic query
embeddings in their recorded order matched the stored vector exactly and
yielded cosine similarity 0.18253313. The default relevance floor is 0.2.
The fallback SimpleEmbeddingProvider therefore explains the missing context
in this fixture. No production threshold was changed.

The next transport-only attempt uses constant unit vectors to remove semantic
selection from that check. It does not replace this negative result or measure
real-model retrieval. Those quality experiments retain their pinned providers
and criteria.

The exact runner, generated configuration, diagnosis, and cleanup check are
preserved here. The temporary credential copy and runtime descriptor are
absent, and no matching fixture process remained at the cleanup check.
