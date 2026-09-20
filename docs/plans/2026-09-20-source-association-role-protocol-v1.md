# Source-association development: separate query and document vectors

This prospective experiment asks whether the measured prefix-related seed
changes produce useful graph retrieval under the existing graph settings.
It retains the original five development cases, three relation arms, three
deterministic repetitions, and all graph and lifecycle controls. It does not
change production defaults, held-out data, thresholds, or quality gates.

## Inputs and preparation

Use the original authored dataset, SHA-256
`809a3a27ed0052e0fdf872542168af0a38337c1dc42f3d0b2af66d769cee5741`.
Use the archived paired-prefix report, SHA-256
`598f043e266a3ee704853853e9b2b105d36266252d0d94929e48883d2eeb7ab6`,
and its runtime artifact, SHA-256
`873befcfa1c0ed3c3b9ae10ac364a1dcfb9ea11fe0bcf2b8921e588b2c461d76`.
Both arms retain all 184 original fact/control document vectors and five query
vectors exactly. The unprefixed arm has empty prefixes; the other uses
`search_query: ` and `search_document: ` for their respective roles.

Generate only the missing corrected-content vector for each arm. This keeps
the original correction, forgetting, replay, and restart checks in scope.
No event embedding is needed: the fixture stores raw source events and links
them to the fact vectors through persisted evidence.

The preparation runner is `source_association_role_preparation_v1.py`. Use
local Ollama at `http://127.0.0.1:11434`, `nomic-embed-text:latest`, digest
`0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.
Require unchanged Ollama version and model configuration from the paired run,
768 dimensions, `truncate=false`, no retries, and a 60-second request timeout.
Preserve exact request/response bytes and require positive GPU residency,
the pinned digest, and positive observed context after each embedding call.
Bind code, protocol, metadata, and dataset before and after preparation.
Commit this protocol and preparation code before these two model calls.

## Measurement

Use `run_source_association_roles_v1.py` only after the prepared vector artifact
is complete and archived with a verified index. Commit the role-aware helper
changes and measurement runner before execution. Its manifest must bind the
exact prepared-vector digest and this protocol, with input hashes before and
after execution. Output and all SQLite fixtures must be fresh and outside the
repository. No inference occurs during graph measurement.

Run both embedding contracts across all five cases, three relation arms
(`semantic`, `source_association`, `combined`), and three repetitions: 90 rows.
Use all original options unchanged: top five, context budget 4,096, candidate
limit 100, four semantic neighbors, relevance floor 0.2, semantic edge threshold
0.6, activation threshold 0.3, decay 0.2, three hops, 1,000 visited nodes,
graph weight 0.1, source edge weight 1.0, and source fanout limit 16.

Use the production-selected vector seed. Never force the authored seed. The
fixture and lifecycle receive document vectors; engine retrieval and independent
score reconstruction receive query vectors. The same text may have two different
vectors under these roles. Missing vectors must fail before SQLite creation;
do not silently fall back to another role or skip a lifecycle action.

Preserve the original restart, reverse insertion, edge-cut, untouched-semantic
parity, scope, score/budget reconstruction, activation-oracle, and persisted-edge
checks. Retain typed source relations and the semantic-only negative control.
Run the original correction, forget, replay, and restart lifecycle for both
embedding contracts. Its known shared-source preservation conflict stays a
failed control if it recurs; do not drop it from aggregate status.

## Interpretation

Report direct-target candidacy, structural and activation paths, target rank,
top-five appearance, and every failed control separately for each embedding
contract and relation arm. Include every case and scheduled repetition. A
complete measurement with failed controls is not a passing experiment.
Three deterministic repetitions do not create 15 independent cases per arm.

This is synthetic development extraction with real model embeddings. Source
relation views are experimental, read-only graph augmentation. A new path can
demonstrate a mechanism without yielding a useful top-five result. Any positive
result remains separate from held-out utility, all-five-module quality, native
host reliability, and the operational pilot. Preserve the previous negative
measurements and their original bindings.
