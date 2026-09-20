# Paired embedding prefix development result

One frozen paired measurement on five authored development cases completed
with the pinned local Nomic embedding model. Each arm freshly embedded 180
eligible facts, four foreign controls, and five queries. The ten case/arm
observations are not independent repetitions or a general quality estimate.

| Case | Target rank, empty / task prefixes | Seed rank, empty / task prefixes | Bridge rank, empty / task prefixes |
| --- | --- | --- | --- |
| Shared capture | 34 / 28 | 23 / 6 | 36 / 36 |
| Shared extraction | 36 / 36 | 34 / 31 | 32 / 32 |
| Semantic chain | 36 / 32 | 18 / 6 | 1 / 1 |
| Scope isolation | 36 / 36 | 33 / 1 | 25 / 16 |
| Revision and replay fixture | 14 / 7 | 28 / 26 | 35 / 35 |

Ranks are among 36 eligible facts per case. Neither arm puts a target in the
top five. Task prefixes put one authored seed first; empty prefixes put no
seed in the top five. The four foreign controls are excluded by scope in both
arms. Historical graph results remain unchanged. No graph traversal or
lifecycle action ran in this diagnostic, including the revision/replay case.

## Recorded conditions

Source `c31e9710aa7cabd4dd08b1f9076acee7c4d5b28e` committed the
[protocol](../../../plans/2026-09-20-embedding-prefix-development-v1.md)
before inference. The run used loopback Ollama, `nomic-embed-text:latest`,
digest `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`,
768 dimensions, batches of at most 16, `truncate=false`, and no retries.
Each of 40 embedding batches has a matching positive GPU/context observation.
Configured model metadata does not establish an upstream Nomic revision.

The comparison adds `search_query: ` and `search_document: ` exactly once to
the respective roles; the baseline leaves both empty. Both vectors must be
regenerated when changing this contract. Production settings and existing
memory were untouched.

## Verification and limits

The archive retains all 87 raw API calls, before/after model metadata, input
hashes, role-specific vectors, scores, and complete ranks. Independent Python
cosine calculations confirm every rank and all 378 vector bindings against
the raw responses. Ten offline tests pass; strict Python 3.10 type checks and
formatting pass for the four new Python files. The initial live-wrapper test
collection error records that the wrapper did not yet exist; it is not a
reproduced production defect.

These graph-authored queries do not establish general retrieval quality.
Seed improvements justify a separately frozen graph diagnostic using both
query and document vectors. They do not prove graph utility, remedy the prior
failed lifecycle controls, or change T18 acceptance. No held-out labels,
thresholds, gates, live host configuration, or memory data changed.

The index binds raw evidence. This README is outside the index.
