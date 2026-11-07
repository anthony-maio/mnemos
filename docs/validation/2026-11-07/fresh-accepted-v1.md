# Fresh accepted-state retrieval comparison

Status: running. No completed quality result is available yet.

The v7 authored corpus has now been opened for its first retrieval-only model
evaluation. Conversion passed with the unchanged adapter: 200 held-out sessions,
700 retained supported facts, 200 ordinary answerable questions, 220 no-answer
questions and 80 state pairs. The twenty development sessions are excluded from
model inputs. Synthetic and unreviewed labels remain intact.

The frozen comparison runs all 580 query strings three times against vector,
lexical and hybrid baselines, for 5,220 planned observations. It uses the same
declared accepted states, document vectors, scope permissions and context budgets
for each arm. Vector and hybrid share query vectors. The two state-pair wordings
remain distinct; their results cannot isolate a state-module effect.

Executed source commit: `291c6fcc8fb206400a902e43579691242dd110f8`.
Original source SHA-256:
`b129004eacc1364dce6b8ab069a0b2b9cdf5604543aae003b8ea9f1e42a74fec`.
Converted canonical corpus SHA-256:
`d930e6c2e804074e98310b8d67efdc361c1f1db214d912309cce04e71d04f716`.

Uses offline FastEmbed 0.9.1 with `BAAI/bge-small-en-v1.5`, 384 dimensions,
seventeen verified cache assets, top-k 5 and the existing 1,200-token conservative
byte budget. All cognition modules and answer support are disabled. This tests
retrieval on supplied gold states; it does not execute extraction, capture,
correction, forgetting or replay. The original thresholds remain unchanged.

Completion requires the full planned matrix, independent raw-result/vector/state
audit and a verified archive. Partial observations are not a quality pass.
Repeated synthetic families, short sessions and absent human acceptance limit
what even the completed comparison can establish. The corpus is now evaluated
input; later reuse must be described accordingly.

Production source and live configuration are unchanged. The full-module
comparison, remaining native/scale checks, delivery alignment and the actual
two-week pilot remain open. The pilot is inactive at zero days.

- [Frozen protocol](../../plans/2026-11-07-fresh-accepted-retrieval-v1.md)
- [Adapter development validation](accepted-adapter-v1.md)
