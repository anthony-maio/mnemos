# Fresh accepted-state retrieval comparison

Freeze this protocol before opening the v7 authored input for this evaluation.
The preparation record identifies schema `local-memory-heldout-v6`, 200 held-out
sessions, 200 ordinary answerable questions, 220 no-answer questions and 80 state
pairs. The source bytes are bound by SHA-256
`b129004eacc1364dce6b8ab069a0b2b9cdf5604543aae003b8ea9f1e42a74fec`.
Authoring revision v7 is distinct from schema version v6.

## Fixed input and execution

Use the unchanged terminal-gold adapter implemented at
`1c8962d1c4c7d3a3d3030996fa96cfaaec4ebcef`. Read the source blob from the repository
into a new private input directory, verify its recorded hash, then validate the
original contract and convert only the held-out split. Retain synthetic and
unreviewed labels, original fact identities, full source evidence, query wording
and project/global scope permissions. Use only the explicit `after_all_events`
query phase. The twenty development sessions are not model inputs.

Keep supported and conflict terminal facts, with conflict qualifications intact.
Exclude superseded, forgotten and rejected gold facts. Versions begin at declared
version 1. This is a comparison on supplied accepted states, not measured capture,
extraction, correction, forgetting or replay. Do not seed an ingestion pipeline
and report its assumed success. If conversion fails, preserve the failure before
model calls; do not amend labels, timing, permissions or minimums to force a run.

Use local FastEmbed 0.9.1, `BAAI/bge-small-en-v1.5`, 384 dimensions, offline loading
and the seventeen cache assets already verified in the adapter development run.
Bind all product and local-memory benchmark Python sources, adapter tests, the
protocol, wrapper, source bytes, converted corpus and conversion provenance.
Record the clean source commit, hardware, model assets and package versions.

Run vector, lexical and hybrid baseline retrieval three times per query. Each
arm uses the same accepted SQLite state, document vectors, scope and context
budget. Vector and hybrid share one query vector; lexical uses no query vector.
Keep the existing implementation's relevance floor of 0.2, top-k 5, full qualified
records and 1,200-token conservative UTF-8-byte budget. Disable all cognition
modules, extraction, query-state classification and answer support. Use the
existing deterministic shuffled arm order. No threshold or prompt tuning follows
inspection of these results.

The declared counts imply 580 query strings, 1,740 query embedding calls and
5,220 result observations. The preparation record describes 700 retained critical
supported facts; verify the full retained fact count during conversion rather
than silently excluding additional retained conflict facts. Document embedding
calls must equal the number of retained facts across the two hundred snapshots.
Keep all planned questions and failures in the denominator. A failed or degraded
response never earns correct-abstention credit. Do not retry failed quality calls
or overwrite an attempt. A fatal interruption remains an incomplete attempt.

## Scoring and audit

Report ordinary answerable and no-answer results separately from both state-pair
variants. Report any-answer hits, fractional Recall@5, clean hits, nonrelevant
selections, operational failures, scope leaks and stale returns. Retain the
original targets of at least 90% answerable Recall@5 and 90% correct abstention.
Expected-ID relevance and literal retained evidence are not a human semantic
precision judgment. This run cannot accept an answer-support mode or extraction
quality even if a retrieval threshold passes.

Report Wilson proportion intervals for unique questions within each repetition.
Use the existing paired whole-session bootstrap with 10,000 samples, confidence
0.95 and seed 20261107 for lexical/vector and hybrid/vector hit or abstention
differences and fractional-recall differences. Repetitions are not independent
new questions. Report descriptive strata by source host label, scenario tag and
declared snapshot fact count. Tags may overlap; do not sum them as disjoint groups.
Record eligible candidate counts separately from whole-snapshot fact counts.

Keep both original state-pair wordings. They may produce different embeddings;
all three baselines ignore declared state labels. These results cannot isolate
a state-module effect or establish graph benefit. The existing fixed-embedding
diagnostic and full-module treatment remain separate obligations.

Audit the bound file hashes, source-to-corpus fields, all recorded document/query
vectors, complete raw-result matrix, returned IDs/content/versions/scopes/evidence
and unchanged source database contents. Recompute counts and intervals from raw
results without model calls. Preserve a verified private archive and publish only
synthetic/aggregate evidence in the local report. Record run and per-call timing
descriptively; this is not a native-host or 10,000-fact latency test.

The preparation review already notes repeated release-policy competitors, three
correction families, concentrated state pairs and short sessions. No human
acceptance, real-conversation coverage, fully blinded authoring or exhaustive
semantic-independence claim follows from the word "fresh." After this run the
input is evaluated data; subsequent use must be labeled accordingly.

No live configuration, pilot activation, release or publication is part of this
comparison. Those obligations remain governed by their existing requirements.
