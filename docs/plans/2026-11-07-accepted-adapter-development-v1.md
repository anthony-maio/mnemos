# Terminal-gold adapter development protocol

Bridge the existing v4/v5/v6 evaluation contract to the accepted-state retrieval
runner. The original evaluator processes the complete ordered event stream before
asking its ordinary queries and state-pair queries. The adapter must preserve that
phase. It cannot infer earlier snapshots from final gold labels. Earlier query
phases require explicit query-time states through `AcceptedCorpus`.

## Conversion contract

Require an explicit development or held-out split and `after_all_events` phase.
Keep source sessions, fact IDs, fact contents, query IDs, query texts, answer IDs,
target scopes and retained evidence text intact. Preserve original synthetic and
unreviewed labels. Use the original parsing, lifecycle-target and source/scope
validation. Source-file-backed objects must match a freshly validated read of
their bound file. Held-out conversion requires such a file and the full original
dataset minimums. Development conversion may use small authored contract objects;
that does not lower the held-out requirements.

Select terminal `supported` and `conflict` gold facts. Mark conflicts explicitly.
Exclude superseded, forgotten and rejected gold facts from the declared current
snapshot. Keep each retained evidence event once, with its full literal text,
original event identity and declared scope. Refuse ambiguous source partitions,
missing or one-way evidence links and replay-delivery evidence IDs. A replay must
refer to its original event; it does not reintroduce a forgotten fact merely by
appearing later in the input.

Gold fact IDs start at declared version 1. They are fixture identities, not an
observation of the stable IDs or versions produced by a correction workflow.
Source timestamps determine fact creation/update bounds. Set the snapshot cutoff
to the final event time and the synthetic query time one microsecond later. This
anchors the known evaluation phase; it is not an observed wall-clock query time.

Use the legacy retrieval permissions, `project` and `global`, by default. Do not
silently grant workspace access. A workspace answer outside those permissions
must fail validation; an explicitly amended scope list is recorded in provenance.
Project and workspace identities cannot belong to conflicting user profiles.

## Preserve state-pair wording

The original state pairs have distinct semantic and state query strings. Keep
both exactly. Add the explicit `source_wordings` mode in accepted-corpus schema v2.
Require shared snapshots, time, scope, labels and budget while retaining each
variant's original query text. The earlier v1 fixed-query control remains the
default, with its canonical serialization unchanged.

Different query wordings may change embeddings and rankings even with all state
modules disabled. Declared state labels are metadata here, not classifier output
or an injected model state. Do not claim an isolated affective effect. The full
pipeline and the existing fixed-embedding state diagnostic remain separate.

## Frozen development execution

Use two newly authored v6 development sessions with fourteen events. They cover
an explicit correction, a forget and original-source replay, two conflicting
ownership claims, a rejected assistant claim, a global preference, a foreign
profile/project, a workspace fact and restart boundaries. Convert ten gold facts
into seven declared current facts: five supported and two conflicts. The other
three are superseded, forgotten and rejected. These labels are supplied inputs;
the lifecycle and extraction models do not run.

Run nine queries: four ordinary answerable, three no-answer, and the two distinct
wordings of one matched pair. Use three repetitions, three baseline retrieval
strategies, top-k 5 and a 1,200 UTF-8-byte conservative context budget. Expect 81
result rows, seven document embedding calls and 27 query embedding calls. Preserve
all negative results and do not tune labels or thresholds after the run.

Use offline FastEmbed 0.9.1 with `BAAI/bge-small-en-v1.5`, 384 dimensions and the
existing verified local cache. Freeze the source object, converted corpus,
provenance, clean source commit, runner hash and model-asset hashes before model
calls. Reconstruct conversion and role-specific vectors from saved artifacts,
check raw returned IDs/content/versions/scopes, and inspect both original SQLite
snapshots for integrity and unchanged contents after the run. Audits make no
additional model calls.

Existing v4 development data is used only for offline conversion and file-drift
tests. The unused fresh corpus stays unopened and unexecuted during this adapter
development. This screen cannot establish data freshness, human label acceptance,
extraction fidelity, lifecycle correctness, native coverage, scale latency or
pilot readiness.
