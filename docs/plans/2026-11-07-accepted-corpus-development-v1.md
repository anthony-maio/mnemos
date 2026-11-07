# Query-time accepted-state corpus runner

This runner measures baseline retrieval from declared accepted states. It adds
query-time snapshot binding, complete paired scoring and uncertainty estimates
to the existing accepted-state comparison API. It does not infer accepted facts
from events, execute correction or forgetting, or replace the full-pipeline and
module-ablation evaluation.

## Input contract

`AcceptedCorpus` contains declared cases, session clusters, snapshots and queries.
Every query has an observation time and explicit snapshot ID. It must use the
latest declared snapshot whose cutoff is no later than that time. Duplicate
cutoffs, unknown or future snapshots, stale snapshot selection, naive timestamps,
changed content without a newer fact version, foreign answer labels and unused
snapshots are rejected before provider construction. Snapshot contents retain the
existing source-link, scope, timestamp, metadata and literal-excerpt validation.

Labels and critical-fact IDs stay outside stored facts, source events and embedding
requests. All queries use top-k 5, one declared context budget, full memory results
and the baseline profile. Each snapshot is built once. Document vectors are shared
across its queries and repetitions; each query/repetition supplies one query vector
to vector and hybrid retrieval. Lexical retrieval has no query embedding cost.
Strategy copies isolate access writes and start from identical SQLite backups.

Held-out declarations must meet the original coverage minimums: 200 distinct
session clusters, 100 declared critical facts, 100 ordinary answerable questions,
100 no-answer questions, 50 matched semantic/state pairs, three repetitions, all
three supported host labels and multiple queried projects. Development fixtures
report deficits and cannot clear these requirements. These checks validate counts
and structure; they do not prove unseen data, semantic correctness, human review,
native-host coverage or actual retained critical-fact performance.

A matched state pair has identical query text, snapshot, time, scope, labels and
budget, with two distinct declared state conditions. The three baseline strategies
ignore that state condition by design. Record these as controls; they do not
measure a state-aware treatment or satisfy the separate full-module ablation.

## Execution and scoring

`run_accepted_corpus` requires a new output directory, a provider factory, explicit
dimension/provider identity and bound input/source files. It validates before
constructing the provider. It records the canonical corpus, options, file hashes,
raw results and terminal status. It rejects reuse, checks file drift before each
query and closes owned stores on cancellation or error. There is no implicit
resume or retry. A provider-startup failure leaves a failed control record. A
snapshot-building failure supplies explicit failed observations for every planned
query and strategy at that snapshot.

`summarize_corpus` requires exactly the planned case/query/repetition/strategy
matrix. It rejects duplicate, missing, unplanned or differently bound rows. It
checks common snapshot/document-vector bindings across queries and repetitions,
and common query vectors within each vector/hybrid pair. It recomputes grades
from raw IDs, content, versions and scope instead of trusting cached grade fields.
Operational failures remain in the denominator and cannot earn abstention credit.

Report any-answer hits separately from clean hits, fractional recall, unsupported
selections, stale returns and scope leaks. Attach Wilson intervals per repetition
using that repetition's unique questions. Do not pool repeated questions into a
larger independent Wilson sample. Queries within a session may still be dependent;
these Wilson intervals are descriptive. Paired strategy differences use whole
session-cluster resampling, retaining all queries and repetitions in each selected
cluster, with observation-weighted estimates. Report both hit/abstention differences
and fractional-recall differences. Use 95% intervals, 10,000 bootstrap draws and
seed 20261107. No result from this API alone claims the overall quality gate.

## Frozen development screen

Derive one synthetic development case from the existing accepted-retrieval fixture
and the new corpus tests. Declare three states: production retention at thirty
days, a version-2 correction to forty days, and removal of that fact. The first
two snapshots have six facts each; the final snapshot has five. Source links are
updated explicitly in the fixture. This is declared expected state, not evidence
that ingestion or forgetting produced it.

Ask the production-retention question before and after the correction and after
removal, plus an unknown-region question after correction. Two questions have
expected answers and two do not. Run three repetitions and three strategies:
36 result rows, 17 document embedding calls and 12 query embedding calls. Expect
the correct declared version at each answerable point and no removed fact after
its declared removal. Record all other retrieval errors without changing labels.
The single cluster makes paired intervals degenerate when differences are stable;
this screen validates the machinery, not population uncertainty.

Use local FastEmbed 0.9.1, `BAAI/bge-small-en-v1.5`, 384 dimensions and the existing
verified embedding cache in offline mode. Freeze the clean source commit, generated
fixture hash, runner hash and cache-asset hashes before inference. Preserve every
role-specific embedding input/vector. Audit raw results and original SQLite state
after the run without another model call. The unused fresh corpus remains unopened.
