# Accepted-state retrieval development screen

This prospective screen checks the new extraction-independent benchmark path.
It uses six newly authored synthetic facts and seven development queries. It
does not open the unused fresh corpus, establish pilot readiness, or replace
the original extraction, retrieval, latency or daily-use acceptance gates.

The input is a declared terminal snapshot with versioned facts, retained source
events, literal evidence excerpts, scope identities and an explicit observation
cutoff. It is not the output of a measured extraction or correction lifecycle.
Metadata excludes expected-answer labels. Source links, scope partitions,
timestamps and identities must validate before document embedding or database
creation. Literal evidence validation does not prove semantic entailment.

The builder creates only a new SQLite file. It runs no extraction or maintenance
jobs. All five experimental modules, affective/spreading retrieval and query-state
inference are off. Documents are embedded once with the document role. Each
query/repetition gets one query-role vector shared by vector and hybrid arms;
lexical retrieval does not use it. Each arm starts from an identical committed
SQLite backup and has its own access writes. The original accepted state must
remain unchanged. Strategy order is deterministically shuffled per query and
repetition. The full-module ablation remains a separate evaluation.

Use FastEmbed 0.9.1 with local `BAAI/bge-small-en-v1.5`, 384 dimensions and the
existing private model cache, with Hugging Face offline mode selected. Record
the cache asset hashes, dependency version, provider load time, every embedding
input/vector and every retrieval result. The builder must not read a lazy
provider dimension property that could cause an unrecorded inference call.

Run three repetitions, three strategies and seven queries: 63 observations,
with six document and 21 query embedding calls. Each arm uses top-k 5, a 1,200
UTF-8-byte conservative context budget and explicit project-only scope. Four
queries identify one accepted fact; three ask for unknown, conflicting or
foreign-project information. Four asserted facts and two conflicting assertions
remain distinct. The same foreign fact is queried from both project contexts.

Score expected-ID overlap separately from correct empty-result abstention.
Record any-answer hit, fractional recall, precision among returned IDs,
nonrelevant returned IDs, scope leaks and operational failures. An embedding
failure cannot earn abstention credit; lexical retrieval may still complete.
A returned conflict retains its qualification but does not satisfy this
protocol's empty-result abstention criterion. These relevance labels do not
measure extraction precision or generated-answer faithfulness.

Record query embedding time separately from retrieval dispatch time. Shared
embedding, SQLite cloning and source-state validation make this a controlled
comparison, not the native-host or 10,000-fact warm-latency benchmark. Repeated
queries do not create independent semantic cases. Report negative results as
observed, without tuning thresholds, changing labels or rerunning a failed
attempt under the same identity.

Frozen fixture SHA-256:
`d915df29dcaad59455a890a365f8307a11805f315afc28521dfbcf0204ffeca7`.
Frozen private runner SHA-256:
`9faef21da029094668861baa558f302d7759338a04d310a69811f73f2e56e36b`.
The source commit must be clean and recorded before execution. Preserve the
original attempt and all errors. No live host, pilot installation, memory
database or published artifact is changed by this screen.
