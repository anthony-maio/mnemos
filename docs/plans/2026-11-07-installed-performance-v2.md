# Installed performance check: fixture identity repair

This amendment retains the [v1 protocol](2026-11-07-installed-performance-v1.md),
installed wheel, model, settings, fact/query content, sample counts, timers and
original latency thresholds. Use a new private runtime and preserve the failed
v1 attempt and its archive.

V1 created all 10,000 real document embeddings and source-backed fixture facts,
but its direct `SQLiteStore` constructor omitted the embedding-identity binding.
The installed service correctly refused to open existing unverified vectors.
No warm retrieval or hook timing was collected. The failure does not establish
a product startup defect or a performance result.

Before embedding or inserting facts, V2 creates the empty database through
`runtime._configured_store`, the same factory called by the installed runtime.
Verify its stored identity against `runtime.embedding_identity` and require an
empty fact table. Then create the real embeddings and load the same declared
synthetic facts. Keep the product's identity guard unchanged. Do not repair,
relabel or reuse the failed attempt's database.

A model-free regression probe must reproduce rejection for populated unbound
storage and confirm that an empty configured store retains its identity after
inserting a recorded real vector and reopening through the same factory.
That probe is a fixture-contract check, not timed service or native-host evidence.

Record V2 independently, including any new bootstrap failure. Repeated use of
this synthetic performance workload is not fresh semantic-quality evaluation.
