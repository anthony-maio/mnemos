# Per-query benchmark evidence

Benchmark results now include `retrieval_evidence` with schema version 1. This records the dataset-document to stored-chunk mapping, a snapshot of stored chunk metadata before querying, and one observation per query in dataset order. Repeated queries retain separate observations.

Each query observation contains its query-text SHA-256, scope inputs, UTC start time, measured latency, relevant chunk IDs, returned IDs in rank order and metadata copied from the returned chunks. Chunk metadata includes a content hash, creation time, legacy activity time, explicit content-update time, the effective content-recency timestamp, version and access count. Unknown content-update time remains null. Hashes identify text without duplicating its content in the report.

The saved returned/relevant IDs and latencies are the same observations used to compute recall, MRR and latency statistics. Tests reconstruct those metrics from the serialized report for both memory and SQLite stores, with baseline and engine retrieval. Copies of timestamp values prevent later in-memory accesses from rewriting earlier evidence.

Evidence serialization runs outside each retrieval timer. The pre-query store snapshot is also outside that timer, but reading it can warm storage and serialization adds time between queries. This is a changed benchmark harness, not a claim of timing equivalence with older runs. It does not freeze wall-clock time, alter ranking, change fixtures or relax thresholds. Scores inside the retrieval pipeline are not recorded, so these observations alone do not prove why every ranking decision occurred.

The earlier failed prerelease benchmark remains failed. Its missing IDs and timestamps cannot be reconstructed by applying this report format afterward. Any future execution must retain its own source, artifact, fixture and environment identity and must not replace the original result.
