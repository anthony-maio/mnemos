# Installed performance V1: fixture startup failure

V1 stopped before collecting timed retrieval or hook observations. Its fixture
created 10,000 real document embeddings, 10,000 facts, 10,000 source events and
10,000 evidence links. The installed service then refused the database because
the fixture loader had omitted its embedding-identity metadata.

This is a benchmark setup error. The product's refusal prevents unverified
vectors from silently entering a configured embedding space; that guard remains
unchanged. No warm-recall, hook-latency or native-host result was obtained.

The attempt used the installed `749f1213` wheel and completed fixture embedding
and loading in 31.49 seconds, separate from model construction. Its command
exited with code 1. Both owned processes are stopped and its runtime descriptor
is absent. The original database, logs, model-call batch records and bound
inputs are preserved in a verified 110-member private archive, SHA-256
`6edad8f0c3298ecbfff427ddc16d2b9bf13a802a2f67b39f72ec6da002e84029`.

The [V2 amendment](../../plans/2026-11-07-installed-performance-v2.md) creates
an empty database through the installed runtime's configured store factory
before inserting any vectors. A model-free regression probe reproduced V1's
rejection, then verified that the configured empty store retains its identity
after insertion and reopening. It reused one recorded real vector and made
zero new embedding calls. The V2 wrapper hash is
`dff8eeb91cf9e06e35e38b661d350872a47001c0f77a999674efd2c50789e21c`.

V2 will use a new disposable runtime with the original 10,000-fact workload,
sample counts, complete command timers and 500 ms/100 ms thresholds. The failed
V1 database will not be changed or counted as a completed measurement. Product
source, live configuration and the inactive pilot are unchanged.

- [Failure receipt](installed-performance-v1-failure-receipt.json)
- [Original protocol](../../plans/2026-11-07-installed-performance-v1.md)
- [Preparation record](installed-performance-v1-preparation.md)
