# Local memory performance validation

This run measured the proposed latency gates on an isolated temporary SQLite
database. It loaded 10,000 accepted `MemoryChunk` rows with 768 dimensional
vectors, then ran the scoped baseline retrieval path. The vectors were
distinct synthetic perturbations derived from one observed embedding from
`nomic-embed-text:latest`; that fixture exercises storage and ranking at the
target size, while semantic quality remains NOT RUN.

The run used Python 3.13.7 on Windows 11, an AMD64 machine with 20 logical
CPUs, and the lab Ollama endpoint at `http://192.168.86.35:11434`. The
embedding digest was
`0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`. The
discovered cognition model `gemma3:4b` had digest
`a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a`, but no
cognition or extraction calls were made.

The 10,000 row database occupied 324,862,864 bytes after the load and index
build. Fixture loading took 90,340.328 ms. After three warmup queries, 33
queries included a real Ollama query embedding followed by scoped SQLite
candidate selection and baseline ranking. The embedding p95 was 241.218 ms;
the scoped SQL and ranking p95 was 117.791 ms; and the full warm retrieval p95
was 506.111 ms, against the 500 ms gate. The gate therefore FAILS narrowly on
this machine. The lexical candidate p95 was 157.990 ms and mark-plus-refresh
p95 was 290.286 ms. The fixture returned five ranked rows on average, including
the observed-seed query, but its vectors are synthetic and no retrieval quality
claim follows from this run. The raw artifact separates these stages for
follow-up optimization. This artifact uses the batched access path; the prior
pre-batch result remains a preserved negative comparison.

Fifty warm HTTP captures were accepted with status `committed` by an isolated
authenticated loopback service after one warmup request. The client reused one
HTTP client pool on its owning event loop; the server closes individual
responses after each dispatch. Capture p95 was 68.228 ms, against the 100 ms
durable capture gate, so that gate PASSES on this machine. The warm request took
278.027 ms, which records one-time client setup separately from the 50-request
distribution. A subsequent full maintenance drain completed all 51 durable
jobs, including the warmup event, in 4,779.109 ms. A separate synthetic cold
Codex adapter subprocess took 944.559 ms. It exercised the adapter against the
temporary service; a real Codex session and installed-host smoke remain NOT
RUN.

The corrected raw versioned artifact is [performance-10k.json](performance-10k.json).
It records source hashes, configuration flags, raw per-query stages, and the
observed SQLite WAL and synchronous FULL settings. The regular float32 cache
adds storage bytes to this artifact while avoiding JSON vector parsing during
scoped ranking.
The earlier provider wiring mistake is preserved in
[performance-10k-invalid-provider-baseline.json](performance-10k-invalid-provider-baseline.json)
and must not be used as performance evidence.
The runner is [performance.py](../../../benchmarks/local_memory/performance.py).
These results are a negative performance baseline and a measurement of this
machine, endpoint, provider, and implementation snapshot. They do not close
the release gate or establish semantic retrieval quality.

## Instrumented diagnostic at 996e9ab

A later run from detached `996e9ab` added timing around the store methods used
inside each actual retrieval call. The source hashes remained consistent
throughout the run. The original follow-up probes remain available, but they
request 120 candidates; actual retrieval requests 30. Those probe timings must
not be presented as the actual call's component timings.

The instrumented run measured warm retrieval p95 **458.516 ms**, passing the
500 ms point-estimate gate for that run. Its maximum was 500.164 ms. This was a
different source snapshot and distribution, without a deliberate retrieval
latency optimization. It does not establish an improvement over the preserved
506.111 ms failure or certify the subsequent graph repair.

| Actual stage | p95 ms |
| --- | ---: |
| Query embedding | 224.190 |
| Scoped semantic selection, 30 candidates | 59.168 |
| Lexical selection, up to 30 candidates | 130.450 |
| Durable access update | 115.447 |
| Read current returned facts | 1.367 |

The stage distributions are measured within full retrieval calls and their
percentiles must not be added. Thirteen queries returned 30 lexical candidates
and averaged 444.39 ms overall; twenty returned none and averaged 322.57 ms.
Lexical work and durable commits explain much of the observed variation.
Capture again passed its existing threshold. No cognition or extraction calls
were made, and semantic quality remains unmeasured for this synthetic load.

The raw record is [performance-10k-instrumented-v1.json](performance-10k-instrumented-v1.json).
Candidate limits, matching, and durable access updates remain unchanged. Any
future concurrency optimization needs a separate controlled experiment that
preserves correction, forgetting, and scope behavior.

## Frozen graph v4 measurement at b7a9d9d

The final-source measurement completed with consistent source hashes and
preserved SQLite WAL/FULL durability. Warm capture passed at **57.410 ms p95**
across 50 events. Warm recall failed at **1,314.583 ms p95** across 33 queries
against 10,000 facts. Median recall was 346.599 ms; maximum was 1,359.907 ms.
The synthetic load used 324,862,864 bytes after insertion.

Two actual retrieval calls exceeded 500 ms. Query 1 spent 1,083.845 ms inside
`mark_retrieved_many` out of 1,359.907 ms overall; query 4 spent 924.395 ms
there out of 1,314.583 ms overall. Their query embeddings took 206.406 and
202.829 ms respectively. These observed stalls occurred while recording
retrieval metadata. The trace does not identify whether locking, checkpointing,
filesystem scheduling, or another operation caused them.

This result supersedes the prior diagnostic for the current source's latency
gate. The earlier passing and failing runs remain preserved; no result is
discarded, and no durability setting or threshold was changed to obtain a
passing number. Follow-up should isolate the metadata-write cost while
preserving correction/forget ordering and durable behavior. Semantic quality
remains unmeasured by this load test.

Raw query timings and provider/source identity are in
[real-model-v3/performance-10k.json](real-model-v3/performance-10k.json).
