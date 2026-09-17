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

Source review localized the operation to a primary-key access-counter/timestamp
update followed by commit. It does not rebuild vectors, FTS, or graph state;
the FTS trigger only fires on content updates. The embedding is stored in the
same table row, so large-row WAL writes and automatic checkpoint/flush cost
are plausible explanations, not measured causes. The prior same-size run's
maximum metadata-update time was 151.374 ms, which also argues against
claiming a repeatable query-plan regression from these two stalls.

The next bounded diagnostic is an offline comparison of 10,000-row fixtures
with 768-dimensional versus absent embeddings, preserving five-ID updates and
WAL/FULL commits. Record page size, checkpoint configuration, database/WAL
bytes, and per-update timing, then inspect checkpoint state after the timed
operations. Run this outside the active evaluation to avoid competing disk
load. A separate metadata table is only a candidate if the probe confirms
large-row write amplification; it would require migration and consistency
tests. Deferred commits and reduced durability are not part of this repair.

## Duplicate benchmark write found during source review

The v1 runner performs a second `mark_retrieved_many` call and five reads
after each timed retrieval. That extra WAL/FULL commit is outside the recorded
latency, but it changes the database before the next timed query. Additional
semantic and lexical diagnostic reads also run between measured queries. The
measurement therefore includes an interleaved diagnostic workload, not just
the production retrieval sequence.

This distinction is visible in the frozen artifact: query 0's separate
mark-plus-refresh probe took 1,145.564 ms while its actual engine metadata
update took 49.853 ms. Queries 1 and 4 still contain the actual engine stalls
reported above. The extra writes are a confound, not a proven explanation for
those stalls, and the failing result remains preserved.

Before the next latency comparison, remove duplicate mutations from the timed
query sequence and derive component timings from the existing actual-call
trace. Keep any standalone diagnostics in a separate phase or disposable
fixture. Then rerun the unchanged 500 ms gate with WAL/FULL durability and
record the instrumentation change. No product storage optimization is justified
by the current evidence alone. This corrected measurement has not run.

The benchmark-only correction is committed as `554a251`. Its v2 artifact
schema identifies the changed measurement method and the actual 30-candidate
engine request. Four focused tests pass, including a small offline fixture
through `measure_retrieval` that checks one metadata update per engine query.

## Corrected measurement at d94ded6

The v2 measurement completed with consistent frozen source hashes and no
follow-up store calls between timed retrievals. Warm recall still fails:
**1,088.094 ms p95** across 33 queries at 10,000 facts, against the unchanged
500 ms gate. Median recall was 459.713 ms and maximum was 1,131.605 ms.
Durable HTTP capture passed at **64.279 ms p95** across 50 events. SQLite
remained WAL/FULL, with 324,862,864 bytes after fixture insertion.

Within those actual retrieval calls, metadata update p95 was 795.750 ms.
Embedding p95 was 257.696 ms, scoped semantic selection 74.595 ms, and lexical
selection 160.371 ms. These nested stage percentiles must not be added. The
remaining metadata stalls show that removing duplicate diagnostic writes did
not eliminate the latency failure; they do not by themselves identify a cause.

Raw evidence: [v4 performance record](performance-corrected-v4/performance-10k.json).
This remains a synthetic 10,000-fact load with real query embeddings, not an
extraction-quality or complete native-host measurement. The next diagnostic
compares matched metadata updates with and without large embedding payloads
while preserving WAL/FULL commits. No storage optimization is accepted yet.
