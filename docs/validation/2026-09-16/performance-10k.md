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
