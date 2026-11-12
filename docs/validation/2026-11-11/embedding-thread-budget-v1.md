# Local embedding thread budget: opt-in only

The single-thread proposal failed its preregistered recall non-regression gate.
It eliminated measured idle CPU use and preserved all vectors and recall results,
but its ordinary-recall p95 was 161.43 ms versus 150.04 ms for baseline. Do not
adopt it as the default or describe it as a latency improvement.

The final canonical source at `7497d561` preserves upstream automatic allocation
by default. It retains an explicit resource control: `embedding.threads` or
`MNEMOS_EMBEDDING_THREADS`, with zero selecting upstream automatic allocation
and positive integers setting the FastEmbed thread budget. One is an opt-in
choice for reducing idle CPU use. Batch throughput and latency tradeoffs still
depend on the workload. Booleans, negatives, null and nonintegers are rejected.
No model, normalization, role prefix, embedding identity or other provider changed.
This setting is source-only; it is not in the latest private 9c326cbc wheel.
No live setting or pilot installation changed.

The initial installed-provider diagnostic used one cached BGE embedding. It
took 6.1142 ms, then consumed 5,796.875 ms of process CPU while sleeping for
approximately one second without requests. Idle CPU before session creation
and after the worker activity settled was zero. CPU time sums across threads;
it is not an equivalent wall-clock delay or proof of the earlier support tail.
ONNX documents automatic physical-core allocation and enabled spinning;
FastEmbed already exposes a public threads argument, including at version 0.4.0.

The frozen comparison used baseline 9c326cbc and proposed-default source
`69efa81eeaf639e0a4b4467cf97b662f2ed938b5`, with runner `344b5b9c`.
Fresh processes shared the same 57 dependency versions, 17 cached assets and
private identical copies of the 10,000-accepted-fact fixture. Eight synthetic
texts were embedded as a batch and individually. All 16 compared vectors were
bit identical: maximum absolute difference zero and cosine one.

| Measurement | Automatic baseline | One-thread proposal |
| --- | ---: | ---: |
| Process CPU during first 500 ms idle after embedding | 5,328.125 ms | 0 ms |
| Complete ordinary-recall median | 139.40 ms | 126.44 ms |
| Complete ordinary-recall interpolated p95 | 150.04 ms | 161.43 ms |
| Measured recalls | 50 | 50 |

Both arms ran the same two passes over 25 retained measured requests, after
three warmups. All 50 complete result pairs matched, including scores, facts,
versions, retained sources, scope and context. The complete client clock includes
BGE, scoped retrieval, access writes, coherent source assembly, transport and
decoding. No concurrent capture operations were replayed. This is read-only
ordinary recall with no support classifier; it does not replace the earlier
interleaved command workload. Neither arm
establishes the required full support-enabled 500 ms result or semantic quality.
The closed resident support configuration was not run again.

The comparison passed numerical parity, idle CPU reduction and result-pair
agreement, but failed the frozen requirement that candidate p95 not exceed
baseline. The failure remains closed; there was no inference retry or parameter
sweep. A separate offline reconstruction verified the vector/result agreement,
all sample counts, percentiles and terminal receipts without more inference.
The source default was restored only after preserving the executed candidate.

The Windows-owned attempt completed in 29.98 seconds with all 100 measured calls
and zero owned survivors. Exit code one reflects the failed gate. All 4,458
bindings matched at terminal postflight, before the subsequent default
restoration. Both databases passed integrity and foreign-key checks. Fifty-eight
config/runtime/embedding tests pass on both the proposal and final opt-in form.
The bounded source/protocol review found no blocker and did not execute tests
or inference; it explicitly left adoption gated on this comparison.

One preparation failed before inference because the retained log contained 25
measured recalls, not 50. Before freezing the successful preparation, the
protocol was corrected to use two explicit passes over those 25 requests.
No measured run was repeated and no original evidence was rewritten.

Private evidence: `.venv/embedding-idle-v1` and
`.venv/embedding-thread-comparison-prepared-v1`. Archive:
`.venv/embedding-thread-comparison-v1-private.zip`, 234 members,
123,928,279 bytes; 232 included file hashes verified on readback. SHA-256:
`1b1380146f4bc2a451e436d702b4c993c391b6abafc2e039201dcb926d78216b`.
The archive preserves the candidate source and original protocol before
post-run changes. Dependencies and original seed data retain external hash
bindings. Raw databases remain local.

Primary references:
[ONNX thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html),
[FastEmbed 0.4.0 constructor](https://github.com/qdrant/fastembed/blob/v0.4.0/fastembed/text/text_embedding.py).
