# One explicit CPU/GPU resource-policy capacity attempt

This is an explicit amendment permitting one changed runtime configuration.
It revisits the same ModernBERT encoder; it is not a new neural architecture.
The component 200 ms failure, complete-path 500 ms failure and ordinary-recall
non-regression failure all remain failures. Their original evidence is frozen.

New evidence establishes that the default embedding pool consumes substantial
idle CPU, while an implemented explicit one-thread policy eliminates measured
idle use with identical vectors and recall results. It did not improve ordinary
recall p95 and was not adopted as the default. Whether it benefits concurrent
CPU embeddings and the GPU support worker remains an untested hypothesis.
The bounded adversarial review considers one absolute-admission attempt
justified. It does not establish that contention caused the earlier failure.

Use committed canonical source with `embedding.provider=fastembed` and explicit
`embedding.threads=1`, through the actual settings/runtime builder. Keep the
product default at automatic allocation. Keep the GPU worker's original
22-layer checkpoint, weights, random head initialization, dtype, SDPA, thread
settings, TF32 setting, limits and trusted framing unchanged. Keep native and
model dependency inventories unchanged. No Jev, new model, training, live
configuration or pilot activation is included.

Reuse exact copies of the six frozen prepared stores, requests and expected
frames from the failed resident attempt. Revalidate the actual selected cohort
and owned evidence on the current source before freezing. These remain
10,000-accepted-fact synthetic stores. No source is padded, shortened, omitted
or substituted for this attempt. Keep the 8,192-token and 64,000-byte bounds,
including complete framing. The six workloads cover up to eight owned sources
per fact, not every source inventory the serializer could accept.

Preserve the original complete client.retrieve clock after resident setup:
real BGE, scoped retrieval, context/evidence assembly, RPC and queueing, complete
tokenization, GPU forward, synchronized finite CPU readback, current registry
authorization and fact/evidence/source revalidation, serialization and decoding.
Keep all facts and sources regardless of random head outputs. Unavailable,
stale, malformed, oversize, nonfinite or binding failures fail the whole attempt.
This does not replace the separate capture clock or normal-hook workload.

Use two fresh service/model blocks, three warmups per workload per block and
25 round-robin measured calls per workload per block. Require all 300 measured
calls and 36 warmups, with 280 forwards, and original inclusive interpolated
p95 <=500 ms for every workload. Preserve missing-call denominators. Run once
in a Windows-owned job bounded to 580 seconds plus cleanup within 600 seconds.
No thread-count sweep, changed warmups, model/backend/bound adjustment or retry
follows a failure. Record cold setup separately and verify all owned processes
stop and frozen inputs remain unchanged.

This is candidate-only. A passing absolute clock would establish computational
feasibility for this exact configuration and permit a separate learning/data
plan review. It would authorize neither training nor semantic acceptance.
Historical timings are context, not a controlled performance comparison. No
claim that this policy improved the old combined path or caused its failure
is permitted without new paired controls; no such controls are run here.

The original quality, fresh-acceptance, complete timing, native/pilot and
publication requirements remain. Local operation remains required, with Jev
explicitly optional and no automatic fallback.
