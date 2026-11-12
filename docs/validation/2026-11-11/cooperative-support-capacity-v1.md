# Explicit CPU-budget support computation: capacity passed

The sole amended resource-policy attempt passed the original warm-resident
500 ms p95 gate for all six frozen workloads. It completed all 300 measured
calls, 36 warmups and 280 encoder forwards across two fresh service/model
blocks. The highest workload p95 was 386.35 ms. One measured call took
619.25 ms, so this is not a claim that every call finished within 500 ms.
Cold setup and warmups are excluded from the warm recall clock.

| Workload | Tokens | Measured calls | Median ms | Interpolated p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Empty scoped cohort | 47 | 50 | 29.50 | 47.53 |
| One short fact | 142 | 50 | 172.63 | 201.15 |
| Five short facts | 520 | 50 | 181.12 | 224.33 |
| Full token bound, one source per fact | 8,192 | 50 | 353.10 | 386.35 |
| Full token bound, eight sources per fact | 8,192 | 50 | 358.54 | 377.67 |
| Exact 64,000-byte payload | 3,112 | 50 | 210.77 | 251.54 |

This revisits the same encoder under a different explicit resource policy;
it is not a new neural architecture. Canonical runtime source `26516745` was
imported through the exact external native interpreter, with the actual settings
builder selecting `embedding.threads=1`. The product default remains automatic
allocation. Native dependencies remain the same 57-package inventory and the
model environment remains the same 25-package inventory. The pinned 22-layer
checkpoint, random head initialization, float16 SDPA, model-worker thread
settings, TF32 state and full framing are unchanged.

The input stores, requests and expected frames are copied from the earlier
frozen resident attempt and revalidated on the current source. All stores
contain 10,000 accepted synthetic facts. Complete owned retained excerpts come
from the actual selected facts. There is no truncation, source omission, new
padding or score-dependent skipping. The clock includes real offline BGE,
scoped retrieval, context/source assembly, authenticated RPC and queueing,
tokenization, device transfer, full forward computation, synchronized finite
CPU readback, fresh authorization and revision/source checks, serialization
and client decoding. Existing availability and stale-input failures fail the
whole attempt. The six workloads cover at most eight sources per fact; they
do not prove performance for every source inventory the serializer accepts.

All earlier failures remain unchanged: the 200 ms component allocation, the
earlier complete-path attempt, and the ordinary-recall non-regression result.
This is candidate-only admission. Historical timings are not fresh paired
controls. No causal claim that the thread policy caused an improvement or
explains the earlier failure is established. Jev and all hosted inference are
absent. No capture or normal-hook clock was tested.

Forty framing/failure/revalidation checks and 25 policy/default checks passed
before launch, including a private settings-contract check. The bounded review
found an environment-binding gap; before inference the harness was changed to
enforce the exact native interpreter, dependency root, prior native inventory
and current source. Wrong-interpreter rejection and correct-interpreter
acceptance were checked. The original GPU inventory checks were preserved.

The offline audit reconstructed all raw samples, frame and output bindings,
per-block call/forward counts, percentiles and stable source snapshots. Bounded
independent review also verified all 336 sample/frame/output records and all
six p95 calculations. The reviewer rehashed core inputs and outer controls;
the complete 4,490-file checks, inventories and source heads rely on saved
terminal receipts. The Windows-owned job finished in 186.54 seconds, exit zero,
with zero owned survivors. Both workers stopped and both runtime descriptors
were removed. All frozen bindings and inventories matched after execution.

Runner source: `7dbca15fbdfc0a17e55bea0f8781dcf6a887808a`.
Runtime source: `2651674553dd609ca726c9964b17310616ac14a8`.
Freeze SHA-256:
`851b302b91d6ec90556a02a4cfab539d4e4089580b77ad752810511fa178ac2e`.

This establishes computational feasibility only for this explicit configuration
and six workloads. The head is untrained; its outputs are not support
probabilities or semantic decisions. No quality, training readiness, fresh
acceptance, native host, pilot, publication or default-policy change follows.
The permitted next step is a separate reviewed grouped-data and learning plan.
The old 127-case corpus and its three connected training groups remain known
diagnostics, not sufficient new data or fresh acceptance.

Private evidence: `.venv/cooperative-support-capacity-v1`. Archive:
`.venv/cooperative-support-capacity-v1-private.zip`, 277 members,
3,172,311 bytes; all 275 included file hashes verified on readback. SHA-256:
`6aa5e2e6222aacbc16c8796a3f929eb88d3fb6b8cbca3288b2c7b582ddb1a3b5`.
Large model/dependency bytes and frozen seed databases retain external hash
bindings and remain local. No live memory, normal configuration or pilot
installation changed.
