# One resident complete-path computation test

Status: [the sole amended attempt failed](../validation/2026-11-11/resident-support-path-v1.md).
All 300 measured recalls completed, but the two full-token p95 values were
672.67/700.18 ms and the byte-bound p95 was 549.00 ms. This integrated
configuration is closed. The prospective amendment below is preserved.

The [component screen](../validation/2026-11-11/bidirectional-support-capacity-v1.md)
failed its frozen 200 ms allocation. That result remains failed and its files
remain unchanged. The 200 ms limit was a planning heuristic reserving an
unmeasured 300 ms, not a necessary consequence of the original complete 500 ms
requirement. Treating it as a veto on the complete path was a planning error.
This amendment explicitly permits revisiting the same configuration once at
the original complete boundary. It does not turn the component failure into a
pass, reopen other closed configurations, or authorize training.

The [current diagnostic](../validation/2026-11-11/current-recall-profile-v1.md)
contains one 99.8622 ms installed recall. It establishes neither a percentile
nor an improvement over the old run. Never add its duration to a component
percentile and call the sum a complete-path result.

Use installed runtime f1314f6d, the existing disposable 10,000-accepted-fact
snapshot, real offline BGE embeddings and exact scoped SQLite retrieval.
Preserve the original recall clock: immediately before awaiting client.retrieve
through receipt and decoding of its complete response, after resident setup.
Measure cold setup separately. Hook/process launch and the independent 100 ms
capture gate remain separate obligations.

Inside that clock include retrieval, context construction, all selected owned
retained source excerpts, cohort framing/tokenization, resident queue and local
RPC, device transfer, full encoder forward, synchronized finite CPU readback,
fresh authorization and fact/evidence/source revalidation, and HTTP response
serialization/decoding. Sources must come from the actual retrieved facts in
the disposable store. Random head outputs cannot omit a fact, source,
revalidation or response work, and cannot become support decisions.

Keep exactly the prior pinned checkpoint/revision, 22-layer full encoder,
float16, SDPA, reference_compile=false, four/one threads, seed 2026111101,
TF32 off, trusted target markers and native boundary tokens. Keep the 8,192
total-token and 64,000-payload-byte limits without truncation. Run six fixed
workloads: empty, one short, five short, full-token five with one source each,
full-token five with eight sources each, and exact-byte-bound five. Any source
padding must exist as owned retained evidence in a prepared synthetic store
before freezing; it cannot be supplied after retrieval as an unrelated cohort.

Keep accepted native dependencies and the minimal model environment unchanged.
Use a resident authenticated loopback worker to separate their incompatible
tokenizer dependencies. Freeze code, inputs, model hashes, tokenizer framing,
both environment inventories and expected selected IDs before execution. No
normal host credentials, Jev key, live memory or live configuration is needed.

Snapshot visible fact content/version/authority, all owned evidence and source
event state, then revalidate after inference against a fresh registry and a
consistent SQLite read snapshot. Revocation, mutation, malformed/oversize
input, missing source, unavailable worker, nonfinite output or binding failure
fails the whole attempt. Do not silently fall back to Jev or skip the work.

Use two fresh resident service/model blocks, three warmups per workload per
block, then 25 round-robin measured calls per workload per block: 50 measured
calls per workload, 300 total. Preserve the original interpolated p95 method
and require every workload p95 <=500 ms. Record all calls and denominators;
any operational failure aborts the sole attempt with unattempted calls still
reported. No timing retries, shorter inputs, altered backend or budget
reallocation. Bound all owned processes and cleanup to 600 seconds and retain
terminal process receipts. Test framing, failure handling and revalidation
without model execution before freezing this one attempt.

A complete timing pass permits a separate grouped-data and learning-plan
review only. It provides no semantic quality, training authorization, fresh
acceptance, native host, capture, daily-use pilot or release pass. A failure
closes this integrated configuration. Local operation remains required; Jev
remains an explicit optional backend with no automatic fallback.
