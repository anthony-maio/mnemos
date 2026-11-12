# Resident local recall computation: failed

The sole complete-path attempt failed the original 500 ms p95 requirement.
All 300 measured calls and 36 warmups completed across two fresh resident
starts, with 280 actual encoder forwards. There was no operational error,
training or semantic grading. Long-input tail latency failed the gate.

| Workload | Tokens | Measured calls | Median ms | Interpolated p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Empty scoped cohort | 47 | 50 | 24.67 | 40.57 |
| One short fact | 142 | 50 | 189.57 | 465.51 |
| Five short facts | 520 | 50 | 194.21 | 355.38 |
| Five facts, full token bound, one source each | 8,192 | 50 | 392.14 | 672.67 |
| Five facts, full token bound, eight sources each | 8,192 | 50 | 399.74 | 700.18 |
| Five facts, exact 64,000-byte payload | 3,112 | 50 | 235.24 | 549.00 |

The clock surrounds the installed client's complete retrieve call, after
resident setup. It includes real offline BGE embeddings, exact scoped SQLite
retrieval over 10,000 accepted synthetic facts, context assembly, all selected
owned retained excerpts, authenticated loopback worker RPC, full framing and
tokenization, full encoder computation, synchronized finite CPU readback,
fresh authorization and fact/evidence/source revalidation, then complete HTTP
response serialization and client decoding. These are measured combined
durations, not sums of component percentiles. Cold setup is recorded separately.
This is not the hook/process-launch or 100 ms capture result.

Source: `9d11490434177c795ed59698da45af2797aa79c8`. The exact installed
runtime remains `f1314f6d`; neither accepted native dependencies nor its
99 installed files changed. The resident model uses the same pinned
ModernBERT checkpoint, full 22-layer float16 SDPA geometry, seed and limits
as the prior component screen. Its random shared head measures computation
only. Every fact and source still returns regardless of its random score.
Local operation remains required; no Jev request, key loading, cloud fallback,
live memory access, pilot refresh or normal host configuration change occurred.

The amendment explicitly corrected the earlier arbitrary 200 ms component
allocation. That earlier result remains failed. This combined result also
fails and closes this integrated configuration: no timing retry, shorter
input, threshold rescue, altered backend or automatic training follows.
No precision, retention, abstention, fresh acceptance or daily-use claim is made.
Even a timing pass would only have permitted a separate learning-plan review.

Forty model-free checks passed before execution. A separate audit script
reconstructed every raw sample, workload count, percentile, target/frame
binding, finite output, per-block forward count and stable source snapshot.
All 155 frozen bindings matched afterward, including 17 embedding assets;
native/model inventories remained 57/25 packages. The owned Windows job
finished in 186.98 seconds, exited 1 for the failed gate and recorded zero
owned survivors. Both services removed their descriptors and both workers
terminated. The wrapper's failed status reflects that gate failure.

Review limitation: the bounded adversarial review was incomplete and did not
certify the final revision, cleanup fixes or wrapper. It found no verified
reason to stop the six-workload attempt. It flagged the prototype's eight
owned-source limit versus the serializer's 64-source limit. These measurements
cover the six specified workloads only; they do not establish unchanged
admission or performance for every input the serializer accepts. Production
input coverage and all original semantic denominators remain obligations.

Preparation failures are retained: the first preparation lacked a required
synthetic project identity; the second verified all retrieval cohorts but
stopped on the historical preregistration document's post-run status change.
Both stopped before support inference and created no attempt marker or freeze.
The existing second preparation was then sealed after checking the original
preregistration bytes in the preserved evidence archive. No measured attempt
was repeated and no historical failed evidence was rewritten.

Private evidence is under `.venv/resident-support-path-prepared-v1`: `freeze.json`,
`cases.json`, `frames.json`, the six prepared stores, both block directories,
`samples.jsonl`, `result.json`, `execution-control.json`, `job/process.json`
and `audit.json`. The wrapper and audit scripts are retained beside that folder.
The first preparation remains at `.venv/resident-support-path-v1`.
The portable public receipt records hashes and summary results.

Private archive: `.venv/resident-support-path-v1-evidence.zip`, 192 members,
614,690,324 bytes, SHA-256
`b7247a60f8a89ec1f47b40c20637726f9d31493f5049536d5fb46147f343312e`.
All 153 included frozen files were rehashed after extraction in memory; the
two excluded weight files retain their exact hash bindings. The archive
preserves preregistration bytes from the executed revision, before adding
these post-run status updates. Raw databases and private evidence stay local.
