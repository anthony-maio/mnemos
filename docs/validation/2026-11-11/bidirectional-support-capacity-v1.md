# Bidirectional encoder misses its planned component budget

The one frozen ModernBERT capacity attempt completed all 120 measurements and
106 root-model forwards. Both full-length workloads exceeded the prospective
200 ms p95 component allowance. This configuration is closed before training.
Do not shorten its bound, change depth/dtype/attention, raise its admission
threshold or rerun it to obtain a pass.

| Synthetic workload | Tokens | Samples | Median ms | p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Empty cohort, framing only | 43 | 20 | 0.42 | 0.60 |
| One short candidate | 107 | 20 | 26.17 | 31.73 |
| Five short candidates | 367 | 20 | 25.39 | 31.97 |
| Five candidates, one source each, token bound | 8,192 | 20 | 206.55 | 226.21 |
| Five candidates, eight sources each, token bound | 8,192 | 20 | 209.14 | 213.99 |
| Five candidates, exact 64,000-byte payload | 2,992 | 20 | 49.20 | 56.32 |

The 200 ms allowance reserved an unmeasured 300 ms for the rest of the original
500 ms complete-recall requirement. This result fails that prospective allocation;
it does not prove the original 500 ms complete boundary is impossible for every
bidirectional architecture. No complete recall, 10,000-fact pipeline, quality,
critical retention or fresh acceptance was measured. Twenty samples per workload
are a small computational screen, not a reliable tail-latency guarantee.

The [design](../../plans/2026-11-11-bidirectional-support-design.md) is a substantive
change from the closed causal-prefix/LoRA and zero-shot NLI experiments. It
uses the complete bidirectional `tasksource/ModernBERT-base-nli` backbone,
revision `de4ab7e77845098b7fab7f6ab9d370ddff27b19c`, with its original NLI head
discarded. All 22 backbone layers and 149,014,272 backbone parameters loaded;
there were no missing, mismatched or failed backbone keys. Only the documented
NLI classifier/prediction-head keys were unused. The new shared head was random
and untrained. Its finite outputs are computation witnesses, not support
probabilities, labels or semantic decisions.

Source `32f1db00476ab27a0261519a461661b7f327c498` ran CUDA 0 on the RTX 4080,
driver 610.60, float16, SDPA, reference_compile=false, four intra-op threads,
one inter-op thread, seed 2026111101 and TF32 disabled. Torch 2.10.0+cu128 and
Transformers 4.57.6 and the full runtime inventory were verified before loading.
Loading took 15,941.83 ms separately. Peak CUDA allocated memory was 786,842,112
bytes. The outer Windows-owned job ended after 39.07 seconds, with six owned
processes recorded and zero survivors after cleanup.

Every timed operation reconstructed and validated the exact complete cohort,
tokenized all text, transferred input, executed one forward for a nonempty
cohort and materialized shape-checked finite CPU outputs after synchronization.
No source was dropped. CLS/SEP and per-target markers retain their trusted
positions; token/byte overflow was rejected during preparation. Twenty rounds
used the same six-workload order, with six untimed warmup forwards. The p95 is
the 19th sorted duration of twenty, without trimming. Every expected sample
completed. The nonzero worker exit records a timing-gate failure, not an OOM,
timeout or operational missing sample. All file/runtime bindings stayed intact.

Independent review recomputed every sample count and percentile, reconstructed
all six frames with the pinned tokenizer, checked full source preservation and
target/marker binding, and verified finite output shapes and the 106-forward
count. It rehashed all twelve frozen files and four outer bindings and matched
the runtime inventory. No material scoring or integrity issue was found. The
review performed no model forward or training.

Twenty-six framing and failure-receipt checks passed before execution. Scoped
review resolved runtime revalidation, missing-file receipt handling and external
600-second containment before the first forward. Preparation previously exited
on an unrelated benchmark-package dependency and then absent pip; direct helper
loading and standard-library package inventory fixed those bootstrap assumptions
without adding dependencies. There was only one model-forward attempt.

The [machine-readable result](bidirectional-support-capacity-v1.json) preserves
the failed allocation and unmeasured requirements. Private evidence is in
`.venv/bidirectional-support-capacity-v1`. The verified archive
`.venv/bidirectional-support-capacity-v1-evidence.zip` has 28 members and is
712,144 bytes; 26 file hashes were read back. SHA-256:
`3c54ec6ea35040cd579e986f9b0a7629a7bc8932126c5e002165e0aaa2ee8a9d`.
The separately retained weight hash is
`86c32c52ce38b8f26e028ca959b06daee3a5f3f6947c63258bc8695dde88a465`.
Freeze SHA-256:
`604735a1e597cf63bc905731a499a10c4fc2490f6030effcaf30427da81cfa33`.

No training, corpus-label use, held-out consumption, Jev request, production
integration, live database/configuration change or pilot refresh occurred. The
accepted installed Codex candidate remains unchanged. Further semantic work
requires a concrete architecture and complete-budget decision, not another
model/template/backend sweep or an unsupported claim that a reserved component
allocation is the original end-to-end gate.
