# Joint support capacity screen failed

The frozen float16/SDPA joint-backbone configuration fails the necessary latency screen at its declared 4,096-token bound. The two upper-bound shapes have p95 times of 876.17 ms and 875.35 ms before retrieval, exceeding the original complete-path 500 ms target. This closes the configuration before training. Short requests do not rescue the failed bound, and the workload will not be shortened to obtain a passing clock.

The run used the retained Qwen3-Reranker-0.6B checkpoint as a bare Qwen3Model with a random shared linear head on an RTX 4080, torch 2.10.0+cu128, transformers 4.57.6 and driver 610.60. The model loaded without missing, unexpected or mismatched keys. No corpus labels, semantic grading, optimizer, gradients or adapters were used. This is computational evidence, not a new answer-quality result.

| Shape | Tokens | Payload bytes | Measured samples | Median ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero candidates | 39 | 115 | 50 | 0.39 | 0.86 |
| One short candidate | 98 | 297 | 50 | 48.91 | 56.97 |
| Five short candidates | 334 | 1,029 | 50 | 47.22 | 57.85 |
| Five candidates, one source each | 4,096 | 8,551 | 50 | 870.08 | 876.17 |
| Five candidates, eight sources each | 4,096 | 9,671 | 50 | 870.80 | 875.35 |
| Five candidates, byte bound | 828 | 64,000 | 50 | 79.51 | 110.16 |

Timing runs from serialization/tokenization through CPU-materialized finite logits, including transfer and backbone/head computation, with CUDA synchronization. It excludes retrieval, transport, host launch and initial loading. Loading took 2.30 seconds separately. P95 is the nearest-rank 48th of 50 measured samples. No sum of these percentiles with historical retrieval timing is claimed.

The run completed 30 warmups and 300 measured requests in the frozen interleaved schedule, with exactly 275 backbone forwards across both phases. Empty cohorts invoked no backbone. All output shape/binding checks passed. Peak allocated CUDA memory was 3,911,905,280 bytes and peak reserved memory was 4,034,920,448 bytes. The process exited normally with result status capacity_screen_failed; normal process exit is not a passing result. Twelve model bindings and all frozen runner/input/protocol bindings were unchanged. Raw sample rows, counts and all six reported p95 values were independently recomputed from the saved files.

Preparation history is preserved. The first fixture construction assumed a token increment per filler and failed before weights were loaded or an artifact directory was created. Exact token counting corrected the boundary without changing the declared limits. The first completed freeze, v1, was never executed. Review then found that failed samples needed explicit context and that a cleanup exception could retain a passing status. Both were fixed before loading weights, with two targeted tests passing. V2 requests and frame bindings are byte-identical to v1; only the runner binding changed. No timed run was retried.

The executed v2 freeze SHA256 is `1cc656ab8ce251cf5ebed974610080a0215cbece48d69438080446bcc07ba18b`. Result SHA256 is `8527eb4fa15a9d120b85fda3fe01abd31e904240a7793042a6de4d5483c6fb49`. Private evidence is in `.venv/joint-support-capacity-v2`. The 16-member `.venv/joint-support-capacity-evidence-v1.zip` retains both freezes, requests, frame bindings, raw samples, result, original and corrected runners, serializer, head, protocol and archive helper. Archive SHA256 is `c2a7df1b9c1c31dea7e2303f3433d2b5e383548744dacdf8125404fdf0e5824e`; every archived member was read back and matched against its recorded hash.

Next work requires a concrete design decision about the failed computational path. Do not train this frozen configuration, change its attention/dtype or token bound and rerun to seek a pass, or consume fresh acceptance cases. The reviewed corpus remains preparation evidence. Capture latency, semantic quality, original pilot days and affected-user delivery remain separate outstanding obligations.

Independent read-only result review recomputed all six summaries from the saved samples and confirmed the closed-configuration conclusion. It distinguished rejection of this backbone/runtime configuration from a claim that every jointly trained classifier must fail. The run does not identify the causal performance bottleneck.
