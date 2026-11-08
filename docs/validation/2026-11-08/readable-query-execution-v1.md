# Readable query development result

The readable-label probe completed all 22 calls without retry. It remains a failed development candidate: eight expected facts were retained, but two unsupported selections reduced precision to 80%, and provider p95 was 2,113.34 ms before initial retrieval or source annotation. The original precision and complete-latency gates are not met.

## Results

The 18 semantic cases contain nine scenarios in forward/reversed order. They are authored development cases with supplied annotations, not extraction, independent acceptance or eighteen independent scenarios. Four forced-support controls are reported separately.

| Semantic observation | Result |
| --- | ---: |
| Complete, valid replies | 18/18 |
| Exact judgments and temporal intent | 13/18 |
| Correct final decisions | 16/18 |
| Correct temporal intent | 18/18 |
| Correct relation judgments | 34/36 |
| Correct condition judgments | 31/36 |
| Fully correct claim judgments | 31/36 |
| Expected selected facts retained | 8/8 |
| Correct selections / all selections | 8/10 |

Both orders of the unspecified-condition scenario incorrectly treated the maintenance-window prerequisite as satisfied and selected the conditional fact. Both excluded-condition cases reached the correct absent decision through the wrong intermediate judgments: they excluded the relation while treating the condition as applicable. One source-instruction case also reached the correct final answer with an incorrect condition judgment for the irrelevant claim.

All four support controls had exact judgments and correct final decisions. Across all eleven pairs, temporal intent was consistent in every pair; judgments by original fact were consistent in ten. The source-instruction pair differed in its condition judgment. Consistency does not imply correctness: the unresolved-condition error occurred in both orders.

Provider median was 1,849.67 ms and p95 2,113.34 ms; maximum was 2,435.12 ms. Runner processing median was 1,951.27 ms and p95 2,644.78 ms. Median prompt and generated counts were 770 and 72 tokens. These exclude initial retrieval, extraction and native-host dispatch. There was no semantic warmup. This is not a complete retrieval measurement or a controlled causal comparison with the earlier packed run, which used different cases and prompts.

## Execution and evidence

The wrapper verified the v3 prepared files, exact code bindings, retained model bytes and local runtime binaries, reconstructed the owner requests, and tokenized every full raw prompt before inference. All 22 fit context 8192 with the frozen 256-token output allowance. The model was local Qwen3:14b with temperature zero, seed 2026110808 and thinking disabled. Inputs and grading stayed separate. No source extraction or initial retrieval was performed.

The hash-chained journal retains each full request and provider reply. The independent audit checked original prepared requests, tokenization records, all judgments, final decisions and order pairs. Runtime cleanup verified model unload, absence of the owned parent/native processes, closed ports and unchanged prepared/runtime bindings. Execution session 78452 and audit session 48874 both finished with exit code zero.

The [receipt](readable-query-execution-v1.json) includes per-case results, exact hashes and timing. Evidence archive `.venv/readable-query-probe-execution-v1-evidence.zip` has SHA-256 `fe0ec51aa0a05a7d7523015ad42799f6c53ac846d5999725b575bcf0a920392d`. Wrapper SHA-256 is `aa8816dbc5a522c344c0d3b85c7514d157528389e3d7c3745c05a923318a3131`.

The run is closed. Preserve its original scores; do not tune or rerun this cohort to claim acceptance. The next design must address unjustified condition applicability and query-generation cost together. Source annotation quality remains separately unresolved. Nothing was promoted, published or installed into a live host, and no pilot days were earned.
