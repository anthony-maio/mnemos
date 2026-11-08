# Binary token-support development result

The fixed 0.9 source-witness rule failed development quality gates. This candidate remains experimental. No production default or original pilot gate changed.

| Measure | Result | Required |
| --- | --- | --- |
| Selected-fact precision | 11/13, 84.62% | 95% |
| Critical-fact retention | 11/12, 91.67% | 95% |
| Answerable recall | 11/12, 91.67% | 90% |
| Correct abstention | 11/12, 91.67% | 90% |
| Operational failures | 0/24 questions | Report all |
| Support-only p95 | 619.01 ms | Full warm retrieval must fit 500 ms |

The separate synthetic development cohort contained 24 questions and 52 fact/source witness calls. One unrelated warmup preceded the fixed lexical question order, with source order retained within each question. No question was repeated. The inference runner never parsed the separate grading labels. A later audit recomputed each decision from its saved raw response and checked every selected witness against the recorded outcomes.

The errors were concrete: accepting a fact whose source supported only part of its regional claim, treating a replacement fact as an answer to a question about the reason for replacement, and rejecting a supported fact because of an unsupported peer replacement claim. The raw responses and separate grading rationales are retained. These observations identify development failures, not evidence that a threshold adjustment would fix them.

Every raw prompt matched its frozen tokenized frame, and native token sequences matched the previous preflight before any completion. Reported inference prompt counts matched the expected counts. Settings were raw input, disabled thinking, temperature zero, one output token, seed 2026110801, context 8192, and top 20 token scores. Both offered labels were required by the strict decoder. The threshold remained 0.9 throughout.

This harness executes frozen witness requests and records whole-question support time; it does not run retrieval, extraction, the production composition, live revision races, or a native host workflow. Input hashes were checked around calls. A failed witness would invalidate the whole question, even if another witness passed. The 120-second question deadline is cooperative and does not establish a hard wall-clock bound. Minimum support time was 304.75 ms and maximum was 894.22 ms. These times cannot pass the complete retrieval gate or establish a direct speedup against different earlier cohorts.

The owned model was unloaded, both processes were verified absent, both ports were closed, and runtime and prepared-input bindings were unchanged. The helper exited successfully after cleanup. The controlled server termination returned exit 1, with taskkill exit 0; this is recorded separately from inference success.

Private raw evidence is in `.venv/token-support-execution-v1`, including the pre-inference freeze, exact prompts and tokens, responses, outcomes, grading, process ownership, and cleanup. The execution and grading helpers are archived alongside it. This is authored development data, not independent acceptance evidence. Preserve this closed run without threshold tuning or reruns; a revised mechanism needs a prospective development plan and separate evaluation data.
