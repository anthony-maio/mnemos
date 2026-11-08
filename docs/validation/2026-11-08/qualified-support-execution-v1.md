# Qualified support development comparison

The two-stage candidate failed development quality gates. Cold and warm selection agreed exactly: both recovered the baseline's one missed expected fact but admitted seven unsupported facts, compared with four in the baseline. Neither treatment is ready for promotion.

| Measure | Baseline | Qualified cold | Qualified warm |
| --- | --- | --- | --- |
| Expected facts retained | 11/12 | 12/12 | 12/12 |
| Selected-fact precision | 11/15, 73.33% | 12/19, 63.16% | 12/19, 63.16% |
| Correct absent-answer abstention | 10/12, 83.33% | 8/12, 66.67% | 8/12, 66.67% |
| Operational failures | 0/24 | 0/24 | 0/24 |
| Support-only p95 | 858.02 ms | 1508.05 ms | 489.78 ms |

All 12 expected facts were designated critical. Required precision and critical retention remain 95%; recall and abstention remain 90%. The warm support-only timing does not establish the complete 500 ms retrieval gate: it excludes retrieval and source qualification, uses 25 synthetic facts rather than 10,000, and follows the cold arm in a fixed order that can favor provider caching. There are 24 distinct questions, not 72 independent semantic cases.

The separate source stage made 50 assessments of 25 unique pairs. It falsely supported one unique pair twice: a limited negative detection report was treated as support for universal absence. There were no false source rejections or unresolved decisions. Accurate source qualification elsewhere did not prevent wrong requested-relation and time selections. The final candidate still answered missing contractor, cause, and approver questions, and selected November capacity facts for an October question. The saved evidence supports these observed failures; it does not expose the model's internal reasoning.

Before inference, all 633 frozen frames passed exact native tokenization with add_special=true and parse_special=true. Counts ranged from 207 to 759, allowing one output token within context 8192. The execution repeated tokenizer parity checks, required every runtime request to match a frozen frame, and checked reported prompt counts. The source policy, model blob, runtime identities, dependency inventory, schedule, seed, one-token settings, and helper hashes were frozen before completion calls.

Execution made one unrelated warmup and 200 semantic calls. The runner used the actual baseline and qualified selectors, kept labels outside inference, and retained all raw responses and observations. A separate grader matched observations to HTTP responses, recomputed token decisions, checked qualification annotations against the cold source assessments, reconstructed selected witnesses, and graded source and answer stages separately. All prepared-input and runtime bindings remained unchanged.

The owned model was unloaded; both owned processes were verified absent and both ports closed. Tokenizer and execution helpers finished successfully. The controlled server termination returned exit 1 with taskkill exit 0, recorded separately from inference status.

Private evidence is retained in `.venv/qualified-support-tokenizer-v1` and `.venv/qualified-support-execution-v1`, with execution, preparation, tokenization, and grading helpers archived alongside them. The cohort is authored development data, not independent acceptance. Preserve this closed run, its failed gates, and the unchanged 0.9 rules. Do not promote the cache or selector based on warm timing. Further support work needs a revised mechanism and separate evaluation data; the original capture, pilot, release, and fresh acceptance obligations remain open.
