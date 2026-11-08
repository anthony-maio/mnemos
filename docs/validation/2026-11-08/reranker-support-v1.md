# Trained reranker support screen failed

The frozen Qwen3-Reranker-0.6B configuration retained all 40 expected facts but selected 73 unsupported facts. Precision was 35.40%, and correct absent-answer abstention was 5/28, or 17.86%. It fails the original quality gates and is closed without promotion or threshold tuning.

| Measurement | Result |
| --- | ---: |
| Expected retention | 40/40 |
| Answerable recall | 40/40 |
| Selected-fact precision | 40/113, 35.40% |
| Correct abstention | 5/28, 17.86% |
| Operational failures | 0/68 |
| Support-only median | 174.23 ms |
| Support-only nearest-rank p95 | 375.64 ms |

Each target witness received the full candidate cohort, including competing values. The model nevertheless selected facts about unrequested attributes, both sides of current-duration conflicts, and historical/current values together. It also selected facts for missing approver, cause and region questions. These observations show that the trained ranking model and this target-witness framing did not provide dependable answer support. They do not reveal the model's internal reasoning or establish that all rerankers fail.

Execution used the pinned revision, fixed 0.9 two-label score threshold, one unrelated warmup and one lexical-order pass over 68 known questions. All 123 witness token frames matched preflight exactly, without truncation; lengths were 402-702 tokens. PyTorch 2.10.0+cu128 ran CUDA float16 on an RTX 4080, using SDPA, no generated tokens and no CPU fallback. Complete per-question support timing includes tokenization, transfer, forward pass, decoding and selection with CUDA synchronization. It excludes retrieval and therefore does not establish the full 500 ms target. Earlier CPU or different-model timings are not controlled speed comparisons.

All 21 frozen file bindings remained unchanged. The audit reconstructed every decision from saved no/yes logits, checked token counts and question order, and graded against the separate labels. A second calculation independently reconstructed witness selections using the fixed softmax formula. The runner exited zero and its recorded process was absent. No server or persistent inference worker was started.

The first launcher used Python isolated mode, which hid the sibling decoder import. It failed before model loading or the execution-start marker. That failure is preserved; direct script invocation repaired only the launch mechanism. No semantic request was retried. The earlier CPU QA environment and live pilot were not modified.

The 23-member evidence archive retains requests, separate labels, raw logits, token frames, pinned metadata, execution bindings, warmup, launch failure, helpers and audits. SHA256: a665e250de547d4fff6cba5ee18bd7211e232a95af659aa8c25564afdc89bac2. Large weights remain separately retained with their hashes in the frozen manifest. The adjacent JSON contains exact scores, per-question errors and receipt hashes.

This screen used development data, not fresh acceptance. Critical-fact retention on independent data, full retrieval quality/timing, native-host readiness and actual pilot days remain open. A passing support-only timing cannot offset failed precision and abstention. Do not integrate this configuration or rerun it as a threshold search.
