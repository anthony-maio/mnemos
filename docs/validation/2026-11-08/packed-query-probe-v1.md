# Packed query probe v1: failed semantic screen

The closed 16-call query-only probe completed without transport or protocol failures. It failed semantic quality. None of the 14 semantic cases had the exact expected temporal header and complete claim judgments. Seven of 28 claim judgments were correct, and six of 14 temporal headers were correct. Four of 14 final decisions matched, but those matches all occurred despite incorrect judgments elsewhere in the same reply.

The six answerable semantic cases selected no facts: 0/6 retention on this small supplied-annotation development set. No fact was selected in any case, so selection precision is undefined. Both conflict cases missed the expected conflict; one incorrectly returned absent. Only one of three expected-absence cases returned absent, and its underlying judgments were wrong. The three expected-unavailable cases returned unavailable, also with incorrect judgments. These are development counts, not estimates of fresh acceptance performance.

The two separately reported controls supplied unsupported or unresolved whole-fact annotations. Both reached the expected final decision, but neither had an exact correct reply. They do not contribute to semantic accuracy. This demonstrates why matching the final decision alone is insufficient evidence of correct query interpretation.

All replies used the required syntax and covered both positions. Reversing candidate order did not rescue field association or conflict handling. The source-instruction case returned the code string requested inside the evidence and marked both claims as matching, producing conflict rather than the expected owner selection. That observation is consistent with instruction interference, but one case does not establish its cause or general attack resistance.

## Execution and timing

The run used the frozen 16-case schedule, authored source annotations, local Qwen3:14b Q4_K_M, context 8192, output limit 128, temperature 0, seed 2026110807 and raw ChatML with thinking disabled. There was no semantic warmup, retry, repaired score or unattempted suffix. The tokenizer checked all exact raw prompts before inference. Model identity, runtime binaries and prepared inputs were verified; labels stayed outside provider input.

| Measurement | Result |
| --- | --- |
| Actual input tokens | 556-597; median 561.5 |
| Actual generated tokens | 5-14; median 14 |
| Provider wall time, all 16 | median 536.49 ms; nearest-rank p95 954.73 ms |
| First provider call | 954.73 ms; no generation warmup |
| Provider wall time, following 15 | median 534.26 ms; nearest-rank p95 584.36 ms |
| Generation alone, all 16 | median 310.44 ms |
| Runner processing, all 16 | median 643.55 ms; nearest-rank p95 1302.03 ms |

Runner processing includes response journaling, decoding and source/code checks after the call. It excludes request journaling, initial retrieval, source annotation, startup and cleanup. These measurements cannot satisfy the full warm 10k retrieval gate. The smaller representation reduced generated token counts relative to the earlier verbose format, but these are different inputs and policies, not a controlled end-to-end speed comparison. No supported fact was delivered, so shorter output produced no retained-answer benefit in this probe.

Cleanup passed: the model was unloaded, the owned parent and native processes were absent, both ports were closed, and runtime/prepared bindings remained unchanged. The owned server exit code of 1 followed the explicit verified process-tree stop. No live configuration, database or pilot setting changed.

## Evidence and next action

The complete request records, original wire replies, tokenizer records, labels, wrapper and audit are retained in `.venv/packed-query-probe-execution-v1-evidence.zip`.

- Archive SHA-256: `e35c68ee75d3c90cfa2f1a2def00f5c187947a8364ca70c68f695344fe837248`
- Event journal SHA-256: `7bace07fb5c68f0629b775ecb48599c74c020c7b647efbbb086da9a12f6b6571`
- Runtime freeze SHA-256: `db32aebc7210f99d914245e50f7321f4d5ba8742950207b05dfdde2290e3258e`
- Control SHA-256: `7bb8f34d019aed3d8c88cb595db7bac250937286aa59575ca50db5e2c4ea7466`
- Execution wrapper SHA-256: `0c694f72dc0f028b02116bd6c7a2ff90a916a1749bcb84064a64d2f36438910c`

Keep this candidate closed. Do not tune or rerun these cases as acceptance evidence. Packed output is not ready for integration. A different query design would need new evidence for field association, temporal intent, conditions and source-instruction handling. Source extraction defects remain independently unresolved; supplied annotations did not hide those obligations. The original quality, complete latency, fresh acceptance, actual two-week pilot and delivery requirements remain open.
