# Source inventory tokenizer preflight

All 127 prepared source prompts passed the retained Qwen3:14b tokenizer check. Rendered prompts used 1,388-1,469 tokens; every request fits the 8,192-token context with 2,048 output tokens reserved. No semantic completion was requested. Query prompts depend on generated inventories and have not been tokenized.

The isolated wrapper verified the model, executable, native runtime and prepared input bindings. It loaded the model with an empty prompt, recorded exact rendered prompts and token IDs in the durable journal, unloaded the model and stopped its owned process tree. Parent and native processes were absent, both ports were closed, and runtime/input bindings were unchanged at completion. Wrapper session 52499 ended with exit code zero. The stopped server returned code one after the explicit process-tree stop; that is not a semantic evaluation failure.

The independent audit verified all 127 journal records, their hash chain, source-key coverage, rendered prompt hashes, token counts and context bounds. Machine-readable counts and evidence hashes are in `relation-inventory-tokenizer-v1.json`. Raw evidence remains in `.venv/relation-inventory-tokenizer-v1`; earlier preparations and failed screens remain unchanged.

This proves source prompt fit for this policy only. It does not establish annotation accuracy, query prompt fit, latency or full execution readiness. The pre-execution grading review identified an unresolved distinction between present-tense assertions and timeless validity. Resolve that temporal contract and freeze source grading before semantic execution; preserve this preparation if a new policy version is needed. See `docs/plans/2026-11-08-relation-inventory-grading.md`.
