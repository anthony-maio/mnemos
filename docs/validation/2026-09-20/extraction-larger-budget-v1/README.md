# Larger reasoning budget diagnostic

All 40 calls completed on the previously executed 20 authored cases, with two
prompt arms and one repetition. Runtime outcomes: {"production": {"runtime_valid": 20}, "history_only": {"runtime_valid": 19, "error": 1}}.
Only the output budget changed, from 8,192 to 12,288 tokens. Thinking remained
enabled, temperature zero and timeout 180 seconds. Source 05c71d2, prompt arms,
model identity and cases are bound in preparation.json. No selective retries
or prompt adoption occurred. This is diagnostic reuse, not unseen holdout data.

The candidate again failed P03 with an empty final response. Its response
metadata records done_reason=length and eval_count=12288. The larger budget
did not repair that failure. Production's P05 still omits the user's explicitly
retained uncertainty; the candidate preserves it. These observations are not
an aggregate semantic accuracy score. Review of all outputs remains separate.

This is a prompt/extraction experiment with local Qwen3.5:9b, not a five-module
ablation or native-host acceptance result. No facts were committed. Raw requests,
outputs and failures are retained without changing the authored rubric or the
previously recorded P07/N10 scoring ambiguities. T18 remains open.
