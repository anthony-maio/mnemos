# Missing-target prompt diagnostic

All 40 calls completed on the same 20 previously executed authored cases,
with two prompt arms and one repetition. Runtime outcomes: {"production": {"runtime_valid": 20}, "history_only": {"runtime_valid": 17, "error": 3}}.
Only the candidate prompt clause changed. Source 05c71d2, model identity,
think=true, 8,192-token budget, temperature zero and 180-second timeout remain
bound in preparation.json. No selective retries or production adoption occurred.

The candidate repairs P03's PDF correction output but introduces empty-final
length failures on P02, P09 and P10, each at 8,192 generated tokens. It also
returns no fact for P05, dropping the expressly requested uncertainty that the
original candidate preserved. The change is rejected for adoption. Successful
JSON parsing is not semantic acceptance. All proposals and original rubric
ambiguities remain available for review; no aggregate accuracy is claimed.

This is diagnostic reuse, not unseen held-out data, a five-module ablation,
or native-host proof. The original production and candidate comparisons are
preserved. No facts were committed and T18 remains open.
