# Durability and history comparison v3: terminal record

All 92 planned calls completed in the frozen case/arm order. Runtime outcomes:
{"production": {"runtime_valid": 45, "error": 1}, "durability_v2": {"runtime_valid": 46}}. Every raw response and failure is retained. Exact prompts
and schemas were checked: the added durability clause is the only prompt
variation between arms. Both use identical frozen provider settings.

The 26 durability cases and 20 historical/preference cases retain their
original rubrics and pre-execution ambiguity reviews. The internal candidate
arm label durability_v2 denotes the exact v3 clause bound in inputs.json. These are development cases; one stochastic repetition
cannot establish held-out quality, reliability or an adoption decision. The
known native-event failure is intentionally reused as one development case.

Semantic review remains separate and must preserve stated ambiguities. No
facts were committed and no production prompt or configuration changed. This
proposal-only experiment is not an all-module ablation or native-host recovery.
