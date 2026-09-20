# Durability extraction comparison: terminal record

All 52 planned calls completed in the frozen case/arm order. Runtime outcomes:
{"production": {"runtime_valid": 26}, "durability": {"runtime_valid": 26}}. Every raw response and failure is retained. Exact prompts
and schemas were checked: the added durability clause is the only prompt
variation between arms. Both use identical frozen provider settings.

The authored draft, pre-execution rubric corrections, 26 cases and 13 subject
pairs remain unchanged. These are development cases; one stochastic repetition
cannot establish held-out quality, reliability or an adoption decision. The
known native-event failure is intentionally reused as one development case.

Semantic review remains separate and must preserve stated ambiguities. No
facts were committed and no production prompt or configuration changed. This
proposal-only experiment is not an all-module ablation or native-host recovery.
