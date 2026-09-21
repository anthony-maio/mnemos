# Reconciliation decoding-schema diagnostic v6

The generic ReconciliationResponse decoding schema replaces v5's bounded branches. Prompt, model, cases, expected decisions and runtime guards are unchanged. Four provider cases pass runtime validation; five fail it. The tenth rejects foreign scope before a call. Four of ten original expectations match, including deliberately unresolved outcomes and the known contract gap. No production writes or adoption.

The unrelated new-fact omission persists. Support, correction and targeted conflict omit required observed versions and are rejected. The duplicate-target case attempts a correction against an arbitrary identity while its own reason says it should be unresolved; its missing version is rejected first. The original runtime ambiguity guard is unchanged. The mixed case fails because its support lacks a version and still labels the independent fact unresolved.

This result does not support removing the bounded decoder schema. It also shows that a plausible explanatory reason is not a reliable account of the selected operation. The generic schema is a diagnostic runner change only, not a product change.
