# Retrieval during an embedding outage: development evidence

Two regression cases failed against a real Ollama embedding client pointing
at a bound, non-listening loopback endpoint. Baseline and biomimetic retrieval
raised a transient error before SQLite could search an existing matching fact.
Four host regression cases also failed before Claude and Hermes propagated
degraded status. The original failing JUnit records are preserved here.

The repaired source passes 85 focused retrieval cases and 104 host cases.
These are overlapping suites, not a combined unique-case total. The tests
cover scope before candidate limits, memory versus captured-evidence views,
empty results, context budgets, corrections and forgetting during a failed
embedding wait, error taxonomy, no diagnostic persistence into facts, and
healthy retrieval after recovery. Host checks include nonempty and empty
degraded results. Production and test changes are copied for review.

Fallback uses SQLite full-text ranking and the existing scope/view filter.
It returns a request-local degraded status, labels context as lexical-only,
and leaves semantic and total scores unmeasured. It does not call affective
classification or graph traversal. Access metadata can still change; fact
content and versions cannot change merely from recall. Configuration and
database failures propagate. A backend without lexical search retains the
original provider error. No live model, host configuration, or memory data
was changed by these focused checks.

This is development evidence. Package verification and a native host recall
experiment are separate gates. Lexical matching can miss paraphrases and
waits for configured provider timeouts/retries; no outage latency guarantee
or extraction success is established. The frozen real-model matrix uses its
original checkout and is not modified by this repair.

The evidence index binds the raw JUnit records, patch, test copies, and helper.
This README is outside that index.
