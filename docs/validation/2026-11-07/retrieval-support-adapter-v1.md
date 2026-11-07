# Retrieval and support adapter boundary checks

The development adapter calls `LocalMemoryService.retrieve` before applying
the existing QA/contrast intersection. It accepts no supplied candidate IDs or
gold labels. The request fixes baseline retrieval, accepted-memory view,
project/global scope, top-k five and the conservative 2,048-byte context budget.
Both selectors receive the actual returned cohort. Selected facts are packed
whole, then their complete cohort is checked again against current storage.

Twenty-three new checks cover actual candidate selection, foreign-project
exclusion, correction and forgetting during either selector, evidence removal,
degraded retrieval, provider failure, a change between retrieval and support,
a change during final packing, empty-store abstention and budget enforcement.
The first 20 checks passed; three further boundary cases were added before the
focused run. All 71 checks pass across the adapter, intersection, selectors and
existing support comparison. The adapter passes strict typing and both changed
Python files pass formatting. No quality model or native host was run for
these deterministic checks.

Unavailable results contain no selected context and cannot receive successful
abstention credit. Stored conflicts retain the intersection's existing policy.
The final result records retrieval, support and complete elapsed time separately.
These checks establish the integration boundaries, not semantic quality or
performance. Product code and the frozen installed wheel are unchanged.

The [bounded development protocol](../../plans/2026-11-07-retrieval-support-development-v1.md)
fixes the existing 68 questions, two paired repetitions and one permitted
integration repair/rerun. Its private runner, paired snapshots, model bindings
and complete execution manifest still need preparation before inference.
No measured run has started. Separate critical-fact annotations are absent
from these known fixtures; that acceptance requirement remains unmeasured.
