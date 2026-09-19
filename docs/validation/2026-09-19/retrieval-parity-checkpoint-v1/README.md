# Retrieval backend parity checkpoint

Source `8fc79f01ab68d9c211f9be4818f9e566e271cefd` makes the in-memory store follow SQLite's existing exact-tie
ordering by ascending memory ID. Empty query vectors and nonpositive limits
return no candidates before evaluating predicates. The shared storage contract
now states these boundaries. SQLite behavior and similarity formulas are unchanged.

The initial regression fixture had teardown and incomplete scope-specification
errors; those were corrected before evaluating production failures. The valid
pre-fix run had four in-memory failures and eight SQLite passes. All 12 cases
pass after the fix, covering native SQLite, its Python fallback, memory storage,
scope filtering, reinsertion, empty queries, and nonpositive limits. The focused
storage/benchmark run passed 49 tests. Read-only review found no blockers.

The clean snapshot passes 1143 source tests with
4 skips, Black, strict mypy, JavaScript syntax, build, and Twine.
The new isolated installed wheel passes 440 contract cases with
2 matching skips and seven replay assertions. Exact per-case
outcomes, package origins, hashes, and commands are archived. This is package
verification, not native-host or model-quality acceptance.

The repeated offline control retains the same vectors, datasets, scope policy,
top-k, repetitions, and quality thresholds. All 46 baseline query/repetition
pairs now return identical IDs from memory and SQLite. Before this repair,
26 pairs differed on exact cosine ties. The valid new report records 24 runs
and 184 per-query outcomes. Engine quality still does not pass the replacement
gate in either repetition. Synthetic vectors and ingestion differences limit
interpretation; this fixes backend parity, not semantic quality.

The original legacy benchmark was also rerun with explicit simple embeddings
and a missing child configuration path. Both repetitions fail unchanged gates
(MRR lift ranges from -0.4 to 0.0). Exit code 2 and raw reports are preserved.
No cloud credentials or live configuration were used or changed. The active
local model matrix and queued comparisons remain on frozen `fc72cc2`.

The previous [fair control](../fair-comparison-diagnostic-v1/README.md), including
invalid attempts, remains intact. Build artifacts and disposable SQLite fixtures
remain at their bound external paths. Archived helper bytes are stored as text.
