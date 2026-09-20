# Direct reconsolidation returns committed metadata

After the captured-duplicate merge repair, direct MutableRAG.review still
returned its pre-commit proposal. SQLite had committed the correct target row,
including absorbed_captured_ids, combined source_event_ids and observation_count,
but the caller received stale values. The service job path already reloaded the
stored row, so its passing lifecycle did not cover this direct API mismatch.

A new regression reproduced the mismatch by comparing the complete returned
record against the stored target after a captured duplicate was absorbed.
The repair reads the target back after commit and returns that authoritative
record. A missing committed row raises an error instead of returning a proposal
as though it were still stored.

Validation: 20 focused reconsolidation, captured-merge and collision tests pass.
The regression passes again after adding the captured source event to its
fixture. Black, git diff --check and strict Python 3.10 mypy for the changed
module pass. Full-suite and rebuilt-package checks for this small follow-up
remain open. The preceding 26c9ddb package checkpoint remains historical evidence
for that exact version; it does not validate this new return-value change.

No production prompt, live host, live database or publication changed.
