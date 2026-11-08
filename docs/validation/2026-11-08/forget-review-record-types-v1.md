# Record types in related-forget review

Delivery commit `a75cb609d0bcf8bd739d75a6b5c1f02f1181ee74` makes capture
provenance and assertion status visible in related-forget previews. It follows
the failed [Claude v7 attempt](native-claude-batch-v7-checkpoint.md), but does
not change that attempt's outcome or establish a performance benefit.

The source inspection corrected an initial interpretation of the audit:
sleep-disabled captures already use `captured_passthrough` and assertion status
`unknown`. Ordinary memory retrieval excludes them. They were not silently
promoted to accepted memories. The gap was presentation: forget previews showed
these records alongside extracted memories without exposing those distinctions.
Their `exact_assertion` match type describes text containment, not truth or
acceptance.

Targets and candidates now carry `record_kind` and normalized `assertion_status`.
Captured provenance determines `record_kind=captured_evidence`, even when other
metadata is inconsistent. Other records retain `record_kind=memory`, including
unknown records without captured provenance. Guidance explains that asserted is
a stored claim, not an independent truth judgment.

Both displayed classifications participate in the preview digest. Changing one
invalidates an earlier review even if content and version remain unchanged.
Raw metadata changes that normalize to the same displayed classifications need
not invalidate the digest. Candidate discovery and ordering, complete contents,
IDs, versions, dependencies, source selection, atomic deletion and suppression
are unchanged. Every selected record still requires inspection.

Validation: four new metadata-only concurrency regressions passed, covering
source/status changes for target and candidate. The broader related-forget,
similarity, transport, paging, batch-inspection and assistant-suppression checks
passed all 104 tests in 70.13 seconds. JUnit is retained privately at
`.venv/forget-record-types-broad-v1.xml`. Independent review found no blocker.
Ruff reports one pre-existing F841 unused local in `append_episodic_event`;
the same diagnostic was reproduced against the previous delivery HEAD.
The changed tests introduce no additional lint finding.

This is a source change saved to the delivery backup branch. It has not been
built into a new package, installed in the pilot, published or tested in another
native attempt. Do not rerun v7 with new labels and treat it as the same frozen
experiment. Before another host attempt, measure the existing review's content
and presentation overhead offline, preserving every record and source required
by the review contract.
