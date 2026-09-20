# Deletion coverage receipt checkpoint

Frozen source `1a110583a18408cbd3de4a695a9962109c4d02e7` passes 1,439 source tests
with 4 skips, all six source/package checks, 608 installed
cases with 4 matching skips, and seven replay assertions. The isolated
installed run matches individual case outcomes and verifies module origins
and package bytes against the frozen source and wheel.

Wheel: 302,249 bytes; SHA-256
`4e3d16d00f7b9ab54fe1e5350f7cf0fbe6209dc881bfc6d7d9f974f1c3f697a5`.

## What changed and why

In the [six-case development diagnostic](../model-forget-development-v1/README.md),
GPT-OSS made a valid target-only deletion, retained all three seeded copies,
and claimed every related copy was gone. Storage honored the selected IDs;
the receipt did not summarize the records left in its reviewed preview.

Related-forget receipts now include `deletion_coverage`: target-only or
reviewed selection, retained exact and advisory fact counts, and retained
eligible source-event counts. These counts are computed inside the deletion
transaction from the validated preview. They exclude foreign-scope facts
and distinguish advisory similarity from an exact assertion match. Direct
forget reports target-only coverage without counts because no preview exists.
Both receipts set `semantic_erasure_proven` to false, even when counts are
zero. Guidance for Hermes and MCP tells the agent to report this coverage.

This does not select additional records, broaden deletion, claim exhaustive
semantic matching, or cover future captures. No native model has run the
new package. Whether clearer receipts improve model behavior remains untested;
the native v6 failures and direct diagnostic outcomes remain unchanged.

## Regression evidence and limits

Four new regressions exercise the authenticated service/client/Hermes path:
target-only deletion retains a known exact copy; reviewed deletion removes
the selected copy and source; direct deletion has no inventory counts; and
removing all preview records still does not prove semantic erasure. The
first three also preserve an unselected advisory record and foreign control.
The focused green run passes 80 cases.

The red-v1 and red-v2 logs contain test-authoring mistakes: a nonexistent
store method and advisory metadata excluded by the preview contract. Those
are not product defects. Red-v3 has four failures for the intended missing
coverage field, before the production change. All development logs remain.

The first full source run at `2c0f777` passed all 1,439 tests with four skips,
then stopped when Black tried to format six archived helper scripts. Those
files are immutable execution evidence. The successor checkpoint excludes
`docs/validation` from Black and mypy, preserving their exact archived bytes.
Application and benchmark source remain checked; test execution is unchanged.
The earlier stopped run is preserved under `prior-source-checks`.

These are fixture, service, and installed-package results. Automatic capture,
native related-copy selection and erasure, three-host continuity, real-model
quality acceptance, Desktop, and the pilot remain open. Live host settings
and live memory were unchanged. The evidence index binds 48 raw files;
this interpretive README is outside that index.
