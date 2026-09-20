# Stale related-forget previews return an actionable rejection

Frozen source `f2f741443e60f838b392ae8ea1d1a0b4d8c6c1f2` passes 1,547 source tests
with 4 skips, all six source/package checks, 650 isolated
installed cases with 4 matching skips, and seven replay assertions.
The installed run verifies case outcomes, import origins, and package bytes
against the frozen source and wheel.

Wheel: 304,022 bytes; SHA-256 `8552345d2c17c4c037347961eb3043d13b20f790f54a9ccc86a096add6197b0f`.

## Observed defect and repair

The [native v9 cohort](../hermes-native-cohort-v9/README.md) failed all three
lifecycles on `e419c2a`. Attempt 3 requested a preview with limit 100, then
used its digest with limit 20 in every apply call. The first call copied all
ten candidate IDs and their required sources exactly. The limit is part of
the preview digest, so matching IDs did not make this a valid apply request.
The trace lacks a server traceback; the separate disposable regression
reproduces the old installed wheel's opaque HTTP 500 for this mismatch.

Storage still rejects a stale digest and rolls back. A typed ValueError
subclass now lets the service return noncommitted HTTP 409 with a fixed safe
code and a message to obtain a fresh preview and reuse its digest and limit.
The client accepts only the exact known response for `forget_related` and
returns it before replaying pending captures. Hermes receives the rejection.
No model arguments, selections, digest checks, or deletion rules are changed.

The changed-target-version path has the same rejection. Incomplete previews,
unrelated validation errors, unexpected ValueError failures, malformed 409
responses, and other operations retain their existing handling. Private
exception details and selected memory IDs are not included in the response.

## Evidence boundaries

The archived old-wheel test fails because Hermes receives degraded HTTP 500
instead of the expected rejection. Its import origin and all 62 installed
package files were checked against the original wheel. This reproduction was
run after implementation, against the unchanged old installation. An earlier
development failure was observed but its terminal output was not saved.

The repaired focused run passes 75 cases with two opt-in WSL integration
skips, including
the real authenticated service/client/Hermes path, unchanged database state,
pending-outbox preservation, valid apply, and unexpected-error behavior.
The full frozen and installed reports establish the package result above.
The [native argument diagnosis](development-checks/stale-preview-native-diagnosis.json)
preserves the original preview/apply comparison without editing or replaying
that native evidence.

No native model has run this new package. V9 remains zero of three; the
repair does not prove that the model will recover from the rejection or
complete deletion. Native lifecycle, three-host continuity, complete capture,
real-model quality, Desktop, and the operational pilot remain open. No live
host configuration, source, or memory was changed. The evidence index binds
the raw files; this README is outside that index.
