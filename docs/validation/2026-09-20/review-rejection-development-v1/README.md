# Reviewed-forget rejection and outbox repair

The verified 5155 package had a client defect missed by its earlier tests:
an invalid-selection rejection could replay valid queued captures before
returning. The old test's queued record omitted required capture fields, so
replay failed and left the row pending. That did not prove replay was skipped.

The new regression uses a valid queued capture through the authenticated
client and Hermes dispatch. Both initially failed because rejection was
followed by a capture dispatch. The client now returns before draining the
outbox for every recognized related-forget rejection. Storage and queued
records stay unchanged; separately requested replay still commits the capture.
The positive replay check uses the committed receipt's scoped event ID.

Four additional HTTP regressions reproduced generic service failures for
duplicate fact IDs, duplicate source IDs, incomplete previews, and shared-source
dependency conflicts. These caller-correctable paths now use a bounded typed
rejection family. The client and service share one map of safe codes and
messages. Unexpected internal errors remain opaque HTTP 500; scope, expected
versions, digest checks, explicit selections, and deletion behavior are unchanged.

The [native v12 cohort](../hermes-native-cohort-v12/README.md) separately
observed the incomplete-preview HTTP 500 on the preceding installed package.
It remains failed and does not validate this repair.

## Evidence

The final focused transport run passes 32 cases, including malformed rejection
codes, unknown/wrong-operation responses, storage preservation, a valid queued
capture, explicit replay, valid deletion, and lost acknowledgements. Earlier
red reports are retained. Two intermediate positive-replay assertions used
an unscoped ID and then a nonexistent receipt field; both were corrected to
the existing capture contract before the final run. They were test defects,
not additional production changes.

The evidence index binds the raw JUnit reports. Full source/package and
isolated installed validation of this successor remain pending. Native-host
and full-module quality gates remain open.
