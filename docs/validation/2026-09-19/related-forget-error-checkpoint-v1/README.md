# Related-forget source-selection error checkpoint

Frozen source `e454a85f7b4f79ce2b483f738d68d88a15bd72d6` passes 1,427 source tests
with 4 skips, all six source/package checks, 604 installed
cases with 4 matching skips, and seven replay assertions. The isolated
installed run checks individual case outcomes, module origins, and package
bytes against the frozen source and wheel.

Wheel: 301,568 bytes; SHA-256
`94011300cda669ddcabefe9e9077af0f2aa01b6ed5f23bc13c8e6b7f2a5fbf35`.

## Observed defect and narrow repair

The [native v6 cohort](../hermes-native-cohort-v6/README.md) failed all three
attempts. In attempt 3, the model selected additional facts but omitted their
required source-event IDs. Storage correctly rejected the operation and
rolled back, but its ordinary ValueError became HTTP 500 at the service.

That specific domain rejection now has a typed ValueError subclass. The
authenticated service returns HTTP 409, status rejected, committed false,
and a fixed source-selection error code and message. The client accepts only
the strict known contract for forget_related and returns it without replaying
the outbox. Hermes receives the actionable rejection. The change neither
selects sources for the model nor retries the invalid request.

The original storage error text remains compatible. Unexpected ValueError
exceptions, malformed or unknown HTTP 409 bodies, and unrelated operations
keep their prior generic failure behavior. No internal exception text,
memory content, or selected IDs are added to the rejection response. Other
related-forget validation errors are outside this narrow classification fix.

Regression evidence covers the real service/client/Hermes path, transaction
rollback, explicit valid source selection, safe error handling, and outbox
preservation. The red-v1 development log contains one genuine missing-409
regression plus an unrelated test assumption error: the old success schema
does not include a committed boolean. The latter assertion was corrected to
use the existing status and forgotten fields. Do not count it as a product
defect. The green-v3 log records a collection error from a nonexistent test
path, not a test result; green-v4 uses the actual test paths. Frozen source
and installed reports establish the final results.

This is fixture, service, and installed-package evidence. No native model
has run this new package; the v6 failures remain unchanged. Selecting all
requested conversational copies, cross-host continuity, complete capture,
model-quality acceptance, Desktop, and the pilot remain open. Live host
configuration and live memory were unchanged.

The evidence index binds 37 raw files. This interpretive README is
outside the raw index.
