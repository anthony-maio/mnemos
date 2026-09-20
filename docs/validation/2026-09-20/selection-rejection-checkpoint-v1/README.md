# Out-of-preview forget selections return a noncommitting rejection

Frozen source `5155ebddd8e8cf42df1c67f11a1578f5edfb43ec` passes 1,638 source tests
with 4 skips, all six source/package checks, 655 isolated
installed cases with 4 matching skips, and seven replay assertions.
Installed validation checks case outcomes, import origins, and package bytes
against the frozen source and wheel.

Wheel: 304,380 bytes; SHA-256 `3828edfeab80f11a94a3b5519793af0b8b228b4bb5bef9ced34608186d6468e7`.

## Observed failure and repair

The [native Hermes v11 cohort](../hermes-native-cohort-v11/README.md) failed
all three lifecycles on the preceding `697b672` package. Attempt 2 selected
all ten previewed fact IDs but supplied two source IDs absent from the
preview. It reused the correct digest and limit. Both identical apply calls
returned HTTP 500 with no committed receipt. Read-only terminal inspection
found the target retained, no suppressions, and unchanged control state.

Unknown selected facts and sources now raise a typed ValueError subclass.
The service returns a fixed, noncommitting HTTP 409 rejection with code
`related_forget_selection_invalid`. The client recognizes only the exact
known response for related forget and returns before replaying pending
captures. Hermes receives the same rejection. The original scope, digest,
version, dependency, and explicit-source requirements remain unchanged.
No arguments are repaired, no selected IDs are inferred, and no invalid
request becomes a successful deletion.

The [focused development record](../selection-rejection-development-v1/README.md)
preserves two expected red failures followed by 14 passing service-error
cases and 58 adjacent cases. Tests include full SQLite-state preservation,
pending-outbox preservation, Hermes dispatch, valid apply, no identity leak,
and malformed/wrong-operation responses. Unexpected generic ValueErrors
retain HTTP 500. Product formatting and strict Python 3.10 type checks pass.

## Evidence limits

No native model has run this new package. V11 remains zero of three, and the
repair does not establish that the model will choose valid sources or recover
from a rejection. The package also includes a read-only graph source-relation
diagnostic and its 20 offline tests. It does not change production retrieval
or establish graph utility. Native continuity, complete capture, full-module
quality, Desktop, and the operational pilot remain open.

No live host configuration, source, or memory was changed. The evidence index
binds the raw files; this README is outside that index.
