# Hermes native identity package checkpoint

Frozen source `3730fd96c3c50c7ef6179b2b9f70414ab5a8fb17` passes all six
source/package checks: 1,247 tests pass with four skips, Black, configured
Python 3.10 mypy, JavaScript checks, package build, and Twine validation.
An isolated installation outside the checkout passes all 530 selected cases
with two matching skips and seven Claude replay assertions. Imported module
origins and source/wheel parity are recorded in the installed reports.

Wheel: 298,300 bytes, SHA-256
`3772398a8e11b92d2cffd285b0571a733a9fd63cbdf932c30519d933cf0a6dd2`.
No live configuration, memory data, or published release changes here.

The native Hermes adapter preserves persisted row IDs through rewind and
truncated snapshots, retains exact retry timestamps, and rejects invalid IDs
or changed payloads under the same identity. Mixed identified/unidentified
snapshots remain provisional and cannot produce extracted facts. Seventeen
focused cases cover this behavior, including actual service/SQLite restart.

The separate [actual host database fixture](../hermes-native-rows-contract-v3/README.md)
verifies the row IDs supplied by Hermes. Its callback evidence is distinct
from this installed-package check and from a native model conversation.
ID-less fallback, some host loaders, compaction clones, and migration of
existing capture identities remain open. Native model lifecycle acceptance
and the supplemental comparisons are also separate gates.

`superseded-source-checks` preserves the preceding `30113a6` candidate's
passing six checks and 1,242 tests with four skips. Later review found its
native event identity still included role/content, allowing a changed payload
to acquire a new ID. The final candidate removes those fields from native
identity and passes explicit conflict regressions. Earlier passing checks
did not cover that defect.

`evidence-index.json` binds raw source/installed outputs and exact executed
helpers. This README is an interpretation of those immutable records.
