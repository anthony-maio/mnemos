# Copyable forget preconditions package checkpoint

Frozen source `e6887deb89608dca34f7bbb463b5d0a3ba973011` passes 1,721 source tests
with 4 skips, all six source/package checks, 675 isolated
installed cases with 4 matching skips, and seven replay assertions.
Installed validation compares individual case outcomes, import origins, and
package bytes against the frozen source and wheel.

The archived initial installed-check invocation supplied a directory where
the runner required the report file. It rejected the input before executing
any installation command. The passing invocation uses the verified report.

Wheel: 304,845 bytes; SHA-256 `3d37b864d35916dbed0c22bcb0428a4e091d6b6c31a0177491130fa18da34fa1`.

## Changes and preserved protections

The [development record](../forget-apply-preconditions-development-v1/README.md)
records six new tests and 108 adjacent checks. Complete previews now return
the target ID, expected version, digest, and matching limit under the exact
apply-request field names. Incomplete previews return null preconditions.
The object makes no deletion selections and never repairs caller arguments.
Existing scope, version, digest, completeness, selection, and dependency
checks remain unchanged. MCP also now advertises preview as read-only;
a stray first docstring previously hid the correct description.

The first clean-checkout full run failed 15 benchmark tests because Git
converted a frozen JSON fixture to CRLF. The [portability record](../frozen-dataset-portability-v1/README.md)
retains that complete failure and the repair. Git attributes now preserve
the original bytes; the dataset, original hash, and strict check are unchanged.

## Evidence limits

The preceding [native v13 cohort](../hermes-native-cohort-v13/README.md)
finished zero of three complete lifecycles on `db21750`. It did not test this
package. Copyable fields are a model-interface hypothesis, not a native
reliability result. Out-of-preview selections must still be rejected.

No native cohort has run this checkpoint. Three-host continuity, complete
capture, full-module quality, Desktop, and the operational pilot remain open.
The role-aware graph result remains zero top-five targets with 18 failed
lifecycle controls. No live host configuration or memory data changed.
The evidence index binds raw files; this README is outside the index.
