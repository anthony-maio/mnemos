# Correction retry checkpoint

Source `ea88dd48e503adee811a524bf76923e153c17384` passes 1,201 tests with
four skips and all six source/package checks. Its isolated installed wheel
passes 513 contract cases with two matching skips and seven restart/replay
assertions. The six new cases run against both source and installed code.

Previously, a correction that finished before its acknowledgement reached
the client rejected its own retry because the target version had advanced.
The service now checks the durable job identity and payload before requiring
a current version for a new correction. The retry recovers the original job;
it does not apply the correction twice or overwrite a later edit.

The regressions cover actual HTTP acknowledgement loss followed by service
restart, newer corrections, processed-source retention expiry, pending
retries, changed content/target/version, reused identities across operation
kinds, foreign scopes, write policy, and forgotten targets. Five initial
regression cases reproduced the failure before the repair. A sixth case
checks two pending corrections against the same version: only the first
changes the fact, while the other's retry remains pending with its processing
error visible through the job endpoint. Durable acceptance does not mean the
correction has completed. The final processing and storage commit still
enforce expected versions.

These checks use fixture providers, actual HTTP, and SQLite. They establish
the correction retry contract, not native model tool use or extraction
quality. The independent model comparison remains pinned to `fc72cc2`; its
missing-target outcomes are separate from this repair and remain unchanged.

The wheel is 297,954 bytes, SHA-256
`cd7896291f51f8e5ddf75c8a56b314113e6c0dacbd54b08b12755ac10df2956c`.
Reports, JUnit cases, logs, dependency versions, and executed helper bytes are
preserved under the source and installed directories. `evidence-index.json`
binds 34 raw files; this README is outside that index. The website hash was
unchanged throughout source verification.

The first installed-check invocation passed a directory where the helper
requires the verification JSON file. Preflight stopped before installation
or tests. Its failure record is preserved under `invocation-error`; the
corrected invocation uses the same verified source and wheel with the proper
report path. No release, live host configuration, or live data changed.
