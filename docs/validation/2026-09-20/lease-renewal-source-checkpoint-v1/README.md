# Lease renewal source/package checkpoint

Frozen checkpoint bee8357 (product change bfb4a32) passes 2,019 source tests
with four skips, formatting, Python 3.10 compatibility type checking,
JavaScript syntax, wheel/sdist build and twine metadata checks. All six
checks have exit code zero. JUnit outcomes and package hashes are preserved.

Active jobs renew a fenced lease without accumulating excess recovery delay.
Long-running work, competing owners, stale tokens, cancellation, retry and
late writes from non-cooperative processors are covered. The initial mutable
checkout run was deliberately stopped after the cancellation refinement;
only this clean frozen checkpoint is used for the full-suite claim.

Installed-package verification is separate. This does not close real-model
quality, native-host acceptance, provider recovery or the daily-use pilot.
