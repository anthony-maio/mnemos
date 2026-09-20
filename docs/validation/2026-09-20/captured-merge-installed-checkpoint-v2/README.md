# Captured-correction merge installed-package checkpoint

Frozen product 26c9ddb passes 811 installed cases
with 4 skips. The isolated runner verifies each selected test
outcome against the frozen source JUnit, along with installed module origins,
wheel bytes, input hashes and the replay contract. The new captured correction
merge tests are included. Counts and individual outcomes are preserved here.

The first attempt failed before installation because the launcher supplied a
directory instead of verification.json. It remains in the source checkpoint
archive. This corrected run uses fresh output, environment and replay paths.
No test criterion or source code changed between those attempts.

This establishes the selected installed contract, not complete native-host
coverage, full-module quality, or pilot acceptance. The separate Hermes v35
archive proves its bounded installed correction/forgetting lifecycle. Nothing
was installed into live host configuration or published.
