# Reconsolidation return installed-package checkpoint

Frozen product d937892 passes 811 installed cases
with 4 skips. The isolated runner verifies each selected test
outcome against the frozen source JUnit, along with installed module origins,
wheel bytes, input hashes and the replay contract. Captured correction merge regressions are included. The direct-return
reconsolidation test file was omitted from this selection; a separate expanded
installed check is required to validate that regression from the wheel. Counts and individual outcomes are preserved here.

This run uses fresh output, environment and replay paths.

This establishes the selected installed contract, not complete native-host
coverage, full-module quality, or pilot acceptance. The separate Hermes v35
archive proves its bounded installed correction/forgetting lifecycle. Nothing
was installed into live host configuration or published.
