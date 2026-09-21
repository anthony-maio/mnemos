# Evaluation replay v6 source/package checkpoint

Frozen source c1e2217 passes 2,081 tests with 4 skips and
all six source/package checks. All 22 replay tests and four extraction evidence
resolution tests are included and pass. Formatting, Python 3.10 type checking,
JavaScript, build and package metadata checks pass.

The v6 path adds original-source replay while retaining lifecycle/scope/body/state checks and global source
identities while leaving historical dataset rules intact. No fresh dataset or
unseen evaluation is established by unit tests that reuse old contract fixtures.
Original model outcomes, labels and thresholds remain unchanged.

Wheel byte equality with the previously verified installed artifact: True.
See wheel-comparison.json for exact hashes. No new installed run is claimed.
Full semantic quality, module benefit, provider recovery and pilot acceptance
remain open. The independent extraction comparison uses its own frozen source.
