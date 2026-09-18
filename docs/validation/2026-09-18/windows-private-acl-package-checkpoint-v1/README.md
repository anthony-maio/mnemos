# Windows private ACL package checkpoint

Frozen source `5d8cacf437e3f64dc1a956b69ce5ab5eea719ff3` passed 1,039 tests with four skips, Black, strict mypy, JavaScript syntax, build, and Twine. Source and fixtures were isolated on C:. The fresh installed wheel passed 369 selected cases with 2 matching skips and all seven synthetic replay assertions. The verifier compared individual case identities, outcomes, and skip reasons, and checked installed module origins before and after the run.

The installed selection includes nine real Windows ACL regressions for protected inheritance, exact-match write avoidance, drift repair, and query-failure repair, plus assistant assertion suppression, support preservation, advisory similarity discovery, explicit related forgetting, source dependencies, stale-vector rejection, scope isolation, and replay suppression, alongside UUID and host-context repairs. Dependencies, helper hashes, wheel/source parity, and selected fixture hashes are recorded.

The initial installed invocation supplied a directory where the verifier required verification.json. It failed before installation; that report is preserved. The corrected invocation used the unchanged runner and a fresh output directory.

These are package and protocol checks. Native host turns were not rerun by this checkpoint. Native Stop deadline acceptance for this repair, real-model evaluation, recall-latency gate, and operational pilot remain open. No live configuration, live data migration, or publication occurred.

Wheel SHA256: `59492c43ed2b4e5dafaaee0b02688c4eb706a3261a36ad93bcc5ea5849630e96`.
