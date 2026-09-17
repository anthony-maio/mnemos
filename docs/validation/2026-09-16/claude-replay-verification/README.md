# Replay repair verification at 554a251

The frozen source `554a2517dc85bbdb6190b5e523f8511efa54cccf` passed the full
suite: **691 passed, two optional WSL tests skipped**, in 905.01 seconds.
Black, compatible strict mypy, JavaScript syntax, build, and Twine checks
also passed. The two skipped tests require the explicitly configured Hermes
interpreter; they are not counted as passing installed-host checks.

The wheel contains 264,332 bytes and has SHA256
`92dabd9bd79fa5e4f13685923f8ce2e1181fb5ab80bef72a2a5de9101d6a81a6`.
The sdist contains 2,353,452 bytes and has SHA256
`0c8a4eb6a262400fe921fa92150370650f2d1ee2029e972b1133aece7514c2c3`.
Build dependencies and commands are retained in the logs.

The wheel was installed into a new disposable environment, using explicit
target-interpreter arguments. The replay probe bound all 56 installed package
members to the wheel bytes and all 53 tracked package files to the source
after line-ending normalization. All seven positive assertions passed: two
events remained the same two events after reopening SQLite and replaying
Stop, with exactly one native assistant reference. Missing prompt identity
remained explicitly degraded.

This installed check uses real SQLite and the local service contract with
constant vectors and unavailable cognition. It does not establish native
Claude acceptance or extraction quality. The preserved source20f native
results remain separate, and unassociated fallback/native observations can
still produce degraded duplicate evidence.

The source comparison records byte equality for 14 memory-core files and 60
quality-harness/fixture files between frozen evaluation source `b7a9d9d` and
verification source `554a251`. Host, service, capture-contract, and performance
runner changes are listed separately. This is static parity, not proof of
whole-program or runtime equivalence; the evaluation retains its original
source binding and environment caveat.

A later Hermes discovery check found that the provider's availability check
does not use the environment-configured registry path. That separate repair
is pending. This full-suite pass does not certify that untested registration
path or complete the native host acceptance matrix.

The original verification record, logs, JUnit report, install records, exact
runners, source comparison, and installed replay report are indexed in
`evidence-index.json`. The long real-model evaluation ran concurrently, as
recorded in the shared workload log.
