# Hermes registry repair: first full verification failed

Frozen source `2360891b23f68606092add4593cd9c29d303ee1a` completed pytest with
699 passing tests, one failure, and two optional WSL skips in 989.75 seconds.
The verification runner stopped after pytest. Formatting, typing, JavaScript,
build, and package checks were not run in this attempt, and it produced no
verified wheel.

`test_cli_stops_isolated_service_and_releases_offline_lock` failed while
waiting for startup. The child exited with `TimeoutError: Mnemos service
dispatcher did not initialize` from `MnemosService.start()`. The existing
dispatcher deadline is five seconds. The CLI shutdown assertions were never
reached. This differs from the earlier source-d17 failure, where the child
remained alive and the test's 15-second descriptor deadline expired.

The exact failing fixture was copied before pytest cleanup. Its database had
43 schema objects, including the evidence and job tables. The older failed
fixture had stopped during early schema creation. A completed schema in this
copy does not identify where the new startup spent its time: cleanup can wait
for initialization to finish after the startup deadline has already elapsed.

The held-out Ollama evaluation continued concurrently. Timing is therefore
not an isolated-machine latency measurement, and overlap alone does not prove
the cause. The failure is retained for diagnosis; no timeout or durability
setting was relaxed. The focused Hermes tests and
[frozen-source plugin discovery](../hermes-registry-discovery/README.md) remain
separate from this failed broad verification.

`evidence-index.json` preserves the full test log, JUnit report, verification
record, exact runner, fixture inspection, and preservation helper. The raw
synthetic database remains in the private workspace fixture directory.
