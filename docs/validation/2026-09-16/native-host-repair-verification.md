# Native host repair verification

Source `d17dd249424f3e2c11b8fefee4a20bba8f50cf9f` repairs Claude capture when
its transcript is unavailable. The focused adapter suite passed 27 tests.
An installed package then passed the real Claude 2.1.191 capture and seeded
recall check: both conversation roles survived a service restart. The same
package passed the offline Codex Python and Windows shell transport check.
See the [native host record](native-host-smoke.md) for scope and package binding.

The first full suite failed: 638 passed, 1 failed, 2 skipped in 840.29 seconds.
`test_cli_stops_isolated_service_and_releases_offline_lock` did not observe its
runtime descriptor within the existing 15-second startup deadline. The child
had not exited. The test's shutdown assertions were not reached, and its
cleanup terminated the fixture process. The traceback does not establish
whether initialization, filesystem behavior, or scheduling caused the delay.
Other real-model work was running concurrently; that is context, not a proven
cause.

The failed fixture was retained before pytest cleanup. Its partial SQLite
schema contained only `memory_chunks` and the canonical-key index, plus the
table's primary-key index. This places the interrupted startup inside early
schema initialization. It does not identify which wait caused the delay. The
file hashes and schema inspection are preserved with attempt v1; a private
copy remains under `.venv/service-startup-failure-v1`.

The unchanged test passed alone in 4.68 seconds. Both outcomes are preserved
under [attempt-v1](native-host-repair-verification/attempt-v1/verification.json).
No timeout, assertion, or durability setting was relaxed. The independent retry
does not turn the failed full run into a pass. A second full run is in progress
against the same frozen source, writing to `.venv/native-host-repair-checks-v2`.

The first verification coordinator stopped after pytest failed, so its Black,
strict mypy, JavaScript, build, and Twine stages were not run. Earlier `b7a9d9d`
checks remain valid for that earlier source. This checkpoint is not yet fully
verified. No release or live data migration has occurred.
