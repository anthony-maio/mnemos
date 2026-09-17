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

Reopening a separate copy with the unchanged SQLite store completed schema
initialization in 1.866 seconds, produced schema version 7 with 43 schema
objects, and passed `PRAGMA integrity_check`. This [recovery observation](native-host-repair-verification/attempt-v1/startup-recovery.json)
shows that this interrupted synthetic initialization can recover. It does not
diagnose the delay or establish a startup latency guarantee.

The unchanged test passed alone in 4.68 seconds. Both outcomes are preserved
under [attempt-v1](native-host-repair-verification/attempt-v1/verification.json).
No timeout, assertion, or durability setting was relaxed. The intermittent
startup failure remains open even though the independent retry passed.

A second full verification against the same d17 source passed 639 tests with 2
explicit skips in 959.85 seconds. Its Black check, strict mypy check across 69
source files, JavaScript syntax check, wheel and source build, and Twine
metadata check all passed. The first 638-pass, 1-failure, 2-skip run and its
startup fixture remain preserved under attempt-v1; the second pass does not
diagnose or erase that earlier deadline failure. No release or live data
migration has occurred.
The second-run logs, XML result, package comparison, and tool checks are
preserved under [attempt-v2](native-host-repair-verification/attempt-v2/).
