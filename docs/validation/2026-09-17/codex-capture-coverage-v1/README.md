# Codex capture coverage regression

A capture hook with missing, empty, whitespace-only, or non-string text
previously returned `ok` despite capturing no event. The adapter now reports
degraded capture. Its CLI emits only sanitized metadata and keeps Stop
nonblocking. Intentional Stop reentry remains skipped without degradation.

Nine cases failed before the repair, with the reentry control passing. The
same ten cases plus existing adapter and generated-host checks passed after
the repair: 50 passed, two platform skips. Raw pytest logs and XML are retained.

This defect was found while investigating absent assistant capture in native
interactive v2. That host run did not preserve the Stop payload, so the
reporting repair does not establish its root cause or prove complete capture.
