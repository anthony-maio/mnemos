# Preview guidance full regression

The full suite failed on source `6364a38a4f9aa90f079d0feed548c82bb6e55367`,
with product code from `c80eb051a4e44d06d053f42d3a452e648e4eb99c`.
There were 3,547 passes, one failure, no errors and four platform skips
across 3,552 collected cases. Pytest reported 537.81 seconds and exited 1.

`test_competing_launcher_waits_past_dead_descriptor` failed with
`PermissionError: [WinError 5]` while replacing `runtime.json` atomically.
The publication code already retried sharing and access-denied errors for
at most 250 ms. This run did not instrument the handle that prevented the
replacement, so its owner cannot be established from the failure alone.

The follow-up reproduced the same error by holding an actual service or
hook descriptor reader open during publication. Adding delete sharing to
the reader alone still failed: the legacy Windows replacement call rejected
an open destination. A separate filesystem probe confirmed that extended
rename semantics allow the replacement when the reader shares deletion,
while an ordinary reader continues to block it. These observations guide
the follow-up fix; they do not change this run's failed result.

The audit verified unchanged tracked source, the explicitly selected
canonical website and all 208 restored historical fixtures. Both recorded
process identities had exited. The four skips cover two unavailable WSL
Hermes checks and two hook-command checks that require a POSIX shell.

The [receipt](full-regression-preview-guidance-v1-receipt.json) records the
counts and failure. The private archive has 10 members, 87,210 bytes and
SHA-256 `b4df757e88700ff77304459c7838911c1050d0c18f01700317d2f1139fe46e9d`.
It preserves the original console output, JUnit results, control record,
audit and execution helpers.

The earlier [native Codex v11 pass](native-codex-v11.md) remains bound to
its original product and installed artifact. This regression run did not
run a native host, activate the pilot, change live configuration or publish
anything. A passing full regression is still required after the fix.
