# Complete installed Windows HTTP transport results

The Windows transport candidate improves Stop command latency in this
comparison, while prompt submission shows no consistent improvement. Both
complete hook paths still fail the 100 ms requirement. The candidate stays
local; no live host configuration or pilot was activated.

Candidate `788766aa` uses the Windows HTTP API for the existing loopback worker
endpoint. It keeps ordinary Python initialization, authenticated request and
environment binding, response validation and the original adapter fallback.
Native setup must disable redirects, cookies, automatic authentication,
connection reuse and failed-connection resends. If those controls are unavailable
before sending, the client uses the original Python transport. After sending,
an uncertain result never permits another dispatch. The required controls follow
Microsoft's [WinHTTP options](https://learn.microsoft.com/en-us/windows/win32/winhttp/option-flags)
and [failed-connection retry contract](https://learn.microsoft.com/en-us/windows/win32/api/winhttp/ns-winhttp-winhttp_failed_connection_retries).

The new tests exposed a short-body case where WinHTTP returned EOF without
rejecting the declared Content-Length mismatch. The implementation now requires
the complete bounded body. The original failed test log is retained. Forty new
checks and 48 related checks pass, including real Windows requests, no redirects
or authentication retries, UTF-8 bytes, size limits, header-injection rejection,
partial-setup cleanup, lost acknowledgements, scope checks and durable capture.
Black and Windows strict mypy pass for the changed modules.

The measured wheel builds with Hatchling 1.33.0 and passes Twine. SHA-256:
`527f07d69047e1f933609217e4d70c06e784fa93b7e74dba2331141b7a10e834`.
Its 95 installed package files match the artifact, and all 84 Python sources
match the measured checkout. Baseline `5735c502` and candidate use the same
57 dependency versions, 17 model assets and pristine 10,000-fact seed.

The prospective order was baseline-0, candidate-0, candidate-1, baseline-1.
Only one service/worker pair ran at a time. The run retained 320 observations,
including 300 timed samples, and took 657.15 seconds including preparation,
supervision, restart checks and final verification.

| Complete path | Baseline median ms | Candidate median ms | Baseline p95 ms | Candidate p95 ms | Required p95 ms | Candidate result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Recall | 138.34 | 123.38 | 183.92 | 179.36 | 500 | PASS |
| Stop | 402.43 | 324.19 | 607.99 | 549.12 | 100 | FAIL |
| UserPromptSubmit | 637.62 | 639.68 | 1162.05 | 1162.74 | 100 | FAIL |

Stop was faster in 40 of 50 pairs, with a median paired change of -81.81 ms.
Prompt submission was faster in 26 of 50 pairs, with a median paired change
of -2.24 ms. That does not establish a prompt improvement. Recall uses the
unchanged service/client path, so its timing difference should not be attributed
to the hook transport. Candidate maxima were 360.60 ms for recall, 1151.90 ms
for Stop and 1437.50 ms for prompt submission.

| Block | Recall median / p95 ms | Stop median / p95 ms | Prompt median / p95 ms |
| --- | ---: | ---: | ---: |
| baseline-0 | 141.88 / 232.68 | 404.35 / 626.29 | 656.13 / 1324.62 |
| candidate-0 | 143.23 / 201.97 | 330.27 / 564.39 | 598.64 / 930.11 |
| candidate-1 | 117.09 / 122.06 | 321.36 / 498.98 | 715.11 / 1236.32 |
| baseline-1 | 133.70 / 142.98 | 394.58 / 520.01 | 624.95 / 1058.18 |

The Stop median improvement appears in both halves. Prompt behavior varies by
block. The reversed order does not eliminate machine-load or maintenance-timing
effects, and this run does not isolate their individual costs. No startup time
was subtracted, normal maintenance disabled, retrieval skipped or context cut.

All 160 decoded response/context pairs match exactly, including warmups. All
208 commands committed their source/job records and retained them after service
restart. Every capture job completed. Each block ended with its original
10,000 accepted facts plus 52 captured chunks outside that accepted cohort.
Accepted facts, source evidence and vectors remained unchanged; SQLite integrity
and foreign-key checks passed.

All 104 candidate commands used WinHTTP without importing Python's HTTP parser.
All 104 baseline commands used the original parser. No successful command took
ordinary adapter fallback. Twelve private-policy preflights passed. All twelve
owned service/worker lifecycles ended normally, with descriptors removed.
Settings and environment were recorded only as hashes. This comparison does
not establish current native Codex/Claude acceptance or actual daily use.

The independent audit reconstructed every result and verified the source,
artifact, database and model bindings. Its first process check failed because
Windows had reused one numeric process ID. Revision 2 checks creation times:
237 recorded IDs were absent, and the remaining ID belonged to a console process
created after the benchmark ended. The original failed checker and log remain
in the archive. No process was stopped during either audit.

The verified private archive contains 201 members representing 366 logical
files. SHA-256:
`4cdff5ff97be839140bd9945d786da97043e3ca57f19852e9b46f6199e4873e0`.

A separate portability fix at `cddf8adc` defers three Windows-only ctypes
attribute lookups. Sixty-two focused checks pass, including the non-Windows
transport route; Black and strict mypy pass with both Windows and Linux platform
settings. This is not a Linux runtime run. The follow-up wheel builds, passes
Twine, matches all 95 installed package files and 84 Python sources, and passes
a real installed Windows request without importing the HTTP parser. SHA-256:
`c6f6ff3ddf260a2a736867f01ce923879d68e2dd91ff44abbf0f7a22e1e6982f`.
Its [verification receipt](native-hook-portability-v1-receipt.json) records the
separate artifact and archive. The performance numbers above remain bound to
the unchanged `788766aa` wheel; the follow-up was not benchmarked again.

The [subsequent full suite](full-regression-native-hook-v1.md) passes 3,495
checks with four platform skips on the portability follow-up. Fresh
answer-support quality, current native acceptance, remaining context work,
canonical delivery and the two-week private pilot remain open.

- [Prospective protocol](../../plans/2026-11-07-native-hook-performance-v1.md)
- [Verification receipt](native-hook-performance-v1-receipt.json)
- [Prior complete comparison](scoped-rank-performance-v1.md)
- [Startup diagnostic](hook-startup-v1.md)
