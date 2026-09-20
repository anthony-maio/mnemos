# Job lookup error package checkpoint

Frozen source `68693f83b5894198daea114e018031e381a508a3` passes 1,308 source tests with
4 skips, all six source/package checks, 533 isolated
installed cases with 2 matching skips, and 7 replay assertions.
Wheel: 298,409 bytes, SHA-256 `2e121c1ae81717d1b9a32349bd27ea7fafc2e5562e6eadbd7d24ddecd16f5cb8`.

Missing and inaccessible jobs now return the same `404/job_not_found` response.
A dedicated `KeyError` subclass preserves direct caller behavior, while
unexpected storage errors remain generic HTTP 500. Tests cover a mutated ID,
foreign project and profile, valid retrieval, and unchanged job state after
rejected lookup/retry. The initial focused reproduction failed both missing-job
HTTP cases; the repaired focused suite passed 48 cases.

This repairs the error classification exposed by the retained native Hermes
v2 failure. It does not repair the model's copied argument, complete a pending
job, or establish a native lifecycle pass. The next native attempt and the
three-host observation remain open. Live hosts and memory data are unchanged.

`interrupted-source-checks` preserves an earlier verification attempt launched
with the older helper. It was stopped because that helper omitted the fresh
external temporary root and bound website input. Its original report remains
in its last running state; `interruption.json` explains why it supplies no
acceptance result. The successor used the current v3 verification helper.

The evidence index binds source/installed logs, reports, and executed helpers.
