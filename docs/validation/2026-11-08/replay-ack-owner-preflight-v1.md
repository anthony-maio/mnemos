# Replay acknowledgement owner preflight

The comparison owner now starts, captures, drains and stops against both exact
installed package targets prepared in [the package-pair report](replay-ack-packages-v1.md).
This is functional helper validation. The paired performance comparison has
not run; the runner, independent auditor and final input freeze remain pending.

The helper rejects missing or mismatched disposable config, registry, runtime
and descriptor paths before constructing services. Autostart must be disabled.
Its memory service reloads the registry, and the worker retains ordinary Git
resolution, privacy checks and FULL SQLite capture commits. It records real
acknowledgement intervals and hook responses in memory, flushing observations
outside command timing. A drain requires empty queues, completed jobs and idle
replay with no acknowledgement operation running.

Read-only database checks now close their connections explicitly. Cleanup
attempts both services even if one fails. If the memory owner remains alive
after a stop timeout, cleanup preserves its backing stores and reports failure
for the outer Windows job supervisor. Transient Windows control-file sharing
violations have a bounded retry; malformed JSON fails immediately.

Eleven focused tests pass, including missing isolation variables, a live owner
after shutdown timeout, cleanup continuation and control-file failure handling.
JUnit is retained at `.venv/replay-ack-owner-unit-v2.xml`. Black accepts both
tracked Python files. The earlier four-test result is preserved separately.

The final disposable preflight ran two consecutive native Stop commands in
each arm. Both arms produced two queued receipts matching reopened durable
event IDs and content, two completed jobs, an empty outbox, no active replay or
acknowledgement operation, clean owner exits and closed service ports. Package
members, helper and native bytes matched before and after. The outer Windows
job recorded 37 processes and zero active descendants after cleanup.

Final helper SHA256:
`55edb39a6318aa8521672e12d3349f91513c6cc10bf08535f1234afdf25c9fbe`.
Private raw evidence is under `.venv/replay-ack-owner-preflight-v2` and
`.venv/replay-ack-owner-preflight-job-v2`; the initial preflight remains under
the corresponding v1 paths. No latency distribution was computed, no acceptance
threshold was exercised, and no live memory or host configuration was changed.

Final read-only review found no remaining concrete owner-helper
blocker. The review distinguished server dispatch from successful HTTP delivery
and full acknowledgement-method intervals from isolated COMMIT duration; both
distinctions are now explicit in the protocol.

The next step is the scheduled runner and independent audit described in
[the comparison protocol](../../plans/2026-11-08-replay-ack-comparison-v1.md).
This preflight does not satisfy capture latency, semantic quality, native-host
acceptance or either pilot's usage requirements.
