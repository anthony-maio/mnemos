# Native Claude project revocation check

The installed candidate at `ec4a36b262be61c16d807a2b39006bcad7f8367c`
completed the three-turn project revocation workflow in one Claude process.
The original runner recorded failure after the workflow because it incorrectly
required one initialization report per process. A separate offline audit passes
with one initialization report per input turn, preserving the original failure.
No inference rerun was used to obtain this result.

## Observed behavior

- Turn 1: store the first assertion, consolidate, verify completed job, inspect.
- Turn 2: after external project disablement, retrieve, store and forget each
  return the exact uncommitted scope-denial payload. Counts and hashes remain
  unchanged across facts, events, jobs, evidence, revisions and suppressions.
- Turn 3: after re-enablement, inspect and retrieve the original assertion,
  store a second assertion, consolidate and verify its completed job.

There are 12 correlated tool calls, with no overlapping invocations or missing
results. The session identifier and recorded MCP process remain the same across
all three turns. The exclusive MCP bootstrap receipt supports a single startup.
Reopened SQLite contains the two expected facts and events in the exact project
and user profile, and two completed jobs whose results identify those facts.
SQLite integrity checking passes. There are no suppression or revision rows.

The pinned Claude 2.1.191 binary emits its initialization report inside its
per-message submission method. The raw trace has reports at lines 1, 69 and 187,
each identifying the same session, model and sole connected Mnemos server.
These reports are not evidence of process restarts.

## Isolation and evidence

The run used disposable profile paths, explicit MCP configuration, disabled
hooks, six allowlisted memory tools, disabled CLAUDE.md discovery and automatic
memory, and a $0.50 session budget. The final cumulative cost receipt is
$0.1514889. Alternate inherited authentication selectors were removed before
launch. The reviewed executable hash was checked before execution.

The copied credential was removed. Watched normal-profile files and the source
validation credential remained unchanged. The service stopped, its port closed,
and the owning Windows job reported zero active processes after cleanup.
All 98 installed package/resource hashes and frozen runner inputs match.

Private evidence is retained under `.venv/native-registry-session-v1`, with
supervision evidence in `.venv/native-registry-session-job-v1` and the outer
cleanup receipt `.venv/native-registry-session-control-v1.json`.
`offline-audit-v3.json` records the corrected audit and evidence hashes;
`native-result.json` and `result.json` retain their original failed status.
The original runner is preserved unchanged after the run. The separate auditor
is `benchmarks/native_hosts/registry_claude_audit.py`.

The focused trace, timeout, scope and denial-checker suite passed 25 tests before
execution. Static adversarial review identified and resolved isolation and
verification defects before launch; subsequent independent review checked the
raw trace and database without rerunning inference.

## Limits

This proves the focused external revocation/re-enablement workflow with the
installed candidate in the tested native host. It does not replace the complete
restart, correction and forget-without-resurrection acceptance workflow for the
current candidate. Unchanged table snapshots establish no observed changes in
those six tables, not absence of transient writes or revocation during an
already-running operation. Quality, latency, Desktop validation, the actual
14-day pilot, and release/publication alignment remain separate open gates.
