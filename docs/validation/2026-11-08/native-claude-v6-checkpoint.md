# Current-candidate Claude lifecycle attempt v6

The six-phase attempt on installed candidate
`ec4a36b262be61c16d807a2b39006bcad7f8367c` failed at the configured forgetting
budget and also missed required candidate inspections. This is not a
current-candidate full lifecycle pass.

Runner revision `570469df068996e13ad7e2dac43bb0818f9f4085` passed full regression:
4505 passed, 89 skipped, zero failures/errors, 760.12 seconds. Source, 208 frozen
fixture files and the explicitly selected website file remained unchanged.
An earlier run stopped on three optional-test collection errors; those original
results remain preserved. The repaired imports support package and standalone
discovery. A separate optional-runtime adapter check ran 11 tests with 2 skips.

The external installation matched all 98 candidate package files and 57 pinned
dependency versions, passed dependency checks, and passed installed CLI/MCP
restart-recall smoke validation. Candidate wheel SHA-256:
`ff7e97e1ed24a7122dd4094b9dcdd4ed2c5ece3e805ea3becca58e0e62502d5d`.

## Native outcome

| Phase | Result |
| --- | --- |
| Teach | Passed, 3 correlated calls |
| Restart recall | Passed, 2 correlated calls |
| Correct | Passed, 5 correlated calls |
| Corrected recall | Passed, 2 correlated calls |
| Reviewed forgetting | Failed: native budget exhaustion |
| Final restart, source replay and recall | Not attempted |

The forgetting stream reports `error_max_budget_usd`, with a cumulative phase
cost of $0.5058174 against the requested $0.50 limit. It exited with code 1
after 178.875 seconds; the runner did not time it out. The stream includes
16 tool invocations ending with `mnemos_forget_related`. Invocation alone does
not prove complete deletion. The phase did not produce a successful terminal
result, and subsequent replay/recall acceptance checks did not run. Preserve
this failed outcome even if offline inspection finds a committed deletion.

The wrapper and supervisor completed cleanup: zero remaining job processes,
copied credentials removed, watched profiles unchanged, frozen inputs and all
installed files unchanged, and exact runner HEAD preserved. The owner cleanup
receipt passed. No corrective prompt, retry or budget increase was used.

## Evidence and next step

Offline trace review subsequently found a committed apply receipt selecting
the target plus 18 additional facts and 21 source events. Only the target and
eight additional selected facts had separate inspection calls. Ten selected
candidates therefore lacked the required inspection; budget exhaustion was
not the only acceptance failure. No offline interpretation can turn this into
a passing native phase. Read-only database inspection found no remaining
target/release-label content in the inspected live tables, but the original
replay and fresh-host recall checks remain unattempted.

The decoded tool response text comprised 50,566 UTF-8 bytes across five preview
pages and 30,397 bytes across nine inspections. Lossless JSON whitespace removal
would reduce the inspection text to 26,782 bytes, saving 3,615 bytes; the preview
pages are already compact. These are response-byte measurements, not tokenizer
measurements or proven cost savings. The native modelUsage receipt reports
50,820 cache-creation input tokens, 213,028 cache-read input tokens and 9,130
output tokens. Repeated inspection calls and duplicate review presentation
deserve investigation; whitespace removal alone does not establish that a
complete review would fit the existing bound.

The five pages reconstruct all 76 fragments and match the recorded snapshot
digest. The eight additional inspection calls were already grouped into two
assistant messages. A bounded, lossless batch-inspection interface is a
candidate improvement, not evidence of avoided model roundtrips. Any such
interface must preserve per-fact identity, version and complete inspection
results, page within the native output bound, and require explicit coverage of
every selected fact without automatically selecting records. Acceptance must
continue rejecting skipped or stale inspections.

Private native evidence is under
`C:/Users/Ant/AppData/Local/Temp/mnemos-improvement-checks-20261106/native-claude-v6-ec4a36b2`.
Wrapper and supervisor receipts are `.venv/native-claude-current-v6-control.json`
and `.venv/native-claude-current-v6-job/process.json`. Regression results are in
`.venv/full-regression-claude-current-v6b`; the initial collection failure remains
in `.venv/full-regression-claude-current-v6`.

Inspect the preserved forgetting calls, actual apply result and durable state
before proposing another attempt. Distinguish required review work from
avoidable response or reasoning cost; do not remove inspection, source
selection, replay or retained-control requirements to obtain a passing run.
Quality, latency, actual pilot days and publication remain open obligations.
