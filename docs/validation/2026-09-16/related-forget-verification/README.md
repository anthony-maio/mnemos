# Reviewed captured-copy removal

Commit `c805cc25bce39ae8c2849c754303f5558f8ad7b3` adds a separate preview and
apply operation for captured copies of a remembered assertion. The existing
single-record action remains available. Additional facts and retained sources
require selection from a complete preview within the same project and profile.
Apply revalidates that preview inside the deletion transaction.

The operation follows current and historical assertion text, fact evidence,
and job references, including expired source IDs. It rejects shared unselected
dependencies, removes selected job payload content, invalidates leases, and
retains content-free replay suppression. Foreign source details are redacted.

## Completed checks

| Check | Observed result | Scope |
| --- | --- | --- |
| Affected tests | 83 passed | Related removal, jobs, service, backup, MCP, control plane, and UI routes |
| Expanded safety tests | 15 passed | Includes retained/expired sources shared through jobs, target redaction, and rollback across every SQLite table after reopen |
| Browser component | Six check groups passed | Actual UI code with synthetic API responses; selection, escaping, stale plans, bounds, cancellation, and late responses |
| Integrated browser v4 | Passed | Actual loopback HTTP, control plane, in-process service client, and SQLite; selected target/copy removed, unrelated fact preserved, restart and replay suppressed |
| Native Codex runner | Offline checks passed | Parser, selection, provenance, cleanup, and failure gates only; no native/model/network calls |

These are pre-freeze development checks with recorded source hashes. The
focused test file was subsequently expanded for the final safety cases and
received typing-only cleanup. The full test and package run is executing
separately against the clean frozen commit. This record does not claim that
run has passed or that a new wheel or native host lifecycle has been accepted.

## Preserved failures

The expired-source regression initially passed its two target cases and
failed both captured-copy cases. The operation committed despite an unselected
job dependency. The repair now checks expired candidate dependencies before
mutation; the expanded suite covers both retained and expired sources through
both `event_id` and `evidence_ids`.

Integrated browser v3 performed the deletion but failed an immediate DOM count
assertion before dashboard refresh finished. Read-only SQLite inspection found
only the unrelated record remaining. V4 waits for the target element to disappear
and then verifies reopen and replay. Product code did not change between these
browser runs. The local HTTP handler also logged a connection-aborted exception
during browser shutdown; the v4 lifecycle assertions passed.

The browser uses fixture embeddings and unavailable cognition, with module
switches disabled to reproduce raw-capture overlap. It is not a model-quality
or native-host test. Exact normalized assertion containment does not cover
paraphrases, future unseen restatements, external transcripts, or historical
backups. No live database, host configuration, or release was changed.

The evidence index hashes the preserved reports, logs, runners, and screenshots.
