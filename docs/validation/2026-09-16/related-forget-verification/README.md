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
| Frozen full package | 738 passed, two optional skips | Black, strict mypy, JavaScript, build, and Twine also passed on c805cc2 |
| Installed wheel | Passed | Exact wheel source/assets and Claude replay verified in both base and MCP-extra environments |
| Hermes discovery | Passed | Actual installed Hermes discovered the provider from this wheel; no model or service call |
| Generated Codex MCP transport | Passed | Initialized the generated server, listed lifecycle tools, and retrieved the scoped fixture fact |
| Claude adapter timing | Passed | Installed prompt, Stop, and Stop replay after reopen finished within their fixed ten-second deadlines |

The focused and browser checks are pre-freeze development checks with recorded source hashes. The
focused test file was subsequently expanded for the final safety cases and
received typing-only cleanup. Full verification then passed against the clean
frozen commit. The wheel is 273,064 bytes with SHA256
`7d723e1bbcd095dfb9eb06f1490e10ef2699db42a55263f95df6d4b3dfe27c12`.
Native host lifecycle acceptance remains open.

The instrumented Claude adapter subprocesses took 1,057.0 ms for prompt
capture/retrieval, 963.3 ms for Stop, and 4,730.1 ms for Stop replay after
reopen. They preserved two scoped events without replay duplication. These
are one local diagnostic run using fixture embeddings and unavailable
cognition. They do not establish native Claude latency or SessionStart behavior.

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

The first native Codex attempt used the base wheel installation, which omitted
the optional MCP SDK. Its service-API preflight passed but did not test MCP
startup. The native MCP handshake failed and no tools ran. The fresh
MCP-extra environment preserves the original installation, and the replacement
preflight requires an actual generated-server handshake and scoped retrieval.

With that preflight passed, a second native attempt discovered and attempted
store, consolidate, and retrieve. Every call returned the client error
`user cancelled MCP tool call`; no successful tool result was observed. The
runner stopped after that turn. The client cancellation is under investigation
and does not establish a completed native memory lifecycle or a service defect.

The browser uses fixture embeddings and unavailable cognition, with module
switches disabled to reproduce raw-capture overlap. It is not a model-quality
or native-host test. Exact normalized assertion containment does not cover
paraphrases, future unseen restatements, external transcripts, or historical
backups. No live database, host configuration, or release was changed.

The evidence index hashes the preserved reports, logs, runners, and screenshots.
