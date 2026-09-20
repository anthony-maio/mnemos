# Native v7 text-to-receipt reconciliation

September 20, 2026. This is a new offline inspection of the retained
[three-host v7 run](../three-host-native-v7/README.md), not a new native run.
The product remains `e6887de`; no live configuration or memory changed.

All 31 text-bearing native messages have matching service acknowledgements:

| Host | Native text identities | Service arrivals | Additional arrivals |
| --- | ---: | ---: | --- |
| Codex | 6 | 6 | None |
| Claude Code | 17 | 21 | Four provisional prompt captures |
| Hermes | 8 | 16 | Eight repeated deliveries of existing identities |
| Total | 31 | 43 | Twelve additional arrivals |

Twenty-nine identities were acknowledged for processing and have completed
jobs in the final snapshot. Two Claude assistant messages were acknowledged
as suppressed during forgetting, with no new job. They are accounted for in
the receipt comparison, not counted as newly stored memories. Provisional
Claude arrivals do not substitute for native transcript identities.
Scripted replay operations are excluded from native message coverage.

## Method

The verifier reads expectations from the retained native sources independently
of service receipts. Codex uses actual user events and final responses with
the native turn ID, excluding injected user-role instructions. Claude uses
text-bearing transcript records and their UUIDs. Hermes uses durable native
`_row_id` values; the repeated transcript snapshots must agree. Textless tool
and thinking records are outside this text-capture contract.

Each acknowledgement must match phase, host, profile, project, workspace,
session, role, source kind, source reference, and exact normalized text. The
verifier independently reconstructs both the host event ID and service event
ID. A Hermes durable row keeps its identity when its content changes, so the
text comparison remains a separate requirement. Repeated receipts cannot
inflate message coverage, and conflicting identities cannot pass.

[Source bindings](source-bindings.json) record the hashes and sizes of all
11 full retained native files, their line hashes, and the safe projections
included here. Every full source was parsed and its expected messages compared
with the projection before saving. Projections retain native text, identities,
and turn boundaries; they omit injected instructions, thinking, tool payloads,
and credentials. The full source files remain outside this portable archive.
The projection builder is retained as `projection-builder.py.txt` for review.

The [report](report.json) keeps suppressed acknowledgements, provisional
arrivals, duplicate deliveries, and job status separate. Pending at dispatch
and completed in the final snapshot are different observations. This audit
does not require forgotten source records to remain in the database.

## Verification and limits

The focused suite passes 84 tests, including 28 new reconciliation cases.
Black and strict Python 3.10 mypy pass for the new verifier. The archive
preserves the failing guards and early matcher outputs: an initial version
did not recognize suppression receipts, and a later prototype used the
wrong Hermes event-hash contract. Those are verifier development failures,
not failures of the recorded product run.

Run `python docs/validation/2026-09-20/native-capture-coverage-v1/verify.py`
from this checkout to verify the indexed files and reproduce the report.
The verifier code hash is bound separately in the source bindings. This
command reproduces the projected-source comparison; it does not re-authenticate
the omitted full host files or rerun native hosts.

This closes the text-to-receipt accounting question for this one recorded
11-phase lifecycle. It does not establish exactly-once native callbacks,
arbitrary multi-turn capture, outage recovery, Codex Desktop, extraction
quality, or a daily-use pilot. All five cognition modules were disabled.
The wider capture and readiness gates remain open.
