# Compact recall and host token-counter inspection

Compact recall reduced tool reply text by 55.8% in the controlled transport
measurement while preserving every selected fact and qualification. It is
opt-in through `response_format: "compact"`; the full response remains the
default. This is a byte reduction, not a verified native-host token reduction.

The implementation at `e0ed29e1` keeps the complete `context` or
`evidence_context` string and supplies a reference index containing fact IDs,
versions, scopes, assertion status, source identifiers and source-availability
status. It removes duplicate fact content, source excerpts and retrieval scores
from the recall reply. `mnemos_inspect` still returns the full evidence. Neither
fact selection nor semantic support policy changes, and context-budget metadata
continues to describe the context string rather than the entire JSON reply.

The [protocol](../../plans/2026-11-07-compact-retrieval-v1.md) was frozen at
`c73acccaefe6dd0966d1cda5a55018517f180611` before the successful measurement.
An isolated SQLite service and actual stdio MCP child handled six synthetic
cases over three repetitions: one result, five results, zero requested results,
captured evidence, an embedding outage and conflicting assertions. Fixed
synthetic embeddings controlled candidate coverage. They do not establish
semantic retrieval quality.

All eighteen paired checks retained identical context, selected IDs, versions,
scopes, status, source references and degraded diagnostics. The 36 returned
fact occurrences stayed intact, including Unicode, literal special-token text,
negation and environment qualifications. Foreign-project content was absent.
All 36 follow-up inspections recovered the corresponding content and full source
evidence. Stored facts, embeddings and evidence were unchanged; retrieval access
metadata was allowed to advance.

| Representation over 18 replies | UTF-8 bytes |
| --- | ---: |
| Full, actual indented tool text | 71,289 |
| Full, derived without indentation | 56,631 |
| Compact, derived with indentation | 38,778 |
| Compact, actual tool text | 31,494 |
| Additional replies to inspect every reference | 70,423 |
| Compact recall plus those inspections | 101,917 |

The two derived rows separate field removal from whitespace savings; they are
not additional transport runs. Inspection of every reference makes the compact
workflow larger than full recall. Compact is useful when ordinary recall needs
the complete facts and their references, while detailed source review happens
only when needed. It is not a blanket cost reduction for evidence-heavy work.

Full recall had median/p95 roundtrip times of 50.4/121.2 ms; compact recall had
53.3/106.6 ms. Follow-up inspections added 393.9 ms in total. These small local
samples do not establish a latency improvement. The measurement covers tool text
and local service transport, excluding model inference, host prompt framing and
complete network envelopes.

The independent audit checked every captured reply, all 79 bound source files,
the unchanged stored snapshots, reference-to-evidence correspondence and byte
totals. The final run, session 44202, exited zero. Six recorded child process IDs
from the retained attempts were absent, and their three recorded service ports
were closed. No model calls were made.

The private archive also preserves the fixture failures. The first attempt's
foreign directory was inside the same Git project, so it correctly resolved to
that project and failed the fixture guard before a pair was recorded. The second
completed eighteen pairs, but its adaptive test embedder returned no captured
evidence and changed selections across repetitions. Those replies remain saved
and are not accepted as complete scenario coverage. The next setup omitted an
abstract embedding method and failed before seeding or starting an MCP child;
its cleanup reported the service port closed, without recording the port number.
The final setup uses fixed embeddings and checks coverage before measurement.

The host inspection found no advertised local arbitrary-text counting interface
in the inspected surfaces. Installed Codex CLI `0.164.0-alpha.11.1` generated
175 client methods, including `account/usage/read`, and the notification
`thread/tokenUsage/updated`. Usage updates are documented in the
[Codex app-server protocol](https://developers.openai.com/codex/app-server/);
they do not supply a counter for an arbitrary proposed memory context. Installed
Claude Code `2.1.191` likewise advertised no local counting command in its CLI
help. Claude documents a [remote token-counting API](https://platform.claude.com/docs/en/build-with-claude/token-counting),
which is separate from a local CLI tokenizer. No remote count request or new
authentication session was used. Binary hashes, help snapshots and the generated
Codex protocol are archived; this finding is limited to those inspected versions
and interfaces.

The existing identified `HostTokenizer` callback and conservative byte fallback
remain in place. No approximate encoding was relabeled as a verified Codex or
Claude tokenizer. Native-host token accounting, actual model consumption,
source-backed semantic span compression and fresh quality measurement remain
open. This change does not resolve the failed answer-support or model gates.

Verification passed 52 focused checks, including six new regressions, plus
twelve documentation checks. Strict type checking passed on four source files
and formatting passed on six. The [receipt](compact-v1-receipt.json) binds the
136-member private archive under SHA-256
`b3b45d32135ae6097dfc8f2f8874308228d51453100cd288ad240c6f49f47e97`.
The installed package, live configuration, publication state and inactive pilot
are unchanged. Pilot days remain zero.
