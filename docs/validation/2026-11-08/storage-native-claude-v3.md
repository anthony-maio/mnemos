# Actual Claude Code direct-MCP persistence check

After the user refreshed the isolated validation login, a fresh two-session
Claude Code 2.1.191 run passed against the installed 0.6.1 patch wheel
`a3abff76729ea4cc0fa0e8fbba8a087d60e50b8fb5c2324e6222ff44e6ff4463`.
Claude executable SHA256:
`83397a6a029c7da663fb1ce27211e05174a3546d8b151e42451bf4590b8343d7`.
Both sessions used claude-sonnet-4-6 and direct MCP configuration, with a
mock/simple memory backend and disposable SQLite database. No plugin hooks ran.

The first session made exactly two actual tool calls: store and inspect. The
second fresh process made exactly two: retrieve and inspect. Results match the
exact synthetic fact, project scope and chunk ID across sessions. Independent
read-only SQLite queries after each phase verify the durable row. Final audit
finds one row and no foreign-key errors. Installed package bytes still match
the wheel. This is host-executed tool evidence, not a claim from assistant prose.

Both Claude processes exited successfully, the identified MCP children had
exited, copied credentials were removed and cleanup errors were empty. Normal
and validation credential-file hashes remained unchanged. Each phase retained
the 180-second and USD 0.50 bounds. The command uses a separate config/home and
does not load the user's normal MCP settings or change live memory databases.

Two preceding failures remain preserved:

1. v1 initialized with no tools and Mnemos pending. Claude emitted text claiming
   store/inspect success, but there were no tool calls and zero durable chunks.
   The validator rejected it and did not run restart.
2. Inspection of the installed Claude executable showed that explicit
   MCP_CONNECTION_NONBLOCKING=false selects blocking connection startup. v2
   applied that only in the disposable environment and required the initial
   catalog to contain connected Mnemos tools. Actual store and inspect calls
   succeeded, but the harness rejected Claude's outer result-string envelope.
   This remains a failed harness attempt, not completed workflow evidence.

v3 decodes only the exact single-field result-string envelope, then applies the
same content/scope/identity checks. Offline checks used both saved v2 responses,
an ambiguous envelope and malformed inner JSON. The fresh v3 run supplied the
passing two-session result; earlier attempts were not regraded. All three
attempts passed cleanup and credential-preservation checks.

Private evidence is `.venv/storage-native-evidence-v3.zip`, containing runners,
raw tool streams, process and cleanup receipts, MCP configuration/bootstrap,
execution logs, the decoder check and final database audit. Credential files
Archive SHA256: `3d18e5044c3960692470719161786e0b8514a80914e19e26a4d51621565c5609`.
Its 41 evidence files and manifest were read back and hash-verified. Credential files
are excluded and were absent from every disposable attempt directory. No raw
native conversation or authentication material is committed to source control.

This establishes only direct-MCP remember/inspect/restart-recall for the patch.
Claude Code plugin/hook acceptance, Claude Desktop, the separate functional
correction/forgetting lifecycle, supported-answer quality, complete latency and
the original daily-use pilot remain unresolved. No merge, tag, publication or
live configuration change occurred. Do not rerun this passing check without a
changed runtime or a distinct required scenario.
