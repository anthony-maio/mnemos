# Actual Claude Code plugin validation

Claude Code 2.1.191 with claude-sonnet-4-6 passed two fresh-session checks through
the unchanged plugin from PR #9 source `fcf02a2e`. The documented
MNEMOS_PLUGIN_PYTHON override selected Windows Python 3.11.13 and the installed
0.6.1 patch wheel. The plugin wrapper and hooks were not rewritten. Mock cognition
and simple embeddings served a disposable SQLite database.

The write session made namespaced mnemos_store and mnemos_inspect calls. The
restart session made mnemos_retrieve and mnemos_inspect calls. Both returned
manual chunk `e75d9d3b-62f1-470a-93c5-ab8cfe9ddbfa`, containing the synthetic
AZALEA-815 copper release-marker fact. Each session independently produced a
durable UserPromptSubmit hook row containing its prompt. The final database has
exactly one manual fact and two hook rows, all project-scoped to
synthetic-azalea-815, with no foreign-key violations.

The only connected MCP server was plugin:mnemos-memory:mnemos. Four actual tool
calls have matching successful results. Strengthened offline review verifies
unique call/result IDs, exact correspondence, project scope on every response
and durable row, inspect's returned ID, and exactly one durable manual fact.
These checks used the original streams; no host rerun was needed. Earlier runner
receipts remain intact alongside strengthened-audit-v2.json.

Each phase had a 180-second limit and USD 0.50 budget. Write took 19.83 seconds
and restart 21.94 seconds, including startup and cleanup; these are not retrieval
latency measurements. Reported model costs were USD 0.0371614 and USD 0.0374568.
Each job accounted for 64 processes and recorded zero active after cleanup.
Authentication copies were removed. Checked normal-profile files, source
credentials, plugin source and installed package were unchanged. No normal host
settings or live memory database was modified.

## Bindings and limits

Wheel SHA256:
`a3abff76729ea4cc0fa0e8fbba8a087d60e50b8fb5c2324e6222ff44e6ff4463`.

Private raw records remain under
`C:/Users/Ant/AppData/Local/Temp/mnemos-improvement-checks-20261106/storage-plugin-native-v1`.
The verified archive `.venv/storage-plugin-native-v1-evidence.zip` contains 18
selected evidence/harness files plus its manifest. Every member was read back
and hash-checked. No credential file or profile directory is included.
Archive SHA256:
`48fb050cd4a5239b75ab71fa08e7277043ca33c0e59401c73d35fe453c30973e`.

This establishes the actual Claude Code plugin explicit-runtime path and
UserPromptSubmit capture for this installed patch. It does not establish managed
first-install, PostToolUse failure capture, PreCompact/Stop effects, real-provider
semantic quality, Claude Desktop registration, correction/forget lifecycle on
this artifact, or daily-use pilot completion. Existing functional native evidence
remains separately bound to its own artifacts. No publication or live-pilot
change occurred.
