# Three-host v3: model substitution rejected

The single scheduled v3 attempt failed in Codex setup and execution on
September 20, 2026, from 07:51:06 to 07:52:48 UTC. The command pinned
`gpt-5.5`, but a startup migration dialog selected `gpt-5.6-sol`. The
controller's intended existing-model selection did not take effect.
The new post-run model check rejected both observed turns. No Claude or
Hermes model phase ran, and this result establishes no memory lifecycle pass.

The Codex child exited normally in 93.604 seconds, below the 180-second
deadline. Copied authentication was removed, no child cleanup error was
reported, and the runtime descriptor was removed. Parent cleanup passed.
The separate v3 Hermes import preflight passed without inference or service
writes. The committed preparation passed 150 offline cases, Black on four
changed files, and strict Python 3.10 mypy on three changed benchmark modules.

## What happened

The controller accepted the disposable project and generated hooks, then
sent `2` and Enter intending to select the existing model. The displayed
model changed to `gpt-5.6-sol`. The controller interrupted the turn; the
native MCP store and consolidation calls were cancelled. There was no
committed explicit memory write or completed teaching job. Raw hook captures
did commit and remain in the preserved final state.

The attempted `/exit` shutdown text became a second user prompt and was
captured by the hooks. Two Ctrl+C characters then shut down the client.
These controller mistakes are part of this failed attempt. They are not
native forgetting failures or evidence of memory reliability.

The exact input-delivery cause is unresolved. The version-matched
[migration dialog source](https://github.com/openai/codex/blob/rust-v0.146.0/codex-rs/tui/src/model_migration.rs)
supports numeric selection, so do not claim that numeric input is unsupported.
The saved native configuration and both turn-context records prove the
actual model substitution.

## Binding and evidence

- Protocol: [v3](../../../plans/2026-09-20-three-host-continuity-protocol-v3.md).
- Runner: `3f77cbb`; complete revision and input hashes are in each report.
- Product source: `f2f741443e60f838b392ae8ea1d1a0b4d8c6c1f2`.
- Wheel SHA-256: `8552345d2c17c4c037347961eb3043d13b20f790f54a9ccc86a096add6197b0f`.
- Codex 0.146.0, requested `gpt-5.5`, observed `gpt-5.6-sol` twice.
- `evidence-index.json` binds 104 selected files by exact bytes and SHA-256.

Selected native lines retain all tool call/output, MCP completion, and user
message records, with a map to original line numbers and hashes. Original
session and model observations are bound by file and line hashes. Full
session instructions, account metadata, copied authentication, and the
database binary are omitted. The final state and service dispatch records
are included. Input copies are verified against the recorded runtime hashes
and committed inputs, allowing only CRLF/LF representation differences.
The README and index itself are outside the index.

## Next controlled change

The version-matched
[startup code](https://github.com/openai/codex/blob/rust-v0.146.0/codex-rs/tui/src/app/startup_prompts.rs)
skips the migration dialog when the notice map already acknowledges the
current model and proposed target. Rejecting the dialog writes that same
acknowledgment without changing the selected model. A prospective successor
will supply only that notice acknowledgment in its disposable invocation,
retain the explicit model pin and actual-model check, and preserve all
tool permissions, sandbox settings, deadlines, and native ID requirements.

Both v2 and v3 remain failed. Three-host continuity, complete capture,
Desktop, module quality acceptance, and the operational pilot remain open.
