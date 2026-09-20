# Three-host v4: restarted recall passed, correction failed

The single scheduled attempt ran on September 20, 2026, from 08:02:34 to
08:06:28 UTC. Codex taught the project release label, Claude Code taught the
review convention, and Hermes retrieved both asserted facts after the service
restarted. These three phases passed their recorded checks. The Hermes
correction phase failed. The full three-host lifecycle remains unproven.

## What failed

Hermes first supplied an unsupported `scope` argument to retrieval. The
adapter rejected it without a write. A retry without that argument succeeded,
but the original rejection remains a protocol failure. Hermes then called
correction without the required inspection. The correction receipt committed
an event and pending job; three status calls still reported pending, with
zero processing attempts. The final database retained the original fact at
version 1. A committed receipt did not establish an applied correction.

Normalization stopped at the rejected tool result. Missing inspection and
unobserved job completion are additional failures visible in the raw trace.
The runner invokes explicit maintenance after a successfully normalized
native phase. Validation stopped before that step here. Status reads do not
process jobs, and these pending responses do not establish a production
worker defect. The prompt's completion request and the controller's later
maintenance need an offline timing check before another attempt. Corrected recall, forgetting, replay, and final recall did not run.

## Setup and controller record

The disposable Codex invocation acknowledged the model migration notice and
pinned `gpt-5.5`. The native turn record confirms that model, exactly one user
prompt with the expected hash, and matching direct model/MCP call IDs. The
v2 correlation failure and v3 model substitution remain preserved failures.

The controller trusted the fresh project and six generated hooks, approved
the requested teaching and consolidation operations, then used two Ctrl+C
characters after completion to close the client. No shutdown prompt was
added to the conversation. Claude and Hermes ran noninteractively.

Codex exited normally in 150.567 seconds and Claude in 21.263 seconds, both
inside their 180-second bounds. Hermes recall and correction took 22.814 and
29.041 seconds respectively. All phase cleanup and parent cleanup passed;
copied cloud authentication and the runtime descriptor were removed.
Separate Hermes discovery preflight passed without inference or writes.

All five cognition modules were disabled. The run used real Nomic embeddings
and the context-pinned local Qwen alias in an isolated patched Hermes host.
This is bounded integration evidence, not full-pipeline quality, complete
capture, general reliability, or daily-use acceptance.

## Binding and evidence

- Protocol: [v4](../../../plans/2026-09-20-three-host-continuity-protocol-v4.md).
- Runner revision: `f561f5c`; full revision and input hashes are in the reports.
- Product source: `f2f741443e60f838b392ae8ea1d1a0b4d8c6c1f2`.
- Wheel SHA-256: `8552345d2c17c4c037347961eb3043d13b20f790f54a9ccc86a096add6197b0f`.
- Preparation: 155 offline benchmark cases passed; this is separate from
  the earlier product package test totals.
- [Native report](native/report.json), [dispatch records](native/dispatch-records.json),
  [final state](native/final-state.json), and
  [correction trace](native/hermes-correct/trace.jsonl).
- [Evidence index](evidence-index.json) binds 164 files by bytes and SHA-256.

Selected cloud native records retain original tool and user lines with source
line/hash mappings. Model observations are independently bound to source
lines. Full cloud sessions, account metadata, copied authentication, runtime
tokens, and database binaries are omitted. Selected normalized evidence
omits full host instruction context and records that transformation. The
README and index itself are outside the index. Frozen inputs are retained
with their recorded runtime hashes and committed content parity.

Any successor must preserve this failure and the original criteria. Worker
progress and native inspection/correction behavior need separate evidence;
relabeling a pending receipt as successful correction would hide the gap.
