# Native Codex trace and scoped tool permission amendment v4

V3 remains failed. Code Mode reached two memory calls, but the host required
approval before dispatch while interactive approval was disabled. Neither call
completed. The old session normalizer also lacked the nested call records that
this runtime records in its separate local rollout trace. Preserve the original
attempt and its [receipt](../validation/2026-11-06/native-codex-v3-receipt.json).

Keep the product, wheel, pinned model, runtime executables, experimental defaults,
six single-prompt lifecycle phases and independent database checks from v1-v3.
Commit this amendment and all helper changes before the new attempt. Use a fresh
output directory. No live memory, installed host configuration or normal
credential file is changed.

## Disposable server permissions

The synthetic workflow already authorizes each phase's named memory operations.
Configure only those tools on the disposable `mnemos` server with
`mcp_servers.mnemos.tools.<tool>.approval_mode="approve"`. Retain the exact
`enabled_tools` phase allowlist and set the server default to `prompt`.
Interactive approval stays `never`, the shell sandbox stays read-only, and
unrelated tools remain disabled. No broad approval policy is relaxed.

These are documented per-server and per-tool options in the
[Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
They apply to the child command only. Validate their syntax with the pinned
runtime before inference. Managed requirements, if any, remain authoritative.

## Recorded call correlation

Require one schema-1 native rollout bound to the retained session identity.
Validate manifest paths, ordered event sequence, thread/turn identities, unique
IDs, matched start/end pairs and successful terminal statuses. Read only bounded
payload paths from that trace directory and retain every consumed file hash.

The model's recorded exec call ID must appear both in the retained session and
the raw inference response. The runtime's `code_cell_started` record must bind
that model ID to a unique runtime cell. Each MCP call must identify that cell
through its recorded requester, then have matching native invocation, runtime
start, runtime end and tool-result records. The exact server, tool, arguments,
turn and call IDs must agree. Structured and text MCP results must agree and
contain no error. Require every model cell and every tool call to be accounted
for, including discovery cells without tools.

Inference may finish streaming after a cell has begun. Resolve the complete
recorded ID graph after execution; do not infer a parent from timing or parse
JavaScript to invent tool calls. JavaScript equality across the model response
and cell is an additional integrity check only. Keep the historical direct-call
normalizer unchanged.

The acceptance receipt identifies correlation as
`native_model_id_to_cell_id_to_mcp_runtime_id`. A completed model turn or cell
cannot override a failed MCP call. Missing, ambiguous or unknown evidence fails
the attempt before the next lifecycle phase. Keep raw traces private because
they include host context. This amendment does not establish acceptance, pilot
use, publication or token efficiency.
