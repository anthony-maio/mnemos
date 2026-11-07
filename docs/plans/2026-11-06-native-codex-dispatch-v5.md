# Native Codex dispatch correlation amendment v5

V4 remains failed at the harness gate. Its first phase produced completed MCP
items for store, consolidate and job status, but the parser rejected a native
event absent from the earlier pre-dispatch failure. Preserve that attempt in
its original state and retain the [receipt](../validation/2026-11-06/native-codex-v4-receipt.json).

Successful dispatch emits `mcp_tool_call_correlation_assigned` between the
native runtime start and end. Its `tool_call_id` binds the existing Code Mode
call to a unique `mcp_call_id`. The host records no thread or turn on this
assignment event, so require its exact tool ID to match the already bound
runtime call, require one assignment per MCP call, require unique MCP IDs,
and require the assignment inside that call's runtime interval. Missing,
unbound, duplicate or late assignments fail acceptance.

All other v4 identity, result, permission, installed-artifact and six-phase
lifecycle requirements remain. Validate the new event handling against saved
v4 evidence and synthetic corruptions before inference. Commit the change,
use a fresh output directory and preserve v4's original failed status. An
offline trace parse is not completion of the unattempted lifecycle phases.
