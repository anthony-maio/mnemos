# Native Codex interactive MCP retrieve v2 checkpoint

This archive preserves a bounded interactive Codex 0.146.0 MCP
retrieve against the installed Mnemos wheel bound to source commit
`30eaf4209bc6921b8a4a30f35ecd86fb68e1aae3`. The wheel SHA256 is
`642cded3d32b486b73e781fb28b179a79394580e7b5570107b98db27a280e0af`.

The native process used inherited terminal input/output, normal on-request
approval, a read-only sandbox, and no hook-trust or approval bypass. Codex
selected an `exec` custom tool, waited for its completion, and returned a
successful `mnemos_retrieve` MCP result. The selected result contains the
seeded project fact with its recorded ID, version, scope, and content hash;
the foreign project fact is absent. Service tracing records retrieval arrival
and capture/maintenance activity was retained in the original sanitized
report. The native process later reached its bounded shutdown deadline, so
the original report remains `failed` with `timed_out=true`; that process
status does not erase the independently correlated successful MCP result.

`retrieval-acceptance.json` records the separately verified retrieval criteria.
The matching service dispatch is distinct from two hook-driven recalls.
Retrieval increased the positive seed's access count by three and left the
foreign seed's access count unchanged. Both seeds retained their content and
versions. User capture and maintenance also ran; this is not an unchanged
database assertion. Inspection found no assistant capture for this run.

Five raw JSONL records are retained: the custom `exec` call and its output,
the native `mcp_tool_call_end` completion event, and the `wait(cell_id=1)`
call and its output. The nested MCP results were decoded only for the archive
acceptance check. The full Codex session, prompts beyond the selected call,
reasoning, authentication data, databases, and plugin cache are excluded.

This is bounded native host and MCP transport evidence only. Modules were
disabled and the fixture used deterministic vectors, so this archive provides
no model-quality, automatic-capture, or full lifecycle claim.

The archive README and index were finalized after the initial archive helper
ran. The index uses the shared path-to-hash format, and includes the separate
retrieval verifier and acceptance result. Original report and selected native
event bytes are preserved.
