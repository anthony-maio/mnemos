# Native Codex hook payload v5

This archive preserves one bounded native Codex interactive diagnostic. It
used the installed wheel and source commit `30eaf4209bc6921b8a4a30f35ecd86fb68e1aae3`,
with the normal read-only sandbox, on-request approvals, and no bypass flags.
The fixture was private and synthetic, with all modules disabled and
unavailable cognition. No model-quality or complete-capture claim follows.

The run reached SessionStart, UserPromptSubmit, Stop, and SessionEnd. The
forwarder recorded eight entry/exit records. Stop had nonempty assistant text,
`stop_hook_active=false`, and a successful wrapper exit, but the adapter
returned no stdout and emitted the sanitized diagnostic
`mnemos hook diagnostic host=codex event=Stop status=error exception=UnicodeEncodeError`.
A separate local pipe probe reported cp1252 with surrogateescape. The
[decoder regression](../hook-utf8-regression-v1/README.md) reproduces the
observed exception by feeding UTF-8 text through that decoder. The v5
forwarder did not record its stream encoding, so that causal link combines
native observations with a separate reproduction. Native post-repair
acceptance remains unverified.

The process ended with `status=failed`, `returncode=1`, and
`timed_out=true` after about 180.5 seconds because the interactive shutdown
watchdog expired. Cleanup reported the copied auth absent and the runtime
descriptor absent. The shutdown failure and missing assistant capture remain
separate from the successful bounded MCP read.

The selected `native-mcp-event-line-17.jsonl` is the one authoritative
`event_msg` with `payload.type=mcp_tool_call_end`, server `mnemos`, and tool
`mnemos_retrieve`. Its result was `Ok` with `isError=false`, the requested
project scope, and the seeded fact. The service trace records the matching
query hash. The full Codex session, private logs, database, credentials, and
assistant transcript were intentionally omitted.

The report records three retrieval dispatches, one capture marked committed,
and maintenance. The fact guard observed the seeded positive fact and no
foreign access, but this does not establish complete assistant capture or
whole-database immutability. The source, wheel, helper, and generated-hook
bindings are preserved as hashes and `.py.txt` byte copies in this folder.
