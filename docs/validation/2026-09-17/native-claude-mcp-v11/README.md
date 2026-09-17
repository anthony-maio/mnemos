# Native Claude MCP read attempt

The v11 attempt used frozen source `6600dc2` and its verified installed wheel.
The actual plugin loader ran SessionStart, UserPromptSubmit, and Stop. The
native turn exited successfully in 23.229 seconds and reported $0.027801.
Captured events survived restart, reconstructed Stop replay added no events,
and the provisional identity contract passed.

The requested MCP read contract did not pass. Native initialization listed
the Mnemos server as pending, exposed no MCP tools, and no tool calls occurred.
The answer contained the seeded canary already available through hook context;
it does not prove MCP retrieval. The cause of pending initialization remains
under investigation. The earlier v10 connection observation is preserved
separately and does not supersede this failed attempt.

The new read contract accepts multiple positively associated assistant
transcript events, which are legitimate during tool use. Historical exact-two
criteria remain separate in the report. Parser controls reject wrong identity,
wrong canary scope, forbidden calls, and degraded/error replies. All validation
criteria were fixed before this native run. Raw report and executed helper
bytes are unchanged. Full native correction/forget continuity remains open.

The private authentication copy was removed and the runtime descriptor was
absent after cleanup. The archive excludes credentials and raw native output.
