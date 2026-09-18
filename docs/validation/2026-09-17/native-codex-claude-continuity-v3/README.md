# Native rerun interrupted by host shutdown

The v3 runner remains failed. Native Codex completed store and consolidate, but
did not exit within 180 seconds. The bounded taskkill call also timed out after
ten seconds, so the runner aborted before Claude recall or correction. The
cleanup report retained process_terminate:TimeoutExpired and process_still_running.
Authentication copies and runtime descriptor were removed. A later Win32 check
found the original native PID absent; that does not rewrite the earlier errors
or establish a clean exit. The interactive Stop hook also reported a three-second
timeout. Other local process/read commands stalled during this interval; their
relationship to this failure is unproven.

Selected native MCP completion records show two successful calls. Independent
SQLite reopen found the stored target at version 1. This is narrower than
cross-host acceptance: Claude phases and the second service restart were not run.

V3 changed only the v2 harness cardinality check: it requires at least one old
value record and qualification of every returned captured record, without
requiring exactly two old-value mentions. It never reached that revised check.
The v2 archive separately preserves observed qualification in actual Claude
context and its failed fixed-count assertion. No product code or retrieval
settings changed for this attempt. No live configuration or data was modified.
