# Native Codex UTF-8 capture checkpoint

Installed source1068a62 returned the project seed through an approved native
MCP call and captured both the user prompt and assistant answer through its
generated hooks. No forwarding wrapper or encoding environment override was
used. The assistant answer contains exact U+201C and U+201D quotes and matches
the retained native response byte-for-byte when encoded as UTF-8. Both events
were present after the service closed and a separate SQLite connection reopened
the database. The foreign seed had no access-count change.

The parent process again reached the 180-second shutdown watchdog. Its original
failed report is preserved. This verifies bounded Unicode capture and scoped
MCP retrieval; it does not close shutdown, native correction/forget, cross-host,
or model-quality acceptance. The fixture disabled modules and used deterministic
vectors with unavailable cognition. Database reopen is not a second native
host session.

Only the actual MCP completion and matching assistant response records are
retained from the native session. Captured content is represented by hashes in
the acceptance record. The private database, full session, authentication,
and unrelated host logs are excluded. Cleanup was checked before archiving.
