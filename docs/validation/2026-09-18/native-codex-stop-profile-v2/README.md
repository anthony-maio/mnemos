# Native Codex Stop profile v2

This archive preserves one successful instrumented Codex Stop observation against the installed `5af3d15` package. The selected native MCP retrieve returned the seeded project fact and excluded the foreign project. A read-only SQLite reopen matched the report's user and assistant capture rows by content hash, role, session, project, scope, and profile.

The marker summary records shell entry, Python bootstrap, adapter import, hook execution, profile emission, and exit. `profile-top60.json` contains the 60 records emitted by the fixture's bounded profiler. The timing summary records shell elapsed time, shell-to-Python, adapter import, handler, and the committed service capture dispatch observation. File tracing and the installed adapter bootstrap add overhead; host dispatch to shell entry is not measured, so this is not an uninstrumented latency acceptance result.

The fixture's old-wheel `outbox.json.lock` observation has inherited access rules (`AreAccessRulesProtected=false`). A later source repair corrected the protection flag in separate tests; this v2 run does not verify that repair and must not be used as its evidence.

This is bounded transport, capture, and profiling evidence for one fixture observation. It does not establish complete conversation capture, full native lifecycle acceptance, model quality, full-capture acceptance, or a latency gate. Earlier native timeout evidence remains open. Credentials, full native sessions, and the SQLite database are not archived.
