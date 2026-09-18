# Native Codex Stop boundary v1

This archive records one successful instrumented Codex Stop hook observation against the installed `5af3d15` package. The selected native MCP retrieve completed successfully, returned the seeded project fact, and excluded the foreign project. A read-only SQLite reopen matched the report's user and assistant capture rows by content hash, role, session, project, scope, and profile.

The marker sequence measures shell entry, Python bootstrap, adapter import, hook execution, and exit. The measured values are preserved in `timing-summary.json`: shell elapsed, shell-to-Python, adapter import, handler, and the service capture dispatch observation. The bootstrap adds file tracing and imports the installed adapter before calling it. Host dispatch to shell start is not measured, so these values are not an uninstrumented latency acceptance result.

This is transport and capture evidence for one fixture observation. It does not establish complete conversation capture, full native lifecycle acceptance, model quality, or a latency gate. Earlier native timeout evidence remains open. The fixture used disabled cognition modules and deterministic vectors. Credentials, full native sessions, and the SQLite database are not archived.
