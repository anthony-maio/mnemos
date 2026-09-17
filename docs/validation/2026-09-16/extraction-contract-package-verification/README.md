# Extraction contract package verification

This archive records source and installed package verification for source commit `64471f5b12102e6ba94079fb74417b087223bc24`. The package checks report 776 passed and 4 skipped. The skips are the two prior WSL checks and two new POSIX checks.

The recorded checks cover Black, strict mypy, JavaScript build, Twine metadata, wheel installation, Claude replay, package parity, and extraction evidence contract behavior. The v5 Codex health checks passed offline and through the native Codex path. Their generated MCP configuration used the exact installed interpreter with `-m mnemos.mcp_server`; no PATH repair or approval override was used.

This evidence establishes package installation, MCP health, extraction contract validation, and replay behavior for the pinned build. It does not establish full automatic capture coverage, model quality, cross-host quality, or general memory quality. The logs and reports are preserved without distribution binaries. Auth files, settings, databases, and temporary fixtures are excluded.
