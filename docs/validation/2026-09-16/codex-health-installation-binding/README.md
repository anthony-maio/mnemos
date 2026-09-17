# Codex health installation binding

This archive records the generated MCP health probe for the a38ee70d19f2ce8413d1ab7c9ce2ff26ffd07b46 wheel build. The offline v2 probe passed, while its native run failed after configuring the bare `mnemos-mcp` command. The read-only command audit confirmed that inherited PATH resolved the Microsoft Store Python user launcher, not the wheel launcher.

The v3 native attempt failed before its binding check because of a path comparison bug. V4 corrected that comparison and passed both offline and native health checks. V4 resolved `mnemos-mcp` to the wheel's `Scripts/mnemos-mcp.exe` and exposed the expected annotated health tool set.

These checks establish installation binding and one MCP health call. They do not establish full capture quality, cross-host behavior, model quality, or durable-memory quality. The production host launcher pin was repaired separately in commit 354bbb9; that repair is not a new wheel installation proof.

The helper copies use `.py.txt` so immutable evidence is not collected as active source by repository format and type checks. Reports and logs are copied byte-for-byte. Runtime databases, settings, auth files, and temporary fixture directories are intentionally excluded.
