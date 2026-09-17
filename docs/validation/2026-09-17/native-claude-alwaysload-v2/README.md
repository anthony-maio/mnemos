# Native Claude alwaysLoad startup control v2

This archive preserves one passed, isolated native Claude startup control. Claude Code 2.1.191 started the disposable plugin overlay with `mcpServers.mnemos.alwaysLoad=true`, observed a connected Mnemos server and all 14 advertised Mnemos tools, then made exactly one successful `mnemos_health` call. All nine boolean criteria passed in 13.667 seconds, including exact call/result correlation, fixture database identity, profile/project identity, default permissions, and completion.

The service used a real `ScopeRegistry`, a private SQLite database, explicit runtime descriptor/config/registry paths, unavailable cognition, and a deterministic constant embedding fixture. `MCP_CONNECTION_NONBLOCKING=1` was bound only in the native child environment. The overlay changed only `.claude-plugin/plugin.json`; source snapshot and copied plugin file hashes were checked before and after execution. Private auth, descriptor, database, and native streams remain outside this archive. The executable hash and version are recorded in `provenance.json`.

This is startup and MCP routing evidence for Claude Code 2.1.191. Hook capture was deliberately `not_run`, and the fixture makes no semantic retrieval, cognition, host-wide capture, or model-quality claim. The source pin is 8d865a5206aa99becdf6171c204161543ab60bb1. The native run used an old source snapshot with only the manifest overlay; it is not a whole-package proof for later source revisions.

The evidence index covers every archived file except the index itself, whose self-hash would be circular.
