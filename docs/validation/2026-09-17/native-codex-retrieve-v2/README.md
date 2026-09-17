# Native Codex retrieve v2 checkpoint

This archive preserves one bounded native Codex read probe against the installed Mnemos package bound to source commit `56182d97c5809a4909ae31f81bc6240be6cae58b` and wheel SHA256 `adc44b54ec31010f0ba841899b3e1b83d41f45f900b268561211db5d70f6b6df`.

The probe used one native `mnemos_retrieve` call with project-scoped arguments. The native stream records the call as failed with the result `user cancelled MCP tool call`. No retrieval result was returned, so this run is a failed contract observation and provides no recall or isolation evidence. The model-quality claim is none. No approval, trust, sandbox, or tool policy was bypassed, and no additional native call was made.

The report records `error_stage` as `native_spawn`, preserves the sanitized error digest, and records cleanup with the authentication copy removed and the descriptor absent. The retained native stdout and stderr contain only the reviewed native stream and diagnostic warnings. Private fixture files, Codex state, settings, SQLite databases, authentication data, and full transcripts are excluded.

The retrieve tool's `readOnlyHint` is false because retrieval updates access metadata. That annotation is compatible with ordinary access-counter writes and is not evidence of a cause for the client cancellation. The failure cause remains unresolved.

The archived runner is the executed v2 helper, even though its report's compatibility label says `native-codex-retrieve-v1`. Its SHA256 is recorded in the evidence index and matches the report's `runner_sha256`.
