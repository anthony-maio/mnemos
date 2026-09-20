# Native Codex stopped-service raw capture recovery

One scheduled attempt passed on September 20, 2026, from 15:21:59 to 15:22:48
UTC. Codex produced the exact synthetic user/assistant exchange while the
Mnemos service remained stopped. Both messages were queued in the disposable
outbox. An ordinary client health request after restart drained both entries
into SQLite, with two pending jobs and zero facts. Independent physical source
checks, same-event replay, another restart, receipt reconciliation, and cleanup
passed. All five cognition modules and background processing were disabled.
No local embedding or cognition response occurred.

## Bindings

- Verified product: `9722d92a5003a5dd47b0bf7846cdeede1b5a77f7`.
- [Package checks](../codex-runtime-environment-checkpoint-v1/README.md): 1,857 source tests and 703 installed cases pass, with four skips each; all six source/package checks and seven replay assertions pass.
- Wheel SHA-256: `59160aa424f4ad090dd8c2232c5a32ac87b128f22b13af68a7b7d808af6b23f3`.
- Runner: `fb25094f9ba5413fe510a0b382d211fd20367f33`, with committed [protocol](../../../plans/2026-09-20-native-codex-service-outage-v1.md) and input hashes in the report.
- Codex CLI 0.146.0, requested and observed `gpt-5.5`; code mode disabled and migration notice pinned as in three-host v7.
- Offline runner checks: 89 passed; Black and strict Python 3.10 mypy passed. Import/package preflight passed before native inference.

The controller trusted only the generated disposable project and hooks, then
exited with two Ctrl+C characters after the final answer, task completion, and
Stop hook. It sent no corrective model message or tool approval. Native
records contain no tool calls. Explicit runtime/storage/autostart settings
were present in generated MCP and hook configuration. The service descriptor
and copied authentication were removed; no live host configuration was edited.

## Evidence and limits

The index binds 24 artifacts, including raw service records, independent
SQLite projections, outbox payloads, generated configuration, exact experiment
sources, tests, and a safe projection of the native session. Full native JSONL
remains private outside the archive. Its byte hash and original line hashes
are retained. Omitted rows use placeholders to preserve native line positions;
injected instructions and reasoning are excluded. The archive builder checked
the full transcript before saving the projection and verified identical text
expectations. The offline verifier checks projected evidence; it cannot
independently re-authenticate omitted source bytes.

Run `python docs/validation/2026-09-20/native-codex-outage-v1/verify.py` to
reproduce the archive checks without contacting a model or service. README and
verifier are outside the index. Pending extraction was deliberate: this is one
raw-capture recovery, not fact extraction, provider-outage or in-flight crash
recovery, general reliability, Desktop acceptance, or a daily-use pilot.
The earlier Claude v1 default-store failure remains unchanged and preserved.
