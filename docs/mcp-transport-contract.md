# MCP Stdio Transport Contract

The host-facing MCP contract uses standard stdio. The MCP process is a client of one authenticated loopback service, which owns SQLite, providers, and durable maintenance.

## Contract

- Server startup command:
  - `mnemos-mcp`
  - equivalent: `python -m mnemos.mcp_server`
- Transport:
  - `stdio` only (stdin request stream, stdout response stream)
- Tool contract:
  - `mnemos_store`
  - `mnemos_retrieve`
  - `mnemos_consolidate`
  - `mnemos_forget`
  - `mnemos_stats`
  - `mnemos_health`
  - `mnemos_inspect`
  - `mnemos_list`
  - `mnemos_capture`
  - `mnemos_correct`
  - `mnemos_job_status`
  - `mnemos_feedback`
- Resource contract:
  - `mnemos://stats`
  - `mnemos://architecture`

## Runtime Behavior Expectations

- Startup:
  - resolve an enabled project and registered user profile
  - connect to or start the shared authenticated service; do not open a competing writer
  - closing a stdio client leaves the shared service running
- Reliability:
  - OpenAI/OpenClaw calls use shared retry/backoff policy
  - provider/storage failures are emitted as structured JSON events (`mnemos.provider_failure`)
- Retrieval observability:
  - every retrieval emits `mnemos.retrieval_latency` with `latency_ms` and `result_count`

## Health and Readiness

- CLI:
  - `mnemos-cli doctor` returns profile readiness, dependency checks, and recommendations
- MCP:
  - `mnemos_health` returns observed service state, provider probe status, durable job counts, and store stats
  - a running process or configured provider class is not proof of successful input-specific extraction

## Durable operation responses

Tools return JSON-encoded strings with schema_version. Store and capture acknowledge an event and processing job; processing_status and mnemos_job_status describe fact completion. A queued outbox write is not a committed event. Retrieval returns versioned scoped facts, sources, and measured score traces; baseline is the default.

Correct requires fact_id, content, and expected_version. Forget supports expected_version; its legacy omitted-version path resolves the current fact through the service. Scope IDs must agree with the registered context. Caller metadata cannot override scope or provenance; custom mnemos_store metadata is currently ignored with an explicit metadata_status response.

## Compatibility Scope

Protocol coverage, adapter coverage, and a native host conversation are recorded separately. No support tier substitutes for installed-host evidence.

See [client-compatibility-matrix.md](client-compatibility-matrix.md) for current tested versions and caveats.
