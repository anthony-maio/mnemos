# MCP Stdio Transport Contract

Mnemos is MCP-protocol-first and host-agnostic. The compatibility contract is defined for standard `stdio` transport.

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
- Resource contract:
  - `mnemos://stats`
  - `mnemos://architecture`

## Runtime Behavior Expectations

- Startup:
  - one engine initialization per server process
  - startup emits a structured JSON log event (`mnemos.startup`)
  - degraded profiles emit `mnemos.degraded_mode`
- Reliability:
  - OpenAI/OpenClaw and Qdrant calls use shared retry/backoff policy
  - provider/storage failures are emitted as structured JSON events (`mnemos.provider_failure`)
- Retrieval observability:
  - every retrieval emits `mnemos.retrieval_latency` with `latency_ms` and `result_count`

## Health and Readiness

- CLI:
  - `mnemos-cli doctor` returns profile readiness, dependency checks, and recommendations
- MCP:
  - `mnemos_health` returns the same readiness report plus runtime store stats

## Compatibility Scope

- Tier 1 (contractual): Claude Code, Claude Desktop, generic MCP stdio hosts
- Tier 2 (best effort): Cursor, Windsurf, Cline

See [client-compatibility-matrix.md](client-compatibility-matrix.md) for current tested versions and caveats.
