# Mnemos Release Checklist

Use this checklist before tagging a release candidate or making category-defining claims.

## Quality Gates
- `black --check .` passes
- `mypy .` passes
- `pytest -q` passes
- benchmark gate passes for required local store modes:
  - `memory`
  - `sqlite`
- Tier 1 MCP checks pass as real end-to-end flows:
  - Claude Code plugin flow
  - Claude Desktop stdio registration
  - generic stdio host harness

## Documentation and Contract
- `tests/test_docs_consistency.py` passes
- `docs/mcp-transport-contract.md` reflects current tool/resource contract
- `docs/client-compatibility-matrix.md` includes tested versions and caveats
- `docs/codex.md` reflects the current MCP + `AGENTS.md` setup path

## Operational Readiness
- `mnemos-cli doctor` reports expected readiness
- `mnemos-cli doctor` reports legacy unscoped chunk counts when present
- `mnemos_health` MCP tool reports readiness + backend stats + scope-isolation status
- structured logs verified for:
  - startup (`mnemos.startup`)
  - provider/storage failures (`mnemos.provider_failure`)
  - retrieval latency (`mnemos.retrieval_latency`)

## Replacement Claim Gate
- Claim-driving benchmark pack satisfies:
  - `gates.production_replacement.passed == true`
  - `MRR lift >= 15%`
  - `p95 latency ratio <= configured profile threshold`
- No unresolved blocker issues for scope leakage, startup failure, or data corruption

## Pilot Gate
- At least two external pilot users/projects complete 2 weeks of daily usage
- Zero blocker incidents for memory corruption, scope leakage, or startup failure
