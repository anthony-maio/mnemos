# Mnemos Release Checklist

Use this checklist before tagging a release candidate or announcing replacement-ready status.

## Quality Gates
- `black --check .` passes
- `mypy .` passes
- `pytest -q` passes
- benchmark gate passes for all required profiles:
  - `memory-core`
  - `starter-sqlite`
  - `local-performance-qdrant`
- Tier 1 MCP smoke checks pass:
  - Claude Code config validation
  - Claude Desktop config validation
  - generic stdio host harness

## Documentation and Contract
- `tests/test_docs_consistency.py` passes
- `docs/mcp-transport-contract.md` reflects current tool/resource contract
- `docs/client-compatibility-matrix.md` includes tested versions and caveats
- profile docs are current:
  - `docs/profiles/starter-sqlite.md`
  - `docs/profiles/local-performance-embedded-qdrant.md`
  - `docs/profiles/scale-external-qdrant.md`

## Operational Readiness
- `mnemos-cli doctor` reports expected profile readiness
- `mnemos_health` MCP tool reports readiness + backend stats
- structured logs verified for:
  - startup (`mnemos.startup`)
  - provider/storage failures (`mnemos.provider_failure`)
  - retrieval latency (`mnemos.retrieval_latency`)

## Replacement Claim Gate
- Claim-driving benchmark pack satisfies:
  - `gates.production_replacement.passed == true`
  - `MRR lift >= 15%`
  - `p95 latency ratio <= configured profile threshold`

## Pilot Gate
- At least one external beta user/project completes 2 weeks of daily usage
- Zero blocker incidents for memory corruption, scope leakage, or startup failure
