# Mnemos Release Checklist

Use this checklist before tagging a release candidate or making category-defining claims.

## Functional local memory acceptance

The functional-local-memory implementation remains unreleased as of September 19, 2026. The [implementation status](validation/2026-09-16/implementation-status.md) records current observations, preserved failures, and gates still open. Before release:

- Freeze one source commit and validate its wheel outside the source checkout.
- Verify scope isolation, durable retry, restart, expected-version correction, forget/replay suppression, backup/restore, and reindex identity on the persistent service.
- Record native Claude Code, Codex, and Hermes evidence separately from adapter and MCP protocol tests. A configured host is not a successful conversation.
- Complete the versioned real-model baseline/full/five-ablation evaluation: at least 200 held-out sessions, 100 critical facts, 100 answerable queries, 100 no-answer queries, 50 state pairs, three repetitions, and a separate development set.
- Report extraction precision and critical retention against 95 percent, recall@5 and no-answer accuracy against 90 percent, and full-pipeline degradation against a two-percentage-point limit. Include Wilson and session-cluster bootstrap intervals plus missing/unjudged coverage. Preserve neutral and negative results.
- Measure warm recall p95 <= 500 ms at 10,000 facts and durable capture p95 <= 100 ms. Report cold startup separately. Do not weaken gates after measurement.
- Complete the two-week internal pilot on at least two projects; general-availability claims additionally require external pilot evidence.
- Website, paper, package, and release publication remain separate authorization steps. Historical synthetic gates below cannot replace these requirements.

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

## PyPI Publish Approval
- Publish to PyPI only through the GitHub Actions release workflow guarded by the `mnemos-release` environment.
- Before approving `mnemos-release`, verify:
  - the Git tag points at the intended release commit
  - `pyproject.toml` and `mnemos/__init__.py` match the release version
  - `CHANGELOG.md` and GitHub release notes match the shipped change set
  - `black --check .`, `mypy .`, and `pytest -q` are green for that commit
  - the built artifacts for that version were validated locally when needed (`python -m build`, `python -m twine check dist/*`)
- Cancel stale waiting PyPI publish runs before approving a new one.
- Treat direct `twine upload` as emergency-only. If used, cancel the waiting GitHub publish run for the same version so the environment does not later re-publish or fail on duplicates.

## Replacement Claim Gate
- Claim-driving benchmark pack satisfies:
  - `gates.production_replacement.passed == true`
  - `MRR lift >= 15%`
  - `p95 latency ratio <= configured profile threshold`
- No unresolved blocker issues for scope leakage, startup failure, or data corruption

## Pilot Gate
- At least two external pilot users/projects complete 2 weeks of daily usage
- Zero blocker incidents for memory corruption, scope leakage, or startup failure

## Product Hunt Readiness
- If launching on Product Hunt, confirm the pilot gate is closed or explicitly waived.
- Product Hunt traffic destination is live and matches the shipped install path:
  - `pip install "mnemos-memory[mcp]"`
  - `mnemos ui`
  - `Starter SQLite`
  - smoke check
- Website, README, and [docs/client-compatibility-matrix.md](client-compatibility-matrix.md) agree on host posture:
  - Claude Code is the strongest current path
  - Codex and Cursor are documented but secondary
- Listing asset pack is complete:
  - square logo or thumbnail
  - at least three gallery images
  - short demo video
- Product Hunt copy is finalized in [docs/product-hunt-launch.md](product-hunt-launch.md):
  - tagline
  - short description
  - first comment
  - launch-day playbook
- Launch outreach asks for feedback and discussion, not coordinated upvotes.
