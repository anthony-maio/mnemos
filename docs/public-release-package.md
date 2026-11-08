# Public Release Package

Use this document when preparing a public release. The functional local-memory
candidate remains unpublished. These are drafting limits, not publication approval.

## Recommended Release Framing

For the current work, use:

- unpublished development candidate
- locally validated explicit-memory workflow, with quality and timing gates open

Do not announce a released beta or production readiness for this candidate.
The [release checklist](release-checklist.md) retains the required quality,
latency, host and pilot evidence before publication.

## Approved Short Description

Mnemos is a local-first memory layer for coding agents. It keeps project, workspace, and global memory scoped correctly, persists locally, and integrates through MCP with Claude Code, Claude Desktop, generic MCP hosts, and documented Codex setups.

## Approved Claims

Safe to claim today:

- local-first SQLite persistence with explicit project, workspace and global scope
- remember, source inspection, restart recall, correction and reviewed forgetting
  verified in the bounded native workflows listed in the compatibility matrix
- MCP integration and documented host setup, with per-host evidence and limits
- automatic enrichment and biomimetic modules available as experiments, disabled
  in the validated explicit-memory pilot configuration
- recorded negative quality and latency results; bundled synthetic mechanism
  checks do not establish real-model improvement

The optional explicit-memory pilot was activated after separate approval. No
qualifying days for the original two-week pilot are recorded here. Neither
activation nor the closed support experiments establish daily-use reliability.
See [current readiness](validation/current-readiness.md) for exact artifact
bindings and remaining gates. Earlier bounded workflow passes do not certify
the latest candidate, whose complete native lifecycle remains open.

## Claims To Avoid

Do not claim:

- definitive replacement for ClaudeMem, Mem0, or every other memory tool
- production-ready for every team, host, or deployment model
- team memory, hosted sync, or remote SaaS support in v1
- Codex Tier 1 support
- universal superiority over every competitor outside the shipped benchmark and host-validation surface

## Public Links To Include

Every release post, repo announcement, or launch note should point to:

- [README.md](../README.md)
- [docs/public-beta-launch.md](public-beta-launch.md)
- [docs/product-hunt-launch.md](product-hunt-launch.md)
- [docs/release-checklist.md](release-checklist.md)
- [docs/client-compatibility-matrix.md](client-compatibility-matrix.md)
- [docs/codex.md](codex.md)
- [SUPPORT.md](../SUPPORT.md)
- [SECURITY.md](../SECURITY.md)
- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)

## Feedback Routing

Use the GitHub templates for:

- bug reports
- feature requests
- host compatibility reports

Ask users to include:

- `mnemos-cli doctor`
- `mnemos-cli stats`
- host and transport details
- store, LLM, and embedding provider details
- exact scope and `scope_id` when relevant

## Pilot Gate

Before stronger production claims or Product Hunt positioning, Mnemos should have:

- at least two external pilot users or projects
- two weeks of daily usage
- zero blocker incidents for scope leakage, startup failure, or data corruption

## Maintainer Ship Checklist

- confirm `python -m pytest -q`, `python -m mypy .`, and `python -m black --check .` pass
- confirm benchmark gates pass for `memory` and `sqlite`
- confirm Tier 1 MCP tests pass
- confirm [docs/public-beta-launch.md](public-beta-launch.md) matches the actual release posture
- confirm website, README, and changelog use the same release framing
- confirm GitHub issue templates and support docs are present
- confirm release notes do not outrun the compatibility matrix or pilot status
