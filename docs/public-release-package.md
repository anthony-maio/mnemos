# Public Release Package

Use this document when preparing the first broad public OSS release for Mnemos.

## Recommended Release Framing

Use one of these labels:

- public beta
- release candidate
- early production evaluation for solo coding-agent workflows

Do not frame Mnemos as the definitive replacement for every agent memory tool yet. The software is credible enough for public adoption, but the pilot gate is still open.

## Approved Short Description

Mnemos is a local-first memory layer for coding agents. It keeps project, workspace, and global memory scoped correctly, persists locally, and integrates through MCP with Claude Code, Claude Desktop, generic MCP hosts, and documented Codex setups.

## Approved Claims

Safe to claim today:

- reliable scoped memory for solo coding-agent workflows
- local-first persistence with SQLite as the starter profile
- biomimetic retrieval and consolidation under the hood
- Tier 1 support for Claude Code, Claude Desktop, and generic MCP stdio hosts
- documented Codex support through MCP + `AGENTS.md`
- benchmark-gated retrieval improvements in the bundled claim-driving pack

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

## Pilot Gate Still Open

Before stronger production claims, Mnemos still needs:

- at least two external pilot users or projects
- two weeks of daily usage
- zero blocker incidents for scope leakage, startup failure, or data corruption

## Maintainer Ship Checklist

- confirm `python -m pytest -q`, `python -m mypy .`, and `python -m black --check .` pass
- confirm benchmark gates pass for `memory-core`, `starter-sqlite`, and `local-performance-qdrant`
- confirm Tier 1 MCP tests pass
- confirm [docs/public-beta-launch.md](public-beta-launch.md) matches the actual release posture
- confirm website, README, and changelog use the same release framing
- confirm GitHub issue templates and support docs are present
- confirm release notes do not outrun the compatibility matrix or pilot status
