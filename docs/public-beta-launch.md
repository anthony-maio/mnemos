# Mnemos Public Beta Launch

Use this document for the first broad public beta announcement.

## GitHub Release Title

`Mnemos Public Beta: Reliable Scoped Memory for Coding Agents`

## GitHub Release Notes

Mnemos is now available as a public beta.

Mnemos is a local-first memory layer for coding agents. It keeps project, workspace, and global memory scoped correctly, persists locally, and integrates through MCP with Claude Code, Claude Desktop, generic MCP hosts, and documented Codex setups.

This beta is not a “definitive replacement” launch. It is a serious public release meant to get the right users into the product, gather real feedback, and harden the system under daily use.

### What is in this beta

- scoped memory isolation for project, workspace, and global memory
- local-first persistence with SQLite as the starter profile
- biomimetic retrieval and consolidation under the hood:
  - surprisal-gated encoding
  - reconsolidation of stale facts
  - affective re-ranking
  - sleep consolidation
  - spreading activation
- Tier 1 validated MCP support for:
  - Claude Code
  - Claude Desktop
  - generic MCP stdio hosts
- documented Codex support through MCP + `AGENTS.md`
- `mnemos-cli doctor` and `mnemos_health` readiness diagnostics
- benchmark-gated retrieval improvements in the bundled claim-driving pack

### What we want feedback on

- scope isolation in real repo workflows
- retrieval quality on day-to-day coding tasks
- startup and persistence reliability
- host-specific MCP behavior
- Codex MCP + `AGENTS.md` workflows

### What this beta is not

- not a hosted memory platform
- not team memory or sync
- not universal Tier 1 support across every host
- not yet a claim that Mnemos replaces every memory tool in every setting

### Getting started

```bash
pip install "mnemos-memory[mcp]"
mnemos-cli doctor
```

Start with the SQLite profile, then move to Qdrant only when your dataset size or latency targets justify it.

Key docs:

- [README.md](../README.md)
- [docs/release-checklist.md](release-checklist.md)
- [docs/client-compatibility-matrix.md](client-compatibility-matrix.md)
- [docs/codex.md](codex.md)
- [docs/public-release-package.md](public-release-package.md)

### Contributing and reporting issues

If you hit a bug or a host compatibility issue, please open a GitHub issue using the new templates and include:

- `mnemos-cli doctor`
- `mnemos-cli stats`
- host and transport details
- store, LLM, and embedding provider details
- `scope` and `scope_id` when relevant

If you want to contribute, read:

- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [SUPPORT.md](../SUPPORT.md)
- [SECURITY.md](../SECURITY.md)
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)

### Known posture

Mnemos is ready for public beta adoption. Stronger production claims still depend on the open pilot gate:

- at least two external pilot users or projects
- two weeks of daily use
- zero blocker incidents for scope leakage, startup failure, or data corruption

## Short Announcement Copy

### X / LinkedIn post

Mnemos is now in public beta.

It is a local-first memory layer for coding agents: scoped project/workspace/global memory, persistent local storage, MCP support for Claude Code and Claude Desktop, documented Codex setup, and biomimetic retrieval/consolidation under the hood.

This is a serious beta, not a hype launch. I want feedback on scope isolation, real coding workflows, MCP host behavior, and Codex usage.

Repo:
https://github.com/anthony-maio/mnemos

### Short repo announcement

Mnemos is now in public beta. The current focus is reliable scoped memory for solo coding-agent workflows, with Tier 1 support for Claude Code, Claude Desktop, and generic MCP hosts, plus documented Codex setup. Feedback and focused contributions are welcome.

## Recommended Release Strategy

### Positioning

Launch this as a public beta for technical early adopters, not as a category-conquering GA launch. The right message is:

- reliable scoped memory for coding agents
- local-first and inspectable
- biomimetic internals as the differentiator
- honest about what is verified today

### Recommended channel order

1. Owned channels first

- GitHub release
- README and website already aligned
- pinned post on X or LinkedIn
- direct outreach to a small list of technical peers and early users

2. Rented channels second

- one technical post on X
- one technical post on LinkedIn
- one Hacker News submission only if you are ready to actively answer questions

3. Borrowed channels third

- targeted outreach to Claude Code, MCP, and agent-tooling communities
- ask a few credible users to try the beta and report compatibility issues

### Launch cadence

Day 0:

- publish GitHub release notes
- update pinned social post
- ask 10 to 20 targeted technical users to try the beta

Week 1:

- respond quickly to issues
- publish one short follow-up post with lessons or fixes
- collect pilot candidates

Week 2:

- summarize feedback themes
- fix the highest-signal bugs
- decide whether Codex can move closer to Tier 1 or stays documented-only

### What not to do

- do not launch on Product Hunt yet
- do not lead with “we replace ClaudeMem”
- do not sell team or hosted use cases
- do not claim broad production readiness before the pilot gate closes

### Success criteria for this beta launch

- 5 to 10 real technical users install it
- at least 2 external pilots start daily use
- issue reports arrive with enough diagnostics to reproduce
- no blocker incidents around scope leakage, startup failure, or data corruption
