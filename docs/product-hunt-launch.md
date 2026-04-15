# Product Hunt Launch Package

Use this document when Mnemos is ready for a public Product Hunt launch.

The goal is not to reposition Mnemos as universal memory infrastructure. The goal is to present the current product honestly and convert Product Hunt traffic into real installs, trials, and technical conversations.

## Launch Positioning

Launch Mnemos on Product Hunt as:

- public beta
- open source
- local-first
- scoped memory for solo coding-agent workflows

Do not position this launch as:

- a hosted memory platform
- team memory or sync
- universal Tier 1 support across every host
- a definitive replacement for every competing memory tool

## Launch Gate

Before scheduling the launch:

- the pilot gate is closed:
  - at least two external pilots
  - at least two weeks of daily usage
  - zero blocker incidents for scope leakage, startup failure, or data corruption
- the website, README, and compatibility docs all reflect the same posture
- the primary install path works from a clean machine:
  - `pip install "mnemos-memory[mcp]"`
  - `mnemos ui`
  - `Starter SQLite`
  - smoke check passes
- the Product Hunt destination URL is live and usable without a waitlist

## Approved Listing Copy

### Recommended Product Name

`Mnemos`

### Recommended Tagline

`Local-first scoped memory for coding agents`

### Alternative Taglines

- `Inspectable local memory for Claude Code and MCP hosts`
- `Scoped SQLite-backed memory for coding agents`
- `Local memory that keeps coding-agent context separated`

### Recommended Short Description

Mnemos is a local-first memory layer for coding agents. It keeps project, workspace, and global memory scoped correctly, persists in one SQLite database, and integrates through MCP with Claude Code, Claude Desktop, generic hosts, and documented Codex setups.

### Claims That Are Safe On Product Hunt

- local-first memory with a single SQLite database
- scoped project, workspace, and global memory
- inspectable retrieval and consolidation
- Claude Code as the strongest current path
- documented Codex and Cursor setups with honest support tiers
- biomimetic retrieval and consolidation as the differentiator under the hood

### Claims To Avoid

- best memory product
- replaces every memory tool
- universal support across all agent hosts
- production-ready for teams
- hosted sync, remote SaaS, or collaborative memory

## Listing Asset Pack

At minimum, prepare:

- one square logo or thumbnail
- three gallery images
- one short demo video

Recommended gallery sequence:

1. First-run setup

- `pip install`
- `mnemos ui`
- `Starter SQLite`
- smoke check

2. Inspectability

- why a memory was stored
- why a memory was retrieved
- visible scope and provenance

3. Day-two continuity

- recalled repo convention or constraint
- updated stale fact or consolidation summary
- visible “this is why memory helped” outcome

Recommended video:

- 30 to 60 seconds
- no narration required, but captions help
- show install -> `mnemos ui` -> choose preset -> apply host config -> smoke check -> one useful recall

## Launch Page Requirements

The Product Hunt traffic destination should answer these questions above the fold:

- what is Mnemos
- who is it for
- how do I start in 2 minutes
- what works today
- why should I trust it

The website and README should reinforce:

- SQLite-first
- inspectable
- Claude Code primary
- Codex and Cursor documented but secondary
- public beta, not universal GA

## Recommended First Comment

Use the first comment to frame the launch honestly:

> Mnemos is a local-first memory layer for coding agents. I built it because most memory systems either mix scopes together or accumulate stale facts forever. Mnemos keeps project, workspace, and global memory separate, stores everything locally in SQLite, and exposes inspectable retrieval and consolidation through MCP. Claude Code is the strongest path today; Codex and Cursor are documented and usable, but not at the same validation tier yet. I would especially value feedback on first-run setup, recall quality in real repo workflows, and host-specific rough edges.

## Launch Day Playbook

Before launch day:

- finalize the listing copy and visuals
- ensure the website and README match the Product Hunt tagline
- prepare answers for likely questions:
  - why SQLite
  - what makes this different from ClaudeMem or Mem0
  - what hosts are truly supported
  - why the neuroscience framing matters
- line up owned-channel posts:
  - GitHub release or pinned repo note
  - X or LinkedIn post
  - direct outreach to existing users

On launch day:

- be available all day for comments
- reply quickly and concretely
- drive people to the install path, not a vague waitlist
- ask for feedback and discussion, not coordinated voting

After launch:

- follow up with high-intent commenters
- convert interest into installs, GitHub issues, and pilot users
- publish one short retrospective on what people asked and what you fixed

## Product Hunt Rules And References

Review these before posting:

- [How to post a product](https://help.producthunt.com/en/articles/479557-how-to-post-a-product)
- [Product Hunt Featuring Guidelines](https://help.producthunt.com/en/articles/9883485-product-hunt-featuring-guidelines)
- [How do I share my post?](https://help.producthunt.com/en/articles/2690626-how-do-i-share-my-post)
- [Can I ask my community/friends/family to upvote a product?](https://help.producthunt.com/en/articles/484935-can-i-ask-my-community-friends-family-to-upvote-a-product)
- [Why are posts removed?](https://help.producthunt.com/en/articles/484940-why-are-posts-removed)

Key constraint: do not ask people to upvote. Ask them to check it out, try it, and join the discussion.
