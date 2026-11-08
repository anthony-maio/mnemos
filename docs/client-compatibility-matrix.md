# Client compatibility for 0.6.1rc2

This legacy repair candidate is unpublished. Historical host tiers do not
establish acceptance for a new artifact.

| Surface | Evidence | Remaining work |
| --- | --- | --- |
| Generic MCP stdio config | Automated source roundtrip passed in the combined suite | Exact prerelease artifact validation |
| Claude Desktop sample config | Automated source roundtrip passed | Exact artifact and interactive host checks |
| Claude Code plugin manifest | Automated source roundtrip and Windows Python resolution tests passed | Exact artifact and interactive host checks |
| Codex | Configuration and instruction examples | Interactive acceptance and actual pilot use |
| Cursor, Windsurf, Cline | Configuration examples and static checks | Interactive host checks |
| ClawHub skill | Checked-in onboarding source | Separate publication review |

The combined implementation checkpoint b3508f6d passed 353 source tests on
Windows Python 3.11.13. Earlier installed checks belong to 1fb477fa and cannot
qualify this newer schema-3 artifact. Mock LLM and simple embeddings are test
providers; those checks do not demonstrate production retrieval quality.

Before any database upgrade, follow [legacy-upgrade.md](legacy-upgrade.md).
Older clients cannot share the migrated schema. No host settings or hooks have
been activated by this work. The canonical 0.7 candidate and its pilot remain
separate. The legacy benchmark failure and broader quality, latency and pilot
requirements remain open.
