# Exact prerelease across six local environments

The same 0.6.1rc1 wheel passes 39 installed storage/recency cases and three
MCP roundtrips in each of six environments. This increment adds Debian 13.1
under WSL2 on Python 3.10.18, 3.11.13 and 3.12.11 to the previously recorded
Windows runs on those Python versions. No wheel or test case changed.

| Environment | Python versions | Passed per version |
| --- | --- | ---: |
| Windows | 3.10.18, 3.11.13, 3.12.11 | 39 storage/recency + 3 MCP |
| Debian 13.1, WSL2 | 3.10.18, 3.11.13, 3.12.11 | 39 storage/recency + 3 MCP |

These are 252 executions of 42 distinct cases. The new Linux portion contributes
126 executions. Independent XML checks verify identical case IDs with no failures,
errors or skips; each MCP run deselects one static configuration case. Coverage
includes schema migration, concurrent opening of an existing WAL database,
another process deleting a graph source, caller-owned transaction commit and
rollback, FTS preservation, recency invariance under access and stdio roundtrips.

Linux cells use separate external targets and the unchanged wheel with isolated
pip, no index and no dependency installation. Existing isolated dependencies are
reused and recorded. Their 35 prior package files remain unchanged. Installed
module origins and runtime/package versions are checked. The copied wrapper and
sample configurations match the candidate source. Each MCP run uses a disposable
home, mock LLM, simple embeddings and fresh databases. It is not an interactive
Claude or Codex session.

The Linux helpers use POSIX paths and pip rather than Windows uv. Git identity
is verified from Windows because the source worktree's Git metadata uses Windows
paths; direct package-file comparisons still bind the Linux inputs to the exact
wheel. Post-run checks verify all 35 package files in every Linux target. A scan
of process working directories finds none remaining under the three test roots.
The source checkout is clean, and every launched check exited zero.

The [receipt](legacy-rc1-six-environment-v1-receipt.json) binds the six records,
wheel, versions and private archive. The archive contains original Windows
records, new Linux logs, JUnit output, dependency metadata, module origins,
helpers and the post-run audit. Disposable databases remain in their respective
test directories; existing archives and frozen quality scores are unchanged.

This completes this local installed-check matrix. WSL2 is not hosted Linux CI,
and reused dependencies do not establish fresh resolver compatibility. The full
source suite was not repeated in each cell. Original benchmark, semantic quality,
complete-command latency, interactive native hosts, actual pilot and publication
requirements remain open. The schema-3 incompatibility with older clients still
requires the documented upgrade process; no live migration occurred.
