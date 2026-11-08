# 0.6.1rc1 release draft

Status: local, unpublished, awaiting artifact validation and release review.

## Proposed change description

This legacy repair candidate makes graph-neighbor replacement atomic, preserving
caller-owned transactions after success or failure. It separates known content
revision time from access time, so reading stale content does not make it appear
newly written. The Claude plugin wrapper resolves an explicit Python command
against PATH before launch on Windows.

PR #5's missing-source guard is retained. PR #7's MCP 1.x constraint and
content-only full-text index updates are retained. The candidate also completes
float32 array annotations without changing embedding calculations.

## Evidence available

Combined implementation b3508f6d passed 353 source tests on Windows Python
3.11.13, plus strict typing and formatting in the recorded Python 3.12.11
environment. The 0.6.1rc1 metadata and documentation checks are separate.
Older installed-artifact checks belong to their exact earlier commits.

The controlled content-recency comparison is development evidence on known
questions. It does not establish general answer support or abstention. The
original failed benchmark remains a failed result. No new quality, latency,
interactive-host or daily-use claim is approved by these checks.

## Upgrade impact

Schema 3 adds a nullable content revision timestamp. It preserves unknown
revision history instead of copying activity timestamps. Older clients cannot
share the upgraded database. Follow [legacy-upgrade.md](legacy-upgrade.md):
stop old clients, retain an original consistent backup, and rehearse on a copy.
A package downgrade alone does not reverse the migration. Restoring an old
backup after new writes or forget requests can lose or resurrect information.

## Publication state

No tag, release, package upload, marketplace update or host configuration change
has occurred. Artifact validation and the [review checklist](release-checklist.md)
remain required. This candidate is separate from canonical 0.7 development and
its inactive pilot. Any final announcement must name the exact published version
and link to its actual verification evidence.
