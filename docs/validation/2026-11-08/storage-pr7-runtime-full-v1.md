# Transaction fix and plugin runtime integration

A separate local PR #7 candidate at `1fb477fa2a55a1b4d0fedfe587d82dcc60548290`
passes its complete source suite: 325 passed, no failures, errors or skips on
Windows Python 3.11.13. It combines the existing transaction candidate
`e02c3951` with the unchanged plugin runtime repair from `a229d8fd`.

The fresh GitHub metadata check found PR #5 merged at `86644437`, PR #6 open
at `b7afffbc`, and PR #7 open at `9c9ebf84`. These match the bases used for the
prepared transaction candidates. The refresh does not establish current CI.

## Why the additional repair is needed

Full-suite checks on the original candidates expose a Windows plugin launch
failure that the earlier selected storage checks did not exercise. After test
environment repairs, PR #6 passes 343/344 and PR #7 passes 319/320. Both fail
only the Claude Code plugin manifest roundtrip: the wrapper's bare `python`
command launches an interpreter without Pydantic. The prior wrapper repair
resolves that explicit command against PATH before starting the child.

The new candidate carries the exact three-file wrapper commit, including its
five regression cases and documentation. It preserves the transaction repair,
PR #7's FTS update changes and all existing tests. The full suite now passes,
including generic stdio, Claude Desktop configuration and Claude plugin
manifest roundtrips. These are automated MCP configuration checks, not actual
interactive native-host acceptance.

## Retained setup failures and evidence

The first full run had six failures per candidate: sparse checkout omitted the
hero image and skills; subprocess PATH selected Python without dependencies.
Tracked documentation assets were materialized, and the isolated environment
was placed first on PATH. The second run had two failures per candidate because
sample Claude configs required a home directory. A distinct disposable USERPROFILE
for each candidate cleared that setup issue. The third run exposed the one
remaining product failure described above. All three runs remain in the archive;
no test, threshold or product file was changed to repair the harness.

The successful combined run uses the same isolated environment. Source origins,
all test-file hashes, dependency versions and 35 installed package files are
checked before and after. The installed legacy package is not the candidate:
this result is explicitly a source-suite result. An independent XML audit
verifies all 325 cases and both original candidates' remaining failure.

The [receipt](storage-pr7-runtime-full-v1-receipt.json) binds the exact patch and
private evidence archive, including helpers, raw logs, JUnit records, dependency
metadata and upstream snapshots. Test controllers are terminal. Candidate and
original worktrees are clean. Canonical delivery, frozen pilot artifacts and
normal host settings are unchanged.

No package was built for this combined candidate, and no new typing, benchmark,
Python-version matrix or hosted CI result is claimed. The known unsuppressed
legacy typing failures remain recorded in the earlier overlap report. No branch
was pushed and no PR was edited or merged. The original semantic, latency and
actual pilot requirements remain open.
