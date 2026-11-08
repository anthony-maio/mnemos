# Transaction backport compatibility candidates

Two separate local candidates resolve the transaction backport's conflicts
with the recorded heads of open PRs #6 and #7. Each carries the exact graph
savepoint method from `d8a5e647`. Every source byte outside that method,
including each PR's full-text trigger upgrade, remains unchanged. Existing
tests and changelog entries are preserved.

| Base | Local candidate | Selected checks passed |
| --- | --- | ---: |
| PR #6 `b7afffbc` | `22a74b9d` | 89 |
| PR #7 `9c9ebf84` | `e02c3951` | 66 |

The checks run on Windows Python 3.11.13 and cover all storage and engine tests;
the PR #6 candidate also runs its sleep-settings tests. Neither run has a failed,
errored or skipped case. These are selected source checks, not full-suite,
installed-candidate, hosted-CI or native-host acceptance.

The five original graph regressions are copied unchanged. Two new cases first
fail on PR #7 because interrupted replacement leaves a changed edge. After the
repair they verify that graph rollback preserves a caller's pending content and
full-text edit, another connection sees only committed content, and the caller
can commit or roll back. The PR heads' original graph methods are identical;
the baseline was run once, on PR #7.

All three changed Python files pass formatting. The unsuppressed strict type
check reports three generic-ndarray errors in unchanged `embeddings.py`.
Checking the changed storage module with imported-module diagnostics suppressed
passes. The broader type check remains failing in this environment.

Both test runs verify source module origins and preserve all 35 package files
and dependency versions in the reused test environment. That environment's
installed legacy package is not either candidate. Each candidate retains
`0.6.0` source metadata and has no new release artifact.

The [receipt](storage-upstream-overlap-v1-receipt.json) binds both four-file
patches, exact reconstructed trees, test results and the private review archive.
The archive contains separate PR drafts, the two baseline failures, both typing
results and source-preservation checks. Fourteen watched live paths and the
canonical checkout remain unchanged. No production benchmark was run and no
upstream branch, PR, package or pilot was changed.

These candidates are alternatives for the eventual delivery order. They do not
merge PRs #6 and #7 with each other, choose that order or replace the original
backport's evidence. The original functional quality, timing and pilot
requirements remain open.
