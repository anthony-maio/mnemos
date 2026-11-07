# Storage delivery preparation

The improvement branch now includes public `master` at
`866444378e5a42598bba0548f6460d07adcded10`, rechecked through GitHub before
integration. Local merge `7fcca905b780b8d9326423a1e71ede16a8f5dbba` preserves
the tested transaction implementation. Its complete file tree exactly matches
its local parent `86c5d676`; only ancestry changed. The merge's sole conflict
was `mnemos/utils/storage.py`. All 129 source files bound by the running temporal
experiment retained their hashes.

A separate backport is ready at
`d8a5e64789b9d71bba2d7edb8626dd224979c868` on `wt/mnemos-storage-backport`,
directly based on public `86644437`. It changes one storage method, adds five
regressions and updates the changelog. It can be reviewed independently of the
development candidate and its experimental features. No branch was pushed and
no pull request or release was published.

The backport keeps graph-neighbor replacement inside a SQLite savepoint. The
write lock is acquired before checking source and target existence. Failed
insertion restores the previous edges, and both success and failure preserve
a caller's surrounding transaction. The public branch has no development-only
graph-preference batch path, so this backport does not introduce one.

## Verification

The unchanged public implementation passed eight existing storage checks. The
five new regressions then produced four failures and one pass: PR #5 already
handled a source deleted by another process, but partial-write rollback and
caller-transaction preservation still failed. After the backport, all thirteen
storage checks passed, followed by 62 storage/engine checks and the full suite:
316 passed in 19.14 seconds. Tests used disposable databases and external
temporary directories.

Black passed for all 65 Python files. Full mypy reports four diagnostics in
unchanged settings and embedding code. Restoring the original public storage
method temporarily and rerunning the command produced identical diagnostics;
the backport source was then restored byte-for-byte. The changed storage module
passes strict mypy with imported-module diagnostics suppressed. The full type
check remains failing and is identified that way in the prepared PR body.

Review also caught an encoding error introduced while copying the method.
The final file preserves every line outside that method from the public Git
blob. The full passing suite and formatting check ran after this correction.

## Review package and limits

The local review package contains the commit's patch, a PR body following the
repository template, its execution-flow diagram, test output, the unchanged
baseline type diagnostics and the integration receipt. The patch SHA-256 is
`10e5c7e64077c120ee08e827954ba8e87553d8ad48bce17847df02cc2c9e775c`.
The [receipt](storage-delivery-checkpoint.json) binds every private archive
member and the exact source commits.

Package metadata on the backport remains `0.6.0`; no version or release artifact
is claimed. The backport has not been installed in a native host. The separate
development candidate's installed/native evidence does not cover this branch.
Public delivery still requires review and the applicable publication steps.
