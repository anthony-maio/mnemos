# Transaction backport compatibility with open upstream PRs

Prepare separate local candidates on PR #6 at
`b7afffbc625f7fdc9c3b301a38e4c31e88115891` and PR #7 at
`9c9ebf8406856057aac051862cb6f7bf55a802d8`. Both heads were rechecked before
this increment. Preserve the original backport `d8a5e647` and all existing
canonical, candidate, package, benchmark and pilot artifacts.

Resolve the storage conflict by carrying the backport's exact graph-neighbor
method into each base. Preserve every byte outside that method, including the
full-text trigger upgrade. Add the original five graph regressions unchanged
and retain each base's existing tests. Add the transaction entry under the
unreleased changelog heading without replacing the sleep-settings entries.

Add two integration cases for a caller-owned content edit followed by a failed
graph replacement. Verify that graph edges roll back, the pending content and
full-text index remain consistent, another connection cannot see the uncommitted
edit, and the caller can either commit or roll back. Run these cases first on
the unmodified PR #7 storage code. PR #6's storage differs only in a comment;
do not claim a second baseline run.

Freeze each candidate before its acceptance run. In the existing isolated
Windows Python 3.11 environment, run all storage and engine tests plus PR #6's
sleep-settings regressions. Verify imported source origins and the unchanged
installed environment before and after. Allow local IPC, block external model
connections and use temporary databases/configuration. Check typing of the
changed storage module and formatting of the changed Python files.

This is source compatibility coverage for two delivery orders. It does not
repeat hosted CI, qualify a package or native host, rerun a production benchmark,
or change the closed functional quality/timing results. Do not modify either
upstream PR or create a release. Prepare exact patches and review descriptions
with the original and combined evidence identified separately.
