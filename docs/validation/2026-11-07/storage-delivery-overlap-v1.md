# Storage delivery overlap with open upstream changes

A fresh GitHub check confirms that [PR #5](https://github.com/anthony-maio/mnemos/pull/5)
is merged. `git ls-remote` still identifies public `master` as
`866444378e5a42598bba0548f6460d07adcded10`, the prepared transaction backport's
base. The backport at `d8a5e647` remains unchanged and clean.

Two open PRs also change the storage file. Git's merge-tree check, using their
exact current heads and the backport, reports these conflicts without changing
an index, branch or worktree:

| Open change | Checked head | Conflicts with backport |
| --- | --- | --- |
| [PR #6: sleep settings](https://github.com/anthony-maio/mnemos/pull/6) | `b7afffbc625f7fdc9c3b301a38e4c31e88115891` | `CHANGELOG.md`, `mnemos/utils/storage.py` |
| [PR #7: MCP startup and SQLite latency](https://github.com/anthony-maio/mnemos/pull/7) | `9c9ebf8406856057aac051862cb6f7bf55a802d8` | `mnemos/utils/storage.py` |

The inspected PR #7 storage diff narrows full-text updates to changed content,
replaces the legacy trigger, and reformats the existing source-existence guard.
The backport replaces the graph-write method to preserve caller transactions
and roll back partial insertion. These changes address different behavior, but
their shared method region conflicts textually. PR #6 also adds entries at the
same unreleased changelog heading.

Prepare explicit local conflict resolutions against the named PR heads before
choosing a delivery order. Keep each PR's unrelated behavior and tests intact,
retain the original backport and its passing evidence, and validate the combined
storage behavior. The open PR descriptions' CI claims were not independently
verified in this check. Nothing was posted, pushed or changed in either PR.

This check does not reopen the frozen candidate's quality/timing experiments,
change old benchmark results, authorize publication or start a pilot. The
functional candidate already constrains its MCP extras to the 1.x API; that
dependency change is not an outstanding repair in canonical functional code.
