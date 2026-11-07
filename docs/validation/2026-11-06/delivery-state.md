# Delivery state refresh

Checked November 6, 2026, at 20:02 Eastern / November 7 at 01:02 UTC.
This is a read-only inventory. It does not record a release or deployment.

## Published and local states

| Surface | Verified state | Meaning |
| --- | --- | --- |
| Public source | `master` at `866444378e5a42598bba0548f6460d07adcded10` | PR #5 is merged; the public package metadata still says `0.6.0` |
| PyPI | `mnemos-memory` `0.6.0`, uploaded April 1, 2026 | The published files predate PR #5 and the local service candidate |
| Public development branch | `fix/mnemos-delivery-20261026` at `9770c009f8d3b4919ecfd18810d700badc53c638` | Unpublished `0.7.0b1` development source |
| Website example | Pinned to `7d5fc27757ce0a558b42677631ce943238ef789d` | Earlier installed CLI/MCP persistence check, with its original limits |
| Local improvement branch | `356791aec754a20602257426ed32c6b9f06f150f` | Contains later transaction, defaults, support, context and native-validation work; not pushed or published |

The [merged PR](https://github.com/anthony-maio/mnemos/pull/5) adds a missing-source
guard to `replace_graph_neighbors`. The public implementation still commits that
operation directly. The local savepoint, caller-transaction preservation and
rollback work therefore remains a separate delivery obligation. The merge does
not establish that the [PyPI package](https://pypi.org/project/mnemos-memory/)
contains the fix.

A disposable direct-storage probe confirmed the remaining difference. For
present and missing sources, the fetched public method committed the caller's
pending metadata write; a later caller rollback could not remove it. An injected
second-edge insert failure also left partial changes and an open transaction.
The local implementation preserved caller rollback and restored the prior edges
in the corresponding probes. The fetched file's Git blob identity was verified.
This used the public storage module with current type and embedding helpers;
it is not an installed-public-package or native-host validation.

The current [public README](https://github.com/anthony-maio/mnemos/blob/866444378e5a42598bba0548f6460d07adcded10/README.md)
still describes the engine and control plane as ready for real coding-agent use.
The [live website](https://mnemos.making-minds.ai/) already distinguishes PyPI
`0.6.0` from the unreleased candidate and retains the failed release gate.
Its [example](https://mnemos.making-minds.ai/examples/restart-recall/) pins a source
revision, and its [evaluation](https://mnemos.making-minds.ai/evaluation/) retains
the original negative scores and comparison limits. Those pages need an additive
update for new evidence, not replacement with the older website copy.

## Source selection for the next delivery

The live homepage, restart example and evaluation page returned HTTP 200. Their
bytes exactly match the preserved website delivery checkout at `08d393ee`.
The outer workspace has an older website revision with existing local edits.
Those edits were preserved. The three page hashes and the public package file
hashes are recorded in the accompanying receipt.

Continue product work from the improvement branch based on the development
delivery source. Reconcile the newly merged guard with the broader transaction
fix before preparing integration into public `master`. Use the verified website
delivery source for site changes, preserving the old checkout's separate edits.

A subsequent Git merge preflight found one conflict, in `mnemos/utils/storage.py`.
Keeping the tested local implementation resolves it to the exact local tree at
`356791ae`. That implementation already rejects missing sources in both graph
write paths and adds the transaction protections above. The preflight used a
private index; the worktree and real index remained unchanged. Integration has
not yet been committed or published.

Before any delivery approval:

1. Prepare a reviewed integration diff against the current public branch and
   keep the missing-source repair's remaining transaction guarantees covered.
2. Bind the candidate version, source commit, wheel, source distribution and
   installed verification. Rebuild after product changes; a prior wheel does
   not represent later code.
3. Update README, setup instructions and website together. Name the exact
   installable revision for each demonstration and separate the native Codex
   lifecycle result from extraction, retrieval quality and actual pilot use.
4. Retain failed historical gates and original thresholds. A new development
   comparison does not replace fresh evaluation or satisfy the daily-use pilot.
5. Prepare publication and rollback steps for review. Check public branch heads,
   package metadata and hosted pages again at delivery time because this
   inventory is a dated observation.

No host settings, live memory, public branches, package releases or hosted pages
were changed by this refresh.

The [receipt](delivery-state-receipt.json) binds the remote observations, page
hashes, transaction probe and private evidence archive. The updated README and
compatibility summary passed all 12 documentation consistency checks.
