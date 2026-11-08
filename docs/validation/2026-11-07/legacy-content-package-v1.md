# Installed legacy content-recency candidate

The separate content-recency repair is now built and verified as an unpublished
`0.6.2` candidate at `ace62950`. Its runtime implementation matches the
[source-verified repair](legacy-content-recency-v1.md) at `777c5445`, apart from
the version string. Package, plugin and marketplace metadata agree on `0.6.2`.
The README distinguishes this candidate from published installation commands
and labels earlier host-support claims as historical.

The existing `0.6.1` storage wheel and `f5a944ea` functional-memory wheel are
unchanged. This candidate has its own filenames, source archive, environment
and review patch. Nothing was published, tagged or activated.

| Check | Result |
| --- | --- |
| Source metadata, CLI and documentation checks on `ace62950` | 56 passed |
| Installed storage and recency regressions | 41 passed, including 28 content-recency cases |
| Source/archive/wheel/installed package comparison | All 35 files match |
| Installed dependency constraints | 40 matched; bootstrap pip recorded separately |
| Packaging and dependencies | Twine and pip check passed |
| Installed migration | Two copies preserve six chunks and vectors each |
| Migration controls | Graph edge, feedback record, search results and unknown revision history preserved |

The earlier 344-test full suite, strict typing and formatting retain their
`777c5445` binding. They were not rerun or relabeled as a new full-suite result.
The audit verifies that subsequent package-source changes are limited to the
version string; the remaining changes identify and document the candidate.

Hatchling 1.33.0 and build 1.6.1 produce the source archive first, then build the
wheel from that archive. A fresh environment installs the wheel with constrained
dependencies. Installed checks run outside the source checkout with isolated
Python imports. The regression files match the committed tests, and their
session verifies that the package comes from the installation. CLI startup also
passes. These checks use the recorded Windows/Python 3.12 environment; the full
supported platform matrix remains unverified for this candidate.

The installed migration uses copies of the retained six-row legacy database.
One is unchanged before migration; the other adds explicit graph and feedback
controls. The installed code upgrades schema 2 to 3, preserves the original
row projections and leaves all six revision timestamps null. Read-only restart,
integrity, foreign-key and closed-artifact checks pass. This installed check
completed on its first attempt using the corrected source-migration procedure.
The earlier evidence-script mistakes remain in the separate source report.

Schema 3 still requires older positional-row clients to stop before migration.
The original database backup must be retained for any later deployment. These
checks cover existing WAL databases; concurrent first creation and WAL conversion
remain outside the guarantee. Creation-time fallback and tied assertions do
not establish which fact is current.

The [receipt](legacy-content-package-v1-receipt.json) records both artifact
hashes, exact revisions and the private review bundle. The review patch
reconstructs the candidate from the prepared `9ea3852a` storage release, with
an offline PR draft covering the execution flow and migration limits. The
42-member private archive has SHA-256
`73c7ee8c063a11f795b591222f70504aefd6e38558a93cda3ed1054872158363`.

No production benchmark or fresh semantic evaluation was run. The original
failed gate is preserved, and this artifact has no replacement passing score.
A subsequent chronology evaluation must specify source order and prior-access
cases before measurement. Native-host acceptance and release qualification
remain open. This legacy package check does not resolve the frozen functional
candidate's quality or timing failures, and both pilots remain at zero days.
