# Installed package check for the storage backport

The minimal backport passes packaging and isolated installation at unchanged
source `d8a5e647`. Its wheel was built from the source archive. Twine checks
pass, all 35 package files match the clean source and installed wheel, and the
five graph regressions pass outside the checkout. These include a stale client
after deletion by another process, partial-write rollback and preservation of
caller-owned transactions. CLI help and dependency consistency checks pass.

The existing full source suite remains 316 passing tests on this commit; it was
not rerun for packaging. Its four baseline type errors remain unchanged. The
new installed checks do not establish native-host acceptance.

GitHub master was rechecked on November 7 at `86644437`, with
[PR #5](https://github.com/anthony-maio/mnemos/pull/5) merged. The public
[GitHub release](https://github.com/anthony-maio/mnemos/releases/tag/v0.6.0) and
[PyPI package](https://pypi.org/project/mnemos-memory/) remain version 0.6.0.
The private build deliberately retains that metadata to validate the unchanged
three-file patch. It is not an uploadable new release. A release still needs
a new version and the remaining release checks.

The local review bundle now includes the original patch, refreshed PR body,
flow diagram, original source/type/formatting logs, wheel, source archive,
installed test output and exact dependency versions. The 21 members of the
original backport archive were verified without modification. No branch was
pushed, PR created, package published or live configuration changed.

- Wheel SHA-256: `c3150502bd64ee24e1dade77f990c03ba3330d968b2220576527e2dddb6cd216`.
- Source archive SHA-256: `3149098bc4cb85a7d51b7dfcf11f9805fca25596a734eb437754e47afb573e09`.
- Review archive SHA-256: `825e6ffa097a2a5fff1404159bf26454ccb324d16e85a06abd615fba2cf52e79` (25 verified members).

The [receipt](storage-backport-package-v1-receipt.json) binds these checks.
The [original backport report](../2026-11-06/storage-delivery-checkpoint.md)
retains the source regression results and their limits.
