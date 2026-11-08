# Canonical installed package validation

Built wheel and source distribution from canonical delivery commit 5805c2f19121ec4bc018238e7e14a27087981432, including Git discovery stdin isolation. This package uses canonical runtime code; it does not include the experimental SQLite outbox or queued worker changes from the separate capture candidate. Version remains the unpublished development candidate 0.7.0b1.

All 102 selected installed tests passed in 49.49 seconds, with no skips. Coverage includes the original missing-source graph failure using a stale client after independent-process deletion, rollback after partial graph insertion, preservation of caller transactions, graph preference transaction rollback, graph persistence, scoped storage, project identity and stdin isolation, lost acknowledgment, related forgetting and service failures. These are focused regression and lifecycle checks, not the entire repository suite or fresh quality acceptance.

The build used hatchling 1.33.0. All 96 wheel source/resource members matched source and installed bytes; all 103 sdist source members matched source. A fresh Python 3.12.11 environment retained the prior pinned dependency versions. The runner asserted installed imports and changed to the isolated artifact directory before running copied tests, so child Python processes did not inherit a repository working directory. Twelve copied Python files, including test isolation fixtures and the package marker, matched source hashes. Post-test installed parity passed.

Wheel SHA256: 64d8b3fe0c3d685875d6c382003ccdfb9fca69b64e61e6a571e60d6d1268e435.

Sdist SHA256: 34b6a40eafae5ccd240db7ef9fa3c13891be5032979ec42180f4fe428f300218.

The verified 26-member evidence archive contains artifacts, dependency constraints, copied tests and fixtures, JUnit and audit helpers. SHA256: cad124c3bdae8fe015000d0c93e3f27ab0330b1d39c78b6c08b5ee9ad330d778. Exact counts and hashes appear in the adjacent JSON receipt. Artifacts and runtime evidence remain private local files under the improvement worktree's `.venv/canonical-stdin-package-v1` directory.

This closes the scoped canonical installed-validation follow-up for the stdin fix and rechecks the storage repair in that artifact. It does not establish publication, hosted behavior, full native-host acceptance, timing compliance, quality compliance or daily-use pilot completion. Previous artifacts and failed gates remain intact. No live pilot configuration or memory database changed.
