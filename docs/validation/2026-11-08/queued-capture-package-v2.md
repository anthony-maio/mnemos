# Isolated queued-capture package verification

Built the isolated capture candidate `96c3e3c6` with Hatchling 1.33.0 and installed its wheel into a fresh Python 3.12.11 environment. This is a separate unpublished 0.7.0b1 artifact, identified by hash; it does not replace the canonical or pilot installation.

The first successful build omitted two declared source-distribution files because the sparse checkout excluded them: `examples/restart_recall.py` and `scripts/smoke-installed.py`. That artifact remains preserved as package-v1. The files were restored from the same commit, and package-v2 was built into a new directory. No runtime source changed to repair the checkout.

The second wheel's RECORD hashes and sizes passed verification. All 97 packaged source/resource files matched the source checkout and their installed bytes. The source archive's 104 checked members matched source, including both previously omitted files. Wheel SHA-256 is `f9160ee1ca1bc09f169c09f7ad3c61ca5e45019c66150b29d14903fbebb3f5c3`; source archive SHA-256 is `964184d1d063a770ff45f46aaefefb3458e38b49da695c086ab05305144fa610`.

All 42 installed cases passed across worker behavior, queued integration, queue receipts, replay lifecycle, real-service correction/forget suppression, and supplied outbox selection. Tests were copied unchanged into a separate directory and bound by hashes. The runner used isolated Python mode and asserted that Mnemos imported from the new environment. Dependency versions and JUnit output were retained under `.venv/queued-capture-package-v2`.

These are installed-package tests using deterministic embeddings and local test fixtures. They do not establish real-model quality, operating-system worker restart recovery, native-host behavior, or complete acknowledgment latency. No pilot configuration, live data, publication, or production default changed. Next exercise the installed worker as a separately owned process across pending capture and restart before measuring the 100 ms boundary.
