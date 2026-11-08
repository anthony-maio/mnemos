# Distinct 0.6.1rc2 legacy package

The local candidate at `a81dc537c82af41feeeb2cd78fd98a9c0f113ef9` has the distinct version 0.6.1rc2 across package, runtime and plugin manifests. It includes the earlier transaction/content-recency/plugin repairs, bundled benchmark datasets, per-query diagnostic evidence and the temporal word-boundary fix. The previous rc1 artifact and failed benchmark remain preserved.

All 401 source tests passed on Windows Python 3.12.11. Strict mypy passed 31 source files using the retained NumPy 2.2.6 environment; Black passed 66 Python files. These checks did not rerun the closed production benchmark.

Isolated Hatchling 1.33.0 built the source archive and then the wheel from that archive. Twine accepted both. All 38 package files match source, source archive, wheel and external installation, including the three named dataset fixtures. Wheel RECORD digests and lengths passed. Seven external plugin/config/document files matched their source archive entries.

The wheel was installed into a fresh external target with retained Python 3.12.11 dependencies. All 87 installed storage, content-recency, dataset-pack, benchmark-evidence and word-boundary tests passed. Three further automated MCP roundtrips passed using the exact generic, Claude Desktop and Claude plugin configuration surfaces. Imported package modules came from the external target; there was no source package in the consumer directory. Installed bytes remained unchanged after MCP checks.

The audit verified all three JUnit records with no failures, errors or skips. This is one Windows Python environment, not a fresh dependency-resolution test, new platform matrix or interactive native-host acceptance. The standalone installed dataset pack was loaded, but no ranking evaluation was run on the closed fixtures.

- Wheel SHA-256: `8c4ba8464eb1ce66c12e0b1ccaf8bcf31129de3cc322e482b1a672f2f818ec2b`
- Source archive SHA-256: `b49d7a42544f86040dcff5f8720b01d76720d97b62c2b3bbcefa53edbab12744`
- Evidence archive SHA-256: `5d698670ab92988c0b9150a5a7bfc9118077f948dc31a64ea527828296985da7`

The [receipt](legacy-rc2-package-v1.json) records exact test and artifact hashes. The local review packet is `.venv/legacy-integrated-rc2-package-v1/REVIEW.md`; it was written after the evidence archive and is not a member of that archive. The candidate worktree is clean and all build/test sessions are terminal.

Nothing was pushed, tagged, published or installed into a live host. Canonical service delivery and the optional explicit pilot remain unchanged. The release benchmark, semantic quality, complete-command latency, fresh acceptance, supported-platform evidence and actual two-week pilot obligations remain open. Successful package checks do not waive those requirements.
