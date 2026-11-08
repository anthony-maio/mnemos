# Installed benchmark dataset repair

The legacy wheel omitted the three files required by the documented `--dataset-pack claim-driving` command. Its resolver looked for a repository-level directory that a wheel install does not provide. The prior release-gate check supplied these unchanged fixtures separately and recorded that limitation.

The isolated `wt/mnemos-legacy-benchmark-pack` candidate includes the three existing fixtures under `mnemos/benchmark_datasets`. Installed resolution uses that directory; source checkouts continue to use `benchmarks/datasets`. No fixture, ranking implementation, metric, or gate threshold changed.

The same three tests ran against isolated extractions of the original and candidate wheels, outside the repository, with module-origin assertions and an existing Python 3.11 dependency environment. The original wheel failed the pack-loading test and passed both error-handling tests. The candidate passed all three. The tests load documents and query labels, verify that labels reference existing documents, and check missing-file and unknown-pack errors. They do not run a retrieval evaluation.

The candidate wheel was built from its source archive with the existing Hatchling 1.33.0 build environment. All three bundled fixtures match repository bytes. The only changed existing package module is `mnemos/benchmark.py`. The three source tests and formatting checks for the two Python files also passed.

Candidate wheel SHA256: `b14cbb8a957779ccadb27bb5b507d3455629a4ee1a61c2990a9a101c9fa13b35`.

This is an unpublished packaging candidate with the same prerelease version as its parent, distinguished by branch, commit, and artifact hash. It has not replaced the previously tested artifact. Its isolated wheel extraction check does not establish a fresh dependency installation or full platform matrix.

The [failed release benchmark](legacy-rc1-benchmark-v1.md) remains unchanged. Its saved report contains aggregate ranking scores and per-query latency observations, but no per-query returned IDs or content timestamps. Those omissions prevent attributing the failed comparison to a specific chronology or tie-breaking event from that report alone. The packaging repair does not resolve the failure or establish release readiness.
