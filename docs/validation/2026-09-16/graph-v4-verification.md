# Frozen graph v4 verification

Source `b7a9d9d4d9d6947564696457b0e2754f7e95a801` passed the complete
repository checks and an installed-wheel lifecycle check on September 17,
2026. The source was a clean detached checkout throughout these checks.

| Check | Observed result |
| --- | --- |
| Full pytest suite | 624 passed, 2 skipped, 508.75 seconds |
| Black | Passed |
| Strict mypy | Passed using the Python 3.10 compatibility environment |
| JavaScript syntax | Passed |
| Wheel and source distribution build | Passed |
| Twine metadata validation | Passed |
| Installed-wheel lifecycle | Passed outside the source checkout |

The two skipped tests require explicitly enabled WSL host integration. They
are not counted as native host evidence. The [evidence index](graph-v4-verification/evidence-index.json)
records their reasons and hashes every preserved log and report.

The wheel was installed into an isolated environment with dependencies already
present, then tested from `D:\WindowsTemp` with inherited Mnemos settings and
`PYTHONPATH` removed. The script verified imports from `site-packages`, durable
explicit writes, evidence inspection, legacy review, routing review and
dismissal, exhausted-job recovery, restart, stable correction identity/version,
forgetting, graph readiness, packaged UI assets, and host profile export.
The fixture remains at `D:\WindowsTemp\mnemos-final-wheel-kiog2mvk`.

This check used unavailable cognition and simple embeddings. It establishes
packaging and lifecycle behavior; it does not establish extraction precision,
retrieval quality, native host continuity, or human acceptance.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `mnemos_memory-0.7.0-py3-none-any.whl` | 259269 | `a440021bd04479fa97458b6397b163046f61f35a73a7056b0082dc71279e84fe` |
| `mnemos_memory-0.7.0.tar.gz` | 976021 | `adc082103da50709870ff890d6c27392fa7fdb1d6a4a1e9dfdf6904fa5609894` |

The local artifacts are retained under `.venv/graph-v4-checks/dist/`. Commands,
timestamps, interpreter paths, runner hashes, and exit codes are in the
[verification report](graph-v4-verification/verification.json) and
[installed-wheel report](graph-v4-verification/installed-wheel.json).
No package was published.
