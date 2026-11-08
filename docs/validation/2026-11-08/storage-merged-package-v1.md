# Private 0.6.1 build from merged storage fixes

Candidate `37e942f8043d0f60eb007587a1e1d98d39ae78ec` on [wt/storage-release-merged-20261108](https://github.com/anthony-maio/mnemos/tree/wt/storage-release-merged-20261108) prepares patch metadata and release notes from merged master `588a190b`. The branch is pushed; no tag, GitHub release or PyPI upload exists from this increment. A live [PyPI metadata check](https://pypi.org/pypi/mnemos-memory/json) returned 0.6.0 as the published version, with releases 0.1.0, 0.2.0 and 0.6.0.

Version 0.6.1 agrees across pyproject.toml, package __version__, plugin/marketplace manifests and the README citation. The changelog and release notes explicitly identify an unpublished candidate. The release scope covers merged graph transaction, MCP compatibility, FTS metadata-update and plugin interpreter fixes; it does not include the separate functional-memory candidate.

| Check | Result |
| --- | --- |
| Documentation, plugin metadata and graph source tests | 32 passed on Windows Python 3.12.11 |
| Isolated build with Hatchling 1.33.0 | Source archive and wheel built; wheel built from source archive |
| Twine | Both artifacts pass |
| Wheel RECORD hashes/sizes | Pass |
| Source/archive/wheel/installed package parity | 35 files match |
| External installed storage tests | 11 passed on Windows Python 3.11.13 |
| External installed Tier 1 MCP configuration roundtrips | 3 passed; one non-Tier-1 test deselected |
| External wrapper, configurations and MCP test | Match candidate and source archive |

The installed checks run outside the checkout using a new --target directory and the existing isolated legacy dependencies. Module origins point to that new target; no source mnemos directory exists in the consumer. The plugin tests use a disposable home, mock LLM and simple embeddings. All 35 pre-existing installed package files remain unchanged. This is not fresh dependency resolution, a new Python matrix, interactive host acceptance or actual pilot use.

Artifact SHA256:

- Wheel: `2568133cfa5e2d68d54bedc94f3f0764bfc7a6d2c3892ae112da4c0af628f79c`.
- Source archive: `5de4f42ed552e9b7047036896730bbbf02356c156550ff5e16139ac49987d00f`.
- Private evidence archive `.venv/storage-merged-package-evidence-v1.zip`: `871d6f42b20f37462fb324e648093528c6c44e8f30fc6e6b21301f4868f1b28a`.

The 11-member evidence archive contains both artifacts, installed logs/JUnit/results, exact helpers and hash manifest. Every manifest member passed readback. Build, Twine and the 32 source-test results are recorded from terminal output rather than separate archived logs. The installed-result record retains dependency versions and module origins.

The hosted 325-test matrix and legacy benchmark result remain bound to the previously merged implementation, not silently reassigned to this versioned package. The repository release checklist still includes operational readiness, real host flows and two weeks of daily pilot usage. Those requirements are not proved by the passing CI job named release_readiness. Publication uses the guarded mnemos-release environment and requires a separate approval; the approved PR merges did not authorize a package release. No live database, host profile or optional pilot installation changed.
