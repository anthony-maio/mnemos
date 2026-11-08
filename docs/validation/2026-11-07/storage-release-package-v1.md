# Storage patch release preparation

Candidate `9ea3852a23fd315f3bb641bd6450c7889878e951` prepares version `0.6.1`
from public base `866444378e5a42598bba0548f6460d07adcded10`. Packaging passes,
but the legacy production benchmark fails. This candidate is not release-ready.

The original three-file storage backport at `d8a5e647` is preserved. A separate
branch adds the smallest fixes for its four recorded typing errors: remove a
redundant provider cast and complete three NumPy array annotations. Version
metadata is aligned across the package, plugins and documentation. The result
changes eleven files relative to the public base. It does not change the frozen
`f5a944ea` product or its inactive pilot installation.

| Check on the exact candidate | Result |
| --- | --- |
| Source tests | 316 passed |
| Strict typing and formatting | 31 source files and 65 Python files pass |
| Wheel built from source archive; Twine | Pass |
| Source, archive, wheel and installed package | All 35 files match |
| Installed storage regressions outside the checkout | Five passed |
| Installed CLI help and dependency check | Pass |
| Legacy synthetic production gate | FAIL: 10/12 comparisons pass |
| GitHub CI / Python version matrix | Unrun |
| Native Tier 1 flows / actual pilot | Unrun for this candidate |

Checks use CPython 3.12.11, NumPy 2.2.6 and mypy 2.4.1. The frozen dependencies
and raw logs are archived. Passing this local environment does not establish
the result of CI's unconstrained dependency installation.

The legacy benchmark uses the existing three claim-driving datasets, memory
and SQLite stores, baseline and engine retrieval, top-k one and two repetitions.
Both in-memory `contradiction-update-v1` comparisons fail: MRR lift is 0% and
-50%, below the unchanged 15% requirement. All six SQLite comparisons and all
latency ratios pass. No threshold, dataset or scoring change was made. The
benchmark source matches the public base, but this check does not establish
the cause of the failure or prove the base has the same behavior. No matched
base comparison or further benchmark retry occurred in this increment.

Two setup failures are retained before the completed run. The first inherited
the machine's provider configuration and received an OpenRouter 401 before
producing a report; its inputs were public synthetic fixtures. The next attempt
selected local providers but blocked Windows asyncio's internal loopback
socket. The completed run used explicit isolated settings, simple local
embeddings and a mock LLM. Its guard blocked HTTP and non-loopback connections
while permitting numeric loopback for asyncio. These setup repairs did not
change the benchmark.

A read-only audit verifies the package files, test records, gate failures,
original backport and frozen main product. Fourteen watched live paths and
all 96 inactive-pilot package files remain unchanged. The private review packet
contains the exact patch, PR body, release-note draft, source bindings and all
failed attempts. The [receipt](storage-release-package-v1-receipt.json) binds
the archive and artifact hashes. No push, PR, tag, publication or live
configuration change occurred.
