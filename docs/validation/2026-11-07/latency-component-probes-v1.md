# Evidence lookup improves; Git launch remains a development option

An index on `(fact_id, created_at)` reduces median present-fact evidence lookup
from 1.901 to 0.043 ms in the frozen component probe. All 370 result pairs,
including warmups, preserve every returned evidence row and its order. The
same index is now added by transactional schema setup in product source
`f5a944ea9a7f61499733b8b64e829aa10f5920be`. Complete installed latency has not
yet been measured on that product; the original 100 ms hook gate remains open.

The separate Git probe saves roughly 17-20 ms in median paired resolver time
by invoking the installed Git binary directly. Results and identity behavior
match, but outliers remain and the ordinary-repository p95 worsens. Product
Git selection is unchanged.

## Evidence lookup

Both arms use the installed `3d1866d5` SQLiteStore and fresh copies of the
original 10,000-fact seed. The candidate copy receives only the additional
index. Timers include the original lookup method and evidence-model parsing.
Inputs cover 100 evenly spaced existing fact IDs and 20 absent IDs over three
rounds, with alternating arm order and ten warmup pairs.

| Lookup | Timed pairs | Baseline median / p95 ms | Indexed median / p95 ms | Candidate faster |
| --- | ---: | ---: | ---: | ---: |
| Present fact | 300 | 1.901 / 2.459 | 0.043 / 0.068 | 300/300 |
| Absent fact | 60 | 1.812 / 2.615 | 0.023 / 0.040 | 60/60 |

Index creation takes 7.93 ms on this copied seed. The query plan changes from
a full evidence-table scan plus temporary sorting to an indexed search with
no temporary sort. The probe retains 740 observations, 720 timed, and 370
exact-result pairs. A separate fixture preserves equal-timestamp rows in their
original insertion order, multiple facts, absent results and an insert made
after index creation. Both performance copies retain the exact accepted facts,
evidence, source records and vectors; integrity and foreign-key checks pass.

Three new regression cases fail before implementation because the original
lookup scans and sorts the table. They pass after the index is added. They
check new and existing databases, complete row/order preservation, repeated
opening, visibility inside a caller transaction and rollback. All 120 focused
checks pass across schema startup, storage, capture, repair, service behavior,
source review and forgetting. Strict mypy and Black also pass.

The implementation leaves lookup SQL and parsing intact. Schema setup creates
the optional index for existing databases within its existing transaction. No
live database was opened or migrated. The candidate wheel is built with
Hatchling 1.33.0 and passes Twine. All 85 packaged Python source files match the
committed source; among 96 package files, only `mnemos/utils/storage.py` differs
from the prior wheel. The [artifact receipt](fact-evidence-index-wheel-v1-receipt.json)
records its hash. Isolated installation, full regression and a complete installed
comparison remain required for the new product checkpoint.

## Git executable boundary

The probe calls the original installed `canonical_project_root` for both arms.
Only the executable argument changes: the selected Git for Windows `cmd`
launcher versus its `mingw64/bin` executable. Both installed files report
2.51.0.windows.1. All other arguments, environment filtering, filesystem checks,
timeouts, retries and identity validation remain intact. No result is cached.

Each fixture has one warmup pair and ten timed pairs, with rotating scenario
order and alternating arm order. The table reports complete resolver time.

| Fixture | Selected median / p95 ms | Direct median / p95 ms | Direct faster |
| --- | ---: | ---: | ---: |
| Plain folder | 58.88 / 234.20 | 40.91 / 153.45 | 8/10 |
| Ordinary repository | 56.17 / 58.41 | 38.08 / 148.49 | 8/10 |
| Repository subdirectory | 56.42 / 169.94 | 38.14 / 150.36 | 9/10 |
| Linked worktree | 56.23 / 183.04 | 39.32 / 105.74 | 9/10 |
| Bare repository | 58.14 / 176.02 | 38.41 / 102.62 | 10/10 |

The 110 probe calls include ten warmup observations. Four separate
control calls preserve inherited-selector handling and rejection of a broken
Git marker. Every final identity, error and full subprocess output matches.
All 116 subprocesses launched by these resolver calls exit, including the
original resolver's second attempt for each broken-marker control. Fixture
files and executable bytes remain unchanged. Setup commands are retained
separately and are not timing observations.

This supports a possible explicit executable configuration, not a universal
speedup or a hard-coded installation path. The direct path's higher repository
p95 and its remaining outliers prevent claiming a reliable tail improvement.
No product executable-selection change is included in the index implementation.

## Audit and delivery state

The [prospective protocol](../../plans/2026-11-07-latency-component-probes-v1.md)
was frozen at `f4b7ddca905ca4b2eb62eb668a1cb8935667d1e6`. The independent audit
recomputes observation order, full outputs, all summaries and paired changes.
It checks process identities, source/executable bindings, query plans, the
original seed and both accepted projections without model calls or service
startup. The original three regression failures are retained alongside the
validated implementation and checks.

The private archive contains 52 members and 135,116,438 bytes. SHA-256:
`839f0a3753d18c6bc4099c27143a2e7ab5af6b16ff4e37f931262307e1fc9ce2`.
See the [receipt](latency-component-probes-v1-receipt.json) for exact evidence
bindings and the distinction between the unchanged probe product and the
subsequent source change.

These are development component results, not fresh retrieval-quality evidence,
a complete-hook pass, native-host acceptance or a release. Live configuration
is unchanged, publication has not occurred, and the private pilot remains
inactive with zero actual days.
