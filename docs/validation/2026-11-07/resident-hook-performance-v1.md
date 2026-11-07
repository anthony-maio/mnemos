# Installed resident hook comparison

The resident adapter reduces complete command latency, but it does not meet the
100 ms requirement. Median Stop time fell from 917.09 ms to 327.50 ms; median
UserPromptSubmit time fell from 1280.11 ms to 682.62 ms. Ordinary recall passed
its separate 500 ms p95 requirement at 327.62 ms.

Both arms used the same installed candidate, real BGE-small-en-v1.5 embeddings,
memory service and accepted-fact snapshot. The comparison retained process startup,
configuration checks, durable capture, retrieval and host output inside each hook
timer. It did not change the original thresholds or replace the earlier V3 result.

## Complete-command results

Each row has 50 timed observations. Warmups are retained separately. P95 uses
inclusive linear interpolation. All times are milliseconds.

| Path | Median | P95 | Maximum | P95 limit | Latency result |
| --- | ---: | ---: | ---: | ---: | --- |
| Shared recall | 299.44 | 327.62 | 532.74 | 500 | PASS |
| Ordinary Stop | 917.09 | 1610.48 | 1878.90 | 100 | FAIL |
| Resident Stop | 327.50 | 481.16 | 1106.66 | 100 | FAIL |
| Ordinary UserPromptSubmit | 1280.11 | 1932.96 | 2166.85 | 100 | FAIL |
| Resident UserPromptSubmit | 682.62 | 1276.24 | 1580.97 | 100 | FAIL |

The order alternated within each pair. Resident Stop was faster in 49 of 50 pairs,
with a median paired reduction of 590.32 ms. Resident UserPromptSubmit was faster
in 48 of 50 pairs, with a median paired reduction of 582.68 ms. These are observations
from one Windows machine and one run, not estimates across machines or native hosts.

## Output comparison and durability

All 204 hook invocations, including warmups, returned without command or capture
errors. Their source records and jobs survived service restart. Every resident
command used the worker and avoided importing the ordinary adapter runtime.

The frozen exact-output check failed for all 51 prompt pairs: the resident response
orders JSON object keys differently. A separate audit verified that all 102 pairs
have identical decoded output, and all 51 prompt pairs have identical full context
strings. The original byte-comparison failures remain in the results; both prompt
arms therefore retain 50 failed timed comparisons. No latency or comparison score
was replaced after inspecting the output.

The database ended with 10,204 source events and 204 jobs. One maintenance cycle
completed 100 jobs; 104 remained pending. The 50 additional chunks are unknown
captures, not accepted facts. The original 10,000 accepted facts, their evidence
and document vectors stayed unchanged. Durable capture does not mean every job
has completed.

## Configuration and installed artifact

The ordinary arm uses the unchanged private launcher. The resident wrapper uses
the exact environment setup statements from that launcher, verified by AST
comparison. Worker startup applies the full launcher policy. Each event rechecks
effective settings, environment, working directory and current scope membership.
A safe fallback applies the full original policy before invoking the adapter.

Three preflight checks passed: lightweight successful dispatch, checked fallback
with an absent worker, and refusal of an explicit LLM configuration. No service
or model was started during those checks. No live host configuration was changed.

Candidate source: `feb6ac3b3eebd2e178ec584b3be1e72cca1228fc`.
Protocol commit: `73d2ec8e`. Wheel SHA-256:
`51ba20ba811eaa33cd18bde9665c49d21feaa10e7874bffdf878e3b9762b550f`.
All 94 package files and 57 dependency versions were verified. The fixture began
with 10,000 accepted facts, 10,000 sources, 10,000 evidence links and zero jobs.
Its 79 real document-embedding batches reproduced every vector from the earlier
V3 fixture. Seventeen model-cache assets were verified before and after the run.

The memory service, resident worker and restarted service had separately verified
Windows process ownership. Observed startup times were 3608.90, 1828.58 and
3044.75 ms, respectively, including ownership checks of 1272.71, 1109.91 and
1161.43 ms. All owned processes exited with code zero, and their descriptors were
removed. No forced stop was required.

The independent audit checked all 257 observations, including 250 timed samples,
118 frozen file bindings, vector-index entries, source ownership and restart
durability. It made no model calls and left the database bytes unchanged.
Credential values inherited by private configuration metadata were removed before
archiving; an effective-settings digest is retained. Request data, timings and
frozen runtime inputs were unchanged by that metadata cleanup.

The verified private archive has 146 members. SHA-256:
`e51c585c225668c1028e71c180805603ba063e64874cb5283d8714dadf074e37`.
The original V3 evidence remains intact.

This establishes an installed Codex command comparison with real retrieval and
restart durability. Current native-host acceptance, answer-support quality and
actual daily use remain open. The private pilot remains inactive with zero days.
The next performance step is a separately recorded profile of the remaining
resident-command cost, retaining this result and the original latency limits.

- [Comparison protocol](../../plans/2026-11-07-resident-hook-performance-v1.md)
- [Machine-readable receipt](resident-hook-performance-v1-receipt.json)
- [Worker implementation checks](resident-hook-worker-v1.md)
- [Original installed baseline](installed-performance-v3.md)
