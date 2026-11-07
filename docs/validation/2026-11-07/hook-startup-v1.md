# Resident command startup diagnostic

Skipping Python's site initialization saves about 14 ms in this installed
client probe. It leaves nearly all measured client import time intact, while
also omitting the ordinary virtual environment initialization. This result does
not justify changing the launcher or its fallback behavior.

The diagnostic used the installed `5735c502` candidate and the same native
validation interpreter as the complete installed comparison. Four arms ran in
a fixed rotating order for two warmup rounds and 20 observed rounds. All 88
commands succeeded; 80 samples were timed observations. The entry file and
94 installed package files were verified before and after measurement.

| Arm | Complete median ms | Complete p95 ms | Import median ms |
| --- | ---: | ---: | ---: |
| Empty, ordinary site initialization | 118.35 | 439.94 | 0.0034 |
| Empty, site initialization skipped | 105.44 | 263.27 | 0.0033 |
| Client imports, ordinary initialization | 201.59 | 225.70 | 114.58 |
| Client imports, initialization skipped | 186.82 | 312.46 | 114.44 |

Skipping site was faster in 17 of 20 matched client rounds, with a median
paired change of -13.72 ms. For empty commands it was faster in 16 of 20 rounds,
with a median paired change of -10.68 ms. The high tail variation and the
candidate's worse client p95 prevent a broader latency claim. These controls
also serialize a receipt inside the complete timer; they are not minimal
interpreter startup measurements or complete-hook acceptance measurements.

Separate instrumented traces explain where to look next. With ordinary site
initialization, importing `http.client` took 58.18 ms cumulatively, including
44.91 ms for `email.parser` and its dependencies. The package initializer took
31.65 ms cumulatively, including 27.97 ms for `typing`. These nested import
times overlap and must not be added together. The traces ran after measurement
and are excluded from the latency summaries.

Both modes produced the same environment fingerprint. The site-free mode used
the base interpreter prefix rather than the virtual environment prefix, which
confirms that runtime initialization would need explicit handling before a
fallback. This experiment did not test that fallback or call a worker.

An independent audit reproduced the observation order, counts, summaries and
paired differences, verified all 98 frozen bindings and confirmed all 88
measured process IDs were absent. It made no model calls. No service, database,
host configuration or product file changed.

The verified private archive contains 111 members. SHA-256:
`d7f0356aeedd9a8d0b4f0685b36fe942596f08625ad6fe4c13732fd36a8fc7f7`.

The next useful investigation is the client import path and interpreter launch
boundary. Any replacement transport or bootstrap must preserve authenticated,
bounded responses and the distinction between safe fallback and an unknown
dispatch outcome. A smaller import trace alone cannot establish that behavior.
Keep the current runtime initialization until a complete candidate proves it.

The prior complete installed result remains authoritative: recall p95 is
134.88 ms and passes 500 ms; Stop and prompt p95 are 547.11 and 860.06 ms and
fail the 100 ms requirement. Current native-host acceptance, fresh quality,
delivery alignment and the inactive daily-use pilot remain open.

- [Prospective protocol](../../plans/2026-11-07-hook-startup-v1.md)
- [Complete installed comparison](scoped-rank-performance-v1.md)
- [Verification receipt](hook-startup-v1-receipt.json)
