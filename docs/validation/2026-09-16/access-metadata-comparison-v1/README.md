# Access-metadata write comparison

The predeclared order was baseline, candidate, candidate, baseline. Each arm loaded 10,000 facts with deterministic 768-dimensional vectors into a fresh database, then timed 100 durable five-ID access updates under WAL/FULL. Baseline is frozen d94ded6; candidate is frozen 2494e31. All five counters reached 100 after reopen in every arm, and both candidate databases retained 10,000 access-metadata rows.

| Arm | Median ms | p95 ms | Maximum ms |
| --- | ---: | ---: | ---: |
| 1: baseline | 329.776 | 1985.947 | 11126.843 |
| 2: candidate | 74.483 | 137.821 | 4696.202 |
| 3: candidate | 62.556 | 82.437 | 1095.222 |
| 4: baseline | 56.096 | 82.747 | 202.397 |

The large difference between the two baseline arms prevents a reliable speedup claim from this run. The host was not isolated; the real-model queue and other workloads could run concurrently. These measurements support persistence checks and expose timing variability. They do not close the 500 ms end-to-end recall gate, measure semantic quality, or attribute stalls to a specific filesystem activity. WAL file capacity does not count overwritten frames. No durability setting was weakened.

Each worker recorded its actual frozen import, source hashes, helper hashes, command, and settings before measurement, and checked source and helpers afterward. The fixture helper's original standalone main was not used; the comparison worker called its fixture function under the selected source import. Database files remain in ignored diagnostic storage; reports and executed helpers are preserved here. The preserved `status.json` is the last in-progress snapshot; `report.json` records final completion.
