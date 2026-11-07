# Complete installed evidence-index comparison

Product `f5a944ea` was compared with `3d1866d5` using the same experimental
native launcher in both arms. The wheels differ only in `storage.py`. Both use
the same 57 dependency versions, 17 real BGE assets and pristine 10,000-fact
seed. The [frozen protocol](../../plans/2026-11-07-fact-evidence-index-performance-v1.md)
retains the original complete timers and acceptance limits.

| Path | Baseline median / p95 ms | Indexed median / p95 ms | Paired median change ms | Indexed faster | Candidate p95 gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Recall | 110.32 / 116.36 | 97.89 / 104.92 | -12.11 | 49/50 | PASS (500 ms) |
| Stop | 134.98 / 324.25 | 134.42 / 321.73 | +2.51 | 22/50 | FAIL (100 ms) |
| Prompt submission | 257.89 / 385.57 | 245.86 / 365.53 | -8.92 | 39/50 | FAIL (100 ms) |

There are 320 retained observations, including 300 timed observations.
All 160 decoded-output and full-context pairs match. The audit verifies all 208
worker dispatches, with no missing native routes. All 208 captured events
survive service restart, with completed jobs. Accepted facts, evidence,
sources and vectors remain unchanged. Both databases retain integrity; only the
candidate has the evidence lookup index. All recorded process identities have
exited, and source, artifact and model bindings remain unchanged.

Stop is a capture-only control and does not perform the indexed evidence lookup.
Timing changes on that path should not be attributed to faster evidence lookup.
The largest measured prompt command is 1,037.64 ms with the index, versus
787.24 ms in the baseline. Lower median and p95 do not remove these outliers.
These measurements include Windows scheduling and command startup variation;
they describe this run on existing synthetic inputs.

The [receipt](fact-evidence-index-performance-v1-receipt.json) retains per-block
results, failures, paired deltas and the independent audit. Archive SHA-256:
`ab8a155d34633c113d00145a7b5c1866f68005b4307877bbc936668b6db27939`.
Prior native comparisons remain intact. The earlier isolated lookup probe
retains its separate measurements and does not substitute
for complete-path acceptance.

This does not establish fresh semantic quality or native-host acceptance. The
pilot remains inactive at zero days. Live configuration, canonical integration
and publication remain unchanged.
