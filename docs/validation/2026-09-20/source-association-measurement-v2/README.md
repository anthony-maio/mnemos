# Source association measurement v2

Status: all 45 observations completed; coverage and lifecycle controls FAILED.
This is a five-case development diagnostic, not an accepted quality result.

Source: `d5b8577389d2a522c0b18ae59d29d6f4a4929a3c`. Execution ran from
2026-09-20 10:09:56.587238 UTC to 10:11:09.433342 UTC and exited 1 because
controls failed. Committed measurement inputs matched HEAD before and after.

The [protocol](../../../plans/2026-09-20-graph-association-development-v1.md),
cases, [frozen Nomic vectors](../source-association-vectors-v1/README.md),
thresholds, scoring, and production behavior are unchanged from
[v1](../source-association-measurement-v1/README.md). The only measurement
repair passes the declared hop and visit limits to supplementary activation;
its [red/green regression](../source-association-visit-budget-v1/README.md)
records the original mismatch and repair. No provider calls occurred here.

## Results

Each of the semantic, source-association, and combined arms has 15 rows:
six direct-candidate target observations, zero structural paths from the
actual selected seed to the target, and zero target appearances in the top
five. Three deterministic repetitions do not make five cases independent
45 times. No association utility is established by these observations.

All non-lifecycle boolean controls passed, including the activation oracle,
scope checks, semantic parity, restart, insertion-order, and edge-cut checks.
The remaining nine failed controls are the lifecycle case across all arms
and repetitions. Its shared-source preservation expectation conflicts with
the existing `forget_fact` behavior: deletion erases and suppresses the source
and its evidence while retaining other fact rows. The corrected fact remains
at version 2; the forgotten fact remains absent after replay and restart.
These are separate observations. Do not reinterpret the failed preservation
requirement as passing or weaken deletion to obtain a passing result.

Extraction and correction used authored fixtures. Retrieval and storage used
production code with spreading enabled and the other four modules disabled.
This does not establish full-pipeline, native-host, or daily-use quality.

## Evidence

`raw-evidence.zip` preserves all 196 output files, including the measurement
report and SQLite snapshots. `index.json` binds each ZIP member and four
packaged files. `summary.json` retains every failed control and run binding.

- Measurement SHA-256:
  `9956a15b7e987569d1bf2a0f4be9ee8e8d5eb68b1d9cdfc70c03f1c830fefffe`.
- ZIP SHA-256:
  `3894615e36731520b343486b83523e761956b9857df3d3ce544a1859486aad1d`.
- Independent verification checked all 195 artifact hashes, the complete
  case/arm/repetition grid, score arithmetic, scope, typed provenance,
  unchanged direct pools during edge cuts, and terminal lifecycle state.

The original held-out graph failure and T18 quality gates remain open.
