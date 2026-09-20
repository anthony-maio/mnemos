# Source association measurement v1

Status: 45 observations completed; coverage and controls FAILED. This is a
five-case development diagnostic, not a quality acceptance result. The run
remains failed after diagnosis; a repaired helper requires a separate run.

Source: `f163db1aa155b3cfa5e0d6bd2e8abedc52b2240d`. Execution ran from
2026-09-20 10:00:44.604571 UTC to 10:01:55.272503 UTC. The process exited 1
because acceptance controls failed. All committed measurement inputs matched
HEAD before and after execution. Repository status is recorded separately;
preexisting untracked research evidence was retained.

The [protocol](../../../plans/2026-09-20-graph-association-development-v1.md)
and [frozen vectors](../source-association-vectors-v1/README.md) bind five
cases, three arms, and three deterministic repetitions. Repetitions are not
independent examples. There were no provider calls during measurement.
Extraction and correction inputs were authored fixture behavior. Production
retrieval, scoped SQLite, and graph traversal ran with spreading enabled and
the other four cognition modules disabled.

## Observations and failed controls

Each arm recorded 15 rows: six rows had a direct-candidate target, zero had a
structural path from the actual selected seed to the target, and zero returned
the target in the top five. These are raw observations from a failed-control
experiment, not evidence that the proposed association channel is accepted.

The museum case's semantic and combined arms failed the activation oracle in
all three repetitions. The same oracle failure propagated into aggregate
restart, insertion-order, and edge-cut checks, and the semantic parity gate.
The untouched and semantic-view returned IDs and candidate scores actually
matched; the parity gate also required the failing untouched oracle check.

Diagnosis found a benchmark defect: the supplementary activation call omitted
`max_visited`, taking the standalone default of `1 + 3 * 4 = 13`. Production
retrieval and the independent oracle used the declared limit of 1,000. The
omitted museum node had a valid three-hop path with energy
`0.31294273107107173`. This does not establish a production traversal defect.
A subsequent synthetic regression reproduces the helper's 13-versus-15 node
mismatch before repair. Cases, vectors, graph thresholds, and production
behavior are unchanged by that diagnosis.

The lifecycle case also failed all nine arm/repetition controls requiring a
retained shared source. `forget_fact` deliberately erases the shared event and
its evidence and records source suppression. The remaining fact rows survive;
the corrected fact stays at version 2 and the forgotten fact stays absent
after replay and restart. The proposed source-preservation requirement is not
satisfied. Do not change deletion behavior to preserve content the user asked
to forget, or reinterpret this failed protocol as passing.

## Evidence

`raw-evidence.zip` contains all 196 original output files, including
`measurement.json`, per-arm/control reports, and SQLite snapshots.
`index.json` binds the exact ZIP members and the packaged evidence files.
The compact `summary.json` retains the bindings, counts, and every failed
control. The original temporary output and additional working copies remain.

- Measurement SHA-256:
  `b12d46dff66533dfc0131f10dd1fe95a882a38c7bc9a431c22b4c787590f352f`.
- ZIP SHA-256:
  `ceb09229ae631d86ea4ee47f843f95e8e81bd6300fe29d05feb5572e8c78f77e`.
- Independent verification checked all artifact hashes, the complete
  case/arm/repetition grid, scope membership, score arithmetic, typed edge
  provenance, unchanged direct pools in edge-cut controls, and terminal
  correction/forget/replay state.
- Included JUnit reports cover 25 passing offline helper tests: 11 fixture,
  eight view, three measurement, and three lifecycle cases. Black and strict
  Python 3.10 mypy passed for the nine new helper/test files before execution.

The original held-out graph failure and all T18 quality gates remain open.
