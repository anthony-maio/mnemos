# Development graph degree diagnostic

All 15 planned combinations completed with unchanged correctness controls on
verified source `5e4b057`. This offline experiment uses the five existing
development cases and their saved Nomic vectors. There were no model calls,
no new held-out observations, and no production configuration changes.

| Neighbors per fact | Development cases | Structurally eligible | Target returned in top five |
| --- | --- | --- | --- |
| 4, current default | 5 | 0 | 0 |
| 8 | 5 | 0 | 0 |
| 16 | 5 | 1 | 0 |

At sixteen neighbors, case 001 gains a valid path from the query-selected seed
to the target, with activation energy 0.45934. That exceeds the unchanged 0.3
activation threshold, but the target still misses the top five. Its semantic
similarity is 0.71160. Baseline and graph return the same five IDs in that
case. The other four cases remain structurally ineligible from their actual
selected seeds. Connectivity and ranking are therefore separate limitations;
widening this budget alone does not deliver the intended associative recall.

The persisted graph grows from 62-64 edges at degree four to 332-338 at degree
sixteen. Those are fixture graph sizes, not storage or latency guarantees.
Independent-path activation, edge-cut, foreign-scope exclusion, and reversed
insertion controls pass for every combination. Passing those checks establishes
the tested mechanics, not useful answers.

These five cases are renamed variants of one development template. No setting
is selected as a new default, and the original held-out zero-coverage result
remains unchanged. Further graph work needs a justified association/seed
contract and retrieval scoring evidence across disjoint development examples.
Increasing graph weight or rewriting labels to make this fixture pass would
not settle that question.

The [prospective protocol](../../../plans/2026-09-19-graph-degree-development-protocol-v1.md)
was committed before execution. The earlier [v1 parser failure](../graph-degree-development-v1/README.md)
is preserved. This successor runs all combinations afresh, correcting only
the outer parser. `report.json` records the effective configurations, source,
input hashes, and outcome rows; individual JSON files retain all measurement
controls and persisted edges. The read-only source snapshot remained clean.
`inputs/development-inputs.zip` retains the exact authored and frozen-vector
inputs, with a separate member manifest and hash index. SQLite measurement
files remain at their recorded temporary paths; they are not included here.
