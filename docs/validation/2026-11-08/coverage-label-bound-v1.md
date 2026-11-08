# Limits of remapping the recorded coverage labels

Status: offline analysis verified. This rules out a family of repairs to the recorded v10 outputs. It is not a new model evaluation or a product quality score.

A deterministic function of only the baseline relation and coverage label cannot fix the frozen cohort. Exhaustively enumerating all 2,187 mappings of the seven observed label pairs gives an upper bound of 167/198 correct relations, leaving at least 31 errors. Restricting the function to retaining a relation or making it uncertain gives 32 mappings and the same upper bound. None satisfies the unchanged relation gate.

| Baseline relation | Coverage | Expected labels in the recorded cases | Minimum errors |
| --- | --- | --- | ---: |
| Different | Covered | 12 different, 6 uncertain | 6 |
| Different | Incompatible | 14 different | 0 |
| Different | Unspecified | 94 different, 6 uncertain | 6 |
| Uncertain | Covered | 11 uncertain | 0 |
| Uncertain | Unspecified | 6 uncertain | 0 |
| Same | Covered | 24 same, 19 uncertain | 19 |
| Same | Incompatible | 6 same | 0 |

The bound follows directly from these collisions: a single output per row can satisfy at most the most frequent expected label. Enumeration independently matches that sum. The strongest-looking label pair, same plus covered, still combines 24 valid matches with nineteen cases missing necessary identity. Changing a truth table cannot distinguish them.

The independent decision reconstruction first reproduces v8's 167 correct relations, 66 correct operations and six unsafe target choices, and v10's 73, 42 and two. Across all mappings, the largest possible operation count is 66/72. A mapping can also reach 66 with no unsafe corrections, but still fails the relation gate and leaves six operations wrong. These are oracle bounds selected with development labels, not validated alternatives to deploy.

The cohort has 34 distinct literal pairs across twelve seeds and 72 views. One literal pair has two observed label states, yielding 35 distinct pair-and-label states. Removing repeated views still leaves at least six state-level errors: at most 29/35 are correct under a fixed label mapping. This count is descriptive, not an estimate from independent samples.

The analysis verifies the original manifest, fixture, raw artifact and replay-audit hashes before reading the responses. All results use frozen outputs; no model calls, source changes, live configuration changes or pilot days were added. The supplementary archive contains the analysis script, full collision inventory, original evidence receipt and member hashes. The [receipt](coverage-label-bound-v1.json) binds it to the original private archive.

Next implementation work must use additional information, such as literal identity evidence, or change the judgment mechanism. This result does not bound text-aware rules, structured source identities, changed model outputs or future models. Any such design must still preserve unrelated-target distinctions, handle omitted conditions even with one target, separate changed values from identity, and pass fresh evaluation. Do not run another study that merely changes how these same two labels are combined.
