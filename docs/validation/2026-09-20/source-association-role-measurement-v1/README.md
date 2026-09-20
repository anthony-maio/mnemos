# Role-aware source association measurement v1

Status: all 90 observations completed; coverage and lifecycle controls FAILED.
This is a five-case development diagnostic. It does not establish quality
acceptance or independent evidence from 90 different cases.

Source: `366039681ba95734882f43bd870adf1c7e40d961`. Committed inputs matched
HEAD before and after execution. The [prospective protocol](../../../plans/2026-09-20-source-association-role-protocol-v1.md)
keeps the original cases, controls, scoring, limits, and three deterministic
repetitions. Both embedding contracts retain separate query and document maps.
The [vector preparation](../source-association-role-vectors-v1/README.md)
adds only the authored correction's document vector to each arm of the
[paired prefix experiment](../embedding-prefix-development-v1/README.md).
No provider calls occurred during this measurement.

## Results

Each embedding contract has 15 rows per relation arm: semantic,
source-association, and combined. Unprefixed vectors produced zero target
paths. Task-prefixed vectors produced a path in one case, `scope_isolation_pair`,
in all three relation arms and all three repetitions. Every arm and embedding
contract still produced zero target top-five appearances.

The prefixed scope-isolation case selects the authored pump seed. Its valve
target is outside the direct candidate pool. Semantic edges reach it with
energy 0.576922 and final rank 31. Source associations raise its energy to
0.8 and rank to 20; the combined arm gives the same result. This is a measured
rank change, but it does not improve top-five retrieval or add a newly
reachable case beyond semantic edges. Do not report it as no behavioral
change, or as demonstrated practical retrieval utility.

The prefixed shared-capture target also moves from rank 32 to 28 with source
associations despite having no target path. Changes to other candidates can
change its relative position; this is not evidence of target activation.

All non-lifecycle boolean controls passed, including query/document role
separation, scope, restart, insertion order, edge cuts, score parity, and
activation checks. The 18 lifecycle failures cover the same authored case
across both embeddings, all relation arms, and all repetitions. Its source
preservation expectation conflicts with `forget_fact`, which erases and
suppresses the shared source and evidence while retaining other fact rows.
The corrected fact remains at version 2 and the forgotten fact remains absent
after replay and restart. Preserve these distinct observations and the failed
control; deletion must not be weakened to make the experiment pass.

Extraction and correction used authored fixtures. Retrieval and storage used
production code with spreading enabled and the other four modules disabled;
source associations were an experimental read-only overlay. Production
embedding defaults, held-out results, and acceptance gates are unchanged.

## Verification and evidence

- The independent verifier checked all 390 artifact hashes, the complete
  90-row grid, role-specific cosine calculations, score arithmetic and ranks,
  direct pools, scope, relation provenance, and terminal lifecycle state.
- `raw-evidence.zip` preserves all 391 output files, including SQLite
  snapshots and `measurement.json`. `index.json` binds every archive member
  and ten packaged evidence/helper files.
- The focused offline suite passed 31 tests. Black and strict Python 3.10
  mypy passed for seven changed benchmark/test files. These results do not
  replace the product-wide checkpoint at `db21750`.
- The runner's red report records missing-module collection failure before
  implementation; its green report records five passing tests. This was
  test-first development, not a reproduced production defect.

Measurement SHA-256:
`7dae4c375c65c3b8f82b85375f60fb87d30d641a91dc348c30dcd0f9a31d0f3f`.

ZIP SHA-256:
`ef9f600506d99c474634c5a5e35ed597ee69959e179c713716ac9198a273ff10`.

The next useful diagnostic is to inspect the actual chosen seeds and score
margins using these saved traces. A rank improvement in one authored case
does not justify changing scoring weights or rerunning held-out acceptance.
Original negative runs remain available, and T18 graph quality remains open.
