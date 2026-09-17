# Held-out mechanism review

This note reviews the completed `mechanism-heldout` phase only. The phase is bound
by `real-model-v3/queue-status.json` to frozen source `b7a9d9d4d9d6947564696457b0e2754f7e95a801`.
It is not evidence for the current checkout or for source `9e253390`.

## Artifact binding

All eight phase artifact hashes in `queue-status.json` match the files in
`mechanism-heldout/`: the affective and graph observations, classifier and
embedding raw outputs, failures report, manifest, JSON report, and Markdown
report. The manifest also matches the report, classifier raw-output, and
embedding raw-output hashes. The run is `status=complete`, split `heldout`,
seed `20260916`, and repetitions `3`; the execution record reports 50/50 graph
cases and 50/50 affective pairs completed.

The recorded provider is Ollama `gemma3:4b` for cognition and
`nomic-embed-text:latest` for embeddings. The manifest records 768-dimensional
embeddings, a 131072-token cognition context, and the raw-output hashes. These
are provenance records, not a claim of product quality or human acceptance.

## Graph

The artifact separates graph correctness from retrieval utility:

- `module-mechanism.json.graph_correctness.status` is `PASS`, with
  `correctness_pass=true`, `cycle_coverage="OBSERVED"`,
  `cycle_observations=144`, and empty `failures` and `missing` arrays.
- `module-mechanism.json.graph.summary` reports 50 unique cases, 150
  observations, and 50 complete cases, but `eligible_cases=0` and
  `heldout_eligible_cases=0`. All 150 rows in `graph.observations.jsonl` have
  `status="ineligible"`, `eligible=false`, `valid_path=false`,
  `target_in_direct_pool=false`, and `target_in_graph_results=false`.
- The graph coverage gate therefore fails: `module-mechanism.json.coverage.failed`
  is `true`, with 0 eligible cases against a minimum of 40. The graph summary
  has `positive_cases=0`, `baseline_target_direct_cases=0`, and null
  `wilson_95` and `paired_graph_top_k_difference_95` intervals.

The PASS is evidence that the recorded graph control checks ran and passed. It
cannot be read as graph retrieval success, path recall, or graph utility because
no held-out case had a valid measured path.

## Affective ranking and classifier

The affective section has 50 unique pairs, 100 complete conditions, and 300
observations in `module-mechanism.json.affective.summary`.

- Classifier parsing was measured for all 300 rows:
  `classifier_calls=300`, `classifier_parsed_states=300`, and
  `classifier_fidelity_missing=0`.
- The synthetic-state fidelity rule recorded 154 passes:
  `classifier_fidelity_passes=154`. This is 154/300 under the reported
  classifier-distance rule. It measures classifier agreement with synthetic
  gold states, not retrieval utility.
- Semantic ranking recorded
  `semantic_target_top1_rate=0.0`. Affective ranking recorded
  `affective_target_top1_rate=0.33` and `state_appropriate_top1=33.0`.
  These are ranking observations over the complete conditions, not evidence
  that the classifier itself was faithful on every row.
- The paired top-1 interval is
  `paired_top1_difference_95={estimate:0.33, lower:0.18, upper:0.47,
  confidence:0.95, cluster_count:10, sample_count:100,
  method:"paired_bootstrap"}`.
- The paired top-5 interval is
  `paired_top5_difference_95={estimate:0.05, lower:0.0, upper:0.15,
  confidence:0.95, cluster_count:10, sample_count:100,
  method:"paired_bootstrap"}`.
- `non_discriminating_pairs` contains 21 pair IDs. The report states that
  repeated rows are clustered by fixed case ID and are not independent cases.

Here `sample_count=100` counts paired conditions. It is not the number of
bootstrap draws; the scoring function uses its default 10,000 draws. The
ranking comparison uses synthetic gold states, not the classifier output.

The report labels the gold states synthetic and keeps classifier fidelity
separate from affective ranking. No human-acceptance or product-utility claim
is supported by these rows.

## Gates and limits

`module-mechanism.json.acceptance` records `coverage_pass=false`,
`correctness_pass=true`, `overall_mechanism_gates_pass=false`, and
`status="NOT RUN"`; the report explains that this mechanism evaluation has no
human-acceptance gate. The phase-level failure is the graph eligibility gate,
not an affective-pair completion failure.

There are no surprisal metrics or surprisal gate fields in this mechanism
artifact or its `failures-and-coverage.json`. No surprisal conclusion is drawn
from this phase. Raw observations and outputs remain unchanged.
