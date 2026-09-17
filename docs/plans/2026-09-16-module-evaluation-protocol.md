# T18 module evaluation protocol

This protocol preregisters the next module evaluation. It is a mechanism
evaluation, not a utility or product claim. The current
`local-memory-module-challenge-v1` fixture remains frozen and is not edited.
That fixture has 50 held-out state pairs, but its graph target is also the
answer to a direct query, and its controlled diagnostic does not run graph
traversal. This protocol therefore uses a separate explicit-fact fixture.

The experiment uses real Ollama embeddings and the production SQLite graph
builder. It inserts source-backed facts directly with frozen evidence IDs and
frozen affective states. It does not make 25 automatic extraction calls per
case. The deterministic scorer makes no model calls; an explicitly supplied
classifier provider is a separate measured classifier-fidelity path.
Extraction, provider failures, and lifecycle durability remain covered by the
main lifecycle evaluation.

## Frozen execution contract

The implementation must freeze these values before any held-out case is
opened:

The authored fixture uses `version: "module-mechanism-v1"` and has
`graph_cases` and `affective_pairs` arrays. A graph case has `id`, `split`,
`template_id`, a full `context`, `query`, `target_id`, exactly 64 entries in
`facts`, and a separate `scope_controls` array. Each fact has `id`,
`content`, `evidence_id`, scope and identity fields, and a frozen `embedding`
after preparation. An affective pair has `id`, `split`, `template_id`,
`semantic_query`, `context`, `facts`, and `conditions`; each condition has a
query, frozen state, and target ID. Missing candidate state stays null. The
loader infers `no_state_fact_ids` from those null states; when the field is
present it must list exactly those IDs. The authored pre-embedding file and
the enriched frozen-vector file have separate digests.

| Setting | Frozen value | Source |
| --- | --- | --- |
| Retrieval profile for the semantic control | `baseline` | `mnemos/retrieval.py` |
| Graph profile | `biomimetic` with affective routing disabled | `mnemos/engine.py` and `mnemos/retrieval.py` |
| Retrieval top k | `5` | `ExperimentConfig` |
| Baseline candidate pool | `min(max(top_k * 6, 20), max_candidates)`; `30` at top k 5 | `mnemos/engine.py` |
| Retrieval candidate ceiling | `100` | `mnemos/retrieval.py`, `RetrievalConfig.max_candidates` |
| Relevance floor | `0.2` | `mnemos/retrieval.py`, `RetrievalConfig.relevance_floor` |
| Graph score weight | `0.1` | `mnemos/retrieval.py`, `RetrievalConfig.graph_weight` |
| Initial graph energy | `1.0` | `mnemos/config.py`, `SpreadingConfig.initial_energy` |
| Per-hop decay | `0.2` | `mnemos/config.py`, `SpreadingConfig.decay_rate` |
| Activation threshold | `0.3` | `mnemos/config.py`, `SpreadingConfig.activation_threshold` |
| Maximum hops | `3` | `mnemos/config.py`, `SpreadingConfig.max_hops` |
| Maximum visited nodes | `1000` | `mnemos/config.py`, `SpreadingConfig.max_visited` |
| Maximum neighbors per node | `4` | `mnemos/config.py`, `SpreadingConfig.max_neighbors_per_node` |
| Auto-connect threshold | `0.6` cosine similarity | `mnemos/config.py`, `SpreadingConfig.auto_connect_threshold` |
| Graph neighbor refresh candidate limit | `17` (`4 * max_neighbors + 1`) | `mnemos/engine.py` |
| Persisted graph edge-builder contract | `3` in the current source tree | `mnemos/modules/spreading.py` |
| Affective semantic weight | `0.7` | `mnemos/retrieval.py` |
| Affective state weight | `0.3` | `mnemos/retrieval.py` |

Graph transfer is evaluated as:

```text
E_next = E_current * (1 - 0.2) * edge_weight
```

The target is eligible for graph scoring only when its semantic similarity is
at least the frozen `0.2` relevance floor. Graph energy contributes
`0.1 * E_target` to the final score. A target below the semantic floor is a
declared negative case; graph energy is not allowed to manufacture relevance.

Use one fixed embedding provider, model revision, endpoint, normalization
policy, Python version, SQLite schema, graph contract, and source tree for all
repetitions. Generate each query embedding once and reuse that vector in the
paired snapshots. Record model and embedding digests, options, dataset hash,
graph contract, traversal contract, and source hash before execution.

## Graph associative recall

There are at least 50 fixed graph cases, with 3 repetitions per case. Each
case contains exactly 64 same-context candidate facts before retrieval. The
scope controls are a separate list and are excluded from this 64-fact
denominator. The case template contains:

- one query-related seed fact;
- one bridge fact related to the seed and target;
- one answer-bearing target fact whose answer is fixed before embedding;
- 61 distractor facts, including same-scope semantic competitors; and
- separately counted scope controls containing permitted workspace/global
  facts and foreign project or foreign-profile facts with comparable or higher
  semantic scores.

The text, IDs, source event IDs, profile IDs, project IDs, and target labels are
created and hashed before any held-out embedding or retrieval result is
observed. The target query mentions the seed situation and asks for the
answer-bearing target fact without copying its distinctive answer token. The
bridge and target are related in content so the production edge builder can
connect them, while the query is designed to make the target a graph candidate
rather than an obvious direct answer. Cases that fail these structural checks
are retained and reported as ineligible; they are never removed or rewritten.

Facts are inserted as explicit source-backed `MemoryChunk` rows with evidence
links. The evaluator then calls the production graph refresh path for each
chunk, persists the resulting bounded edges, and records the graph schema,
edge-builder version, edge map, and scope predicate. It does not inject a
handwritten edge into the primary case. An edge-removal control may remove one
persisted derived edge in a copied snapshot to test the expected negative
result.

For every case, create two byte-identical SQLite snapshots after fact and graph
construction:

1. `semantic_control`: spreading disabled and affective routing disabled.
2. `graph_only`: spreading enabled and affective routing disabled.

Both snapshots receive the same precomputed query vector, scope context,
candidate limit, and top k. The direct candidate pool is the stable union of
the bounded vector candidate IDs and bounded lexical candidate IDs after the
scope predicate. The production spreading implementation receives the vector
candidate IDs as its seed candidates and chooses one actual seed by its
documented similarity and stable ID tie-break. Lexical-only IDs remain part
of direct target eligibility and are never silently treated as graph seeds. A
declared `seed_id` or `bridge_id` is an authored fixture label, never an
observation. The evaluator records direct vector IDs, direct lexical IDs,
their union, the actual selected seed, final IDs, semantic score, graph
energy, graph contribution, persisted edge path, hop count, and visited-node
count. Access metadata is allowed to differ after the run; it cannot differ
before the paired query.

The primary graph outcome is graph-only associative recall:

```text
target absent from the semantic control direct candidate pool
and target present in graph-only results
and target graph energy >= 0.3
and target has a valid persisted production edge path
```

Report this as a count and Wilson 95 percent interval over all structurally
eligible cases. Also report the full denominator, ineligible cases, direct
target rank when present, and graph-only misses. No graph utility or quality
claim is made from a case where the target was already a direct candidate.

The following controls are required for every repetition that reaches them:

- **Scope control:** a foreign-profile or foreign-project node must never be
  used as a seed, intermediate, or returned graph result, even when its vector
  is highly similar. Permitted workspace/global nodes must obey the explicit
  membership and profile predicate.
- **Cycle control:** record whether the persisted, production-built graph has
  a three-node cycle in the eligible case subgraph. Production embeddings may
  fail to produce one, so cycle presence is structural coverage rather than a
  required injected primary edge. When a cycle is present, the observed
  energy for every node must equal the maximum bounded path energy under the
  frozen formula and must not increase when the cycle is revisited. Missing
  cycle coverage remains visible in the report.
- **Edge-removal control:** remove the declared bridge edge in a copied graph
  snapshot and report every alternate persisted path to the target before and
  after removal. If a deterministic negative is required, remove the target's
  incoming-edge cutset, meaning every eligible persisted edge that enters the
  target on a qualifying path. The graph-only target must then disappear or be
  reported below the activation threshold; an alternate path cannot be
  silently substituted.
- **Insertion-order control:** build an equivalent case in a second insertion
  order. Persisted eligible neighbor sets and ranked graph results must be
  identical after stable ID tie-breaking.

Coverage fails if fewer than 50 fixed cases are present, fewer than 3
repetitions complete, or fewer than 40 of the 50 cases pass the preregistered
structural eligibility checks. A coverage failure blocks any graph recall
interpretation. Zero foreign-scope leakage and no cycle amplification are
correctness gates, not optional metrics.

## Controlled affective ranking

There are at least 50 fixed semantic/state pairs, with 3 repetitions per pair.
Each pair uses one candidate snapshot, one precomputed semantic query vector,
and the same candidate IDs in both state conditions. Spreading is disabled so
graph energy cannot alter the ranking.

Each candidate has an independently frozen `CognitiveState` with the current
three-axis taxonomy:

- `valence`: `-1.0` negative to `1.0` positive;
- `arousal`: `0.0` calm to `1.0` urgent; and
- `complexity`: `0.0` simple to `1.0` complex.

Each query has an independently frozen state label and exact three-axis target
state. Candidate state labels are written directly with the source-backed
facts. They are not inferred from the ranking result. The semantic wording
and state wording must be distinct, while the vector is held constant for the
ranking comparison. Include state-matched, state-mismatched, and no-state
negative controls. The no-state controls must remain explicitly missing; they
must not be converted into a measured state match.

Run two ranking calculations over the same candidates:

1. semantic-only scoring with affective routing disabled;
2. production affective scoring with the frozen `0.7` semantic and `0.3` state
   weights and the frozen query state.

The deterministic scorer's expected score is:

```text
semantic_score * 0.7 + (1 - normalized_state_distance) * 0.3
```

The primary ranking outcome is the paired change in rank and top k inclusion
of the predeclared state-appropriate candidate. Report the paired difference,
95 percent paired bootstrap interval, exact score components, and all negative
controls. A pair is non-discriminating when the frozen formula predicts the
same order in both conditions; retain it in the denominator and report it
separately.

Query classification fidelity is a separate outcome. Run the real production
`AffectiveRouter` classifier on each query wording, record the parsed state or
missing result, and compare it with the frozen query state using the existing
normalized Euclidean distance. Pre-register distance `<= 0.2` as a fidelity
pass. A classifier parse failure is missing fidelity, never a pass and never a
synthetic neutral state. Classifier fidelity does not alter the frozen-state
ranking result.

Coverage fails if fewer than 50 held-out pairs have frozen candidate and query
states with both conditions and all 3 repetitions complete, or any pair lacks
the same candidate IDs and query vector across its two conditions. Report
ranking and classifier fidelity separately. A rank change without state
fidelity is a scorer sensitivity observation, not evidence of useful affective
routing.

## Analysis and artifacts

The run is paired by case or pair ID and repetition. Shuffle the fixed work
items with seed `20260916`, record the resulting execution order, then use the
same order for the paired snapshots. Do not
tune thresholds, candidate labels, prompts, or held-out inclusion rules after
observing a result. The 5 development cases may expose implementation errors;
they cannot change the held-out fixture or thresholds.

Write these artifacts for every attempt:

- a protocol manifest with this document hash, dataset hash, case IDs, random
  seed, model/provider digests, options, source hash, schema and graph
  versions, and all frozen configuration values;
- a case-level JSONL file with direct candidates, graph edges and paths,
  scores, states, scope decisions, eligibility, control results, and status;
- a paired summary JSON with counts, Wilson intervals, paired bootstrap
  intervals, and coverage gates;
- a machine-readable failure and ineligible-case report; and
- a short Markdown report that labels real embeddings, production graph state,
  deterministic scorer output, classifier fidelity, missing measurements, and
  semantic quality separately.

Do not write raw secrets, credentials, lease tokens, or unredacted host
transcripts into the artifacts. Preserve failed attempts and partial results;
the checkpoint must identify the last completed case and repetition without
claiming completion.

## Implementation tasks

1. Add a protocol loader and validator for the fixed graph cases and affective
   pairs. Validate unique IDs, scope membership, explicit evidence links,
   fixed state labels, target labels, and the 64-candidate graph minimum
   before embedding.
2. Add an explicit-fact fixture builder that inserts source-backed chunks and
   evidence directly, with no extraction or cognition calls. Generate and
   reuse real embedding vectors through the configured embedding provider.
3. Build the canonical production graph through `refresh_graph_chunk`, save
   the graph contract and persisted edge map, and create byte-identical paired
   SQLite snapshots before retrieval.
4. Add graph-only evaluation with fixed vectors, scope predicates, direct
   candidate capture, energy/path reconstruction, cycle checks, insertion-order
   checks, and edge-removal controls.
5. Add affective evaluation with fixed candidate/query states, production
   `score_candidate`, and a separate production classifier-fidelity trace.
6. Add artifact recording, checkpoint recovery, interval calculations, and
   tests for denominator preservation, missing-state handling, scope leakage,
   and no extraction calls.
7. Run the protocol only after its manifest and dataset digest are frozen.
   Keep the existing lifecycle evaluation and module challenge artifacts
   separate from this mechanism report.

No result from this protocol should be described as a retrieval-quality lift,
human-like memory, production host success, or proof of utility without a
separate held-out quality and human-acceptance study.
