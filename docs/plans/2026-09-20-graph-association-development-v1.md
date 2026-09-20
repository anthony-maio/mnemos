# Graph association development protocol v1

Status: draft design; NOT RUN. Exact authored cases, vector bytes, relation
weights, fanout bounds, and all configuration values must be fixed in a
committed manifest before execution. This document alone is not a frozen
experiment or evidence of a working association channel.

This protocol is a prospective development diagnostic for a possible
source-backed association channel. It is not a rerun of the held-out
evaluation, a threshold search, or a quality acceptance run. It must use a
fresh development artifact and frozen inputs. It must not change production
defaults, held-out labels, held-out facts, or the meaning of an existing
semantic edge.

The existing held-out result remains failed and must be reported that way:
50 cases, three repetitions each, 150 observations, zero eligible
observations, zero target returns, and no structural path from the actual
selected seed to the target even when the diagnostic walk ignores scope,
depth, and energy limits. The existing degree diagnostic also remains a
diagnostic only: one of five renamed development cases had a path at degree
16, and its target was still absent from the top five. Neither result is
reinterpreted as evidence for a production change.

## Question and boundary

The current graph derives scoped mutual semantic nearest-neighbor edges. The
development data labels `seed_id`, `bridge_id`, and `target_id` describe an
author's intended scenario, but labels do not create edges. The current
fixture creates one episodic event and one evidence row per fact. It therefore
does not contain a source-backed relation that the graph can discover.

Production SQLite already records the relationship that could support a
separate diagnostic: `fact_evidence.source_event_id`. Multiple current facts
may reference one actual source event. The question for this protocol is
whether those durable relations can provide bounded, scope-safe associative
recall, and whether that behavior is sufficiently distinct from semantic
similarity to justify designing a separate typed channel. A source relation
must come from persisted event and evidence rows. It must never be inferred
from fixture labels, matching text, fact IDs, insertion order, or a synthetic
seed/bridge/target link.

This protocol does not decide whether a source relation should be a product
feature. It produces the evidence needed for that decision. A path, a target
becoming reachable, or a rank change is an observation, not a quality claim.

## Five disjoint development cases

Create five cases with unique case IDs, fact IDs, source event IDs, evidence
IDs, scope IDs, and authored content. Cases must also have disjoint project,
workspace, and profile identities unless a case explicitly tests a
cross-scope negative control. Every focal fact must be persisted through the
same SQLite evidence APIs used by the product. The case manifest must record
the authored source-event pattern and the expected relation counts, but the
measurement must derive relations from the database rather than reading those
expectations as edges.

1. **shared_capture_triplet**. One captured synthetic user event produces a seed,
   bridge, and target fact. Their `fact_evidence` rows share one
   `source_event_id`; unrelated distractors come from separate events. The
   target should be outside the direct vector and lexical candidate union if
   that is achieved by the frozen inputs. If it is a direct candidate, retain
   the case as a direct-control observation and do not count it as associative
   success.
2. **shared_extraction_batch**. One captured source and one durable
   extraction job produce three or four supported facts that share the source
   event. The relation must be visible through the evidence rows, not through
   a job ID alone. Include a distractor job with similar wording but a
   different source event.
3. **semantic_only_chain**. Seed, bridge, and target each have distinct
   source events. This is the control for the current semantic graph. The
   source arm must not invent a relation merely because the facts are
   semantically related or because their labels form a chain.
4. **scope_isolation_pair**. Two projects or profiles contain similar facts
   and separate source events. Query one context. A foreign profile or
   project relation must be rejected. A project/workspace relation is allowed
   only where the existing `_nodes_can_connect` scope contract allows it.
   Record both the positive same-scope relation and the foreign-scope negative
   control.
5. **revision_forget_replay**. Begin with shared-source facts, then correct
   one fact to a new version/evidence relationship, forget one fact, replay
   the original event, and restart the store. The old relation must disappear
   when its evidence no longer exists. A remaining dependent fact and its
   source must remain. Replay suppression must not recreate the forgotten
   relation after restart.

Each case should contain roughly eight to twelve facts, including three or
four focal facts and distractors. The exact count is less important than
having real source events, current evidence, and enough negatives to expose
scope leakage and accidental text matching. Case construction must be
complete before any arm is measured. No case may be selected, rewritten, or
removed after seeing an observation.

## Diagnostic arms

Run the same frozen query, context, candidates, and ranking configuration in
three explicitly labelled arms.

* **semantic** uses the current persisted `SEMANTIC_SIMILAR` graph and the
  existing production mutual-nearest-neighbor rules.
* **source_association** derives a diagnostic relation view from actual
  SQLite evidence joins. It must use a separate relation table or edge type
  in disposable storage. It must not write source relations as
  `SEMANTIC_SIMILAR` and must not alter production graph defaults.
* **combined** uses both views while preserving relation type and provenance
  in every edge and score component. It must be possible to attribute a path
  or rank change to semantic edges, source relations, or both.

The source arm is a diagnostic implementation of this protocol, not an
approved production algorithm. A fixed binary source relation is preferable
for the first run. If a weight is used, predeclare it in the protocol and use
the same value in every case and repetition. Do not tune it after observing
target ranks. Bound source-event fanout; an over-limit group is reported as
`overflow` and is not expanded into an unbounded clique.

## Frozen measurement configuration

Freeze embeddings once and reuse the exact bytes for every case, arm,
repetition, restart, and insertion-order control. This protocol makes no
external provider or real-model calls. Deterministic fixture extraction may
exercise capture/job processing and must be labelled as fixture behavior. A separate approved preparation step may create a
new vector artifact, but its manifest and digest must be bound before this
protocol runs.

Keep the existing measurement behavior and predeclare these effective values:

| Setting | Value or rule |
| --- | --- |
| query, document, and symmetric embedding identity | frozen artifact, exact digest recorded |
| vector and lexical candidate union | production candidate construction; record both sets and their union |
| selected seed | actual highest-cosine vector candidate; authored labels never force a seed |
| semantic relevance floor | `0.2` |
| semantic edge threshold | `0.6` |
| neighbor budget | current production default `4`; no post hoc alternatives |
| activation threshold | `0.3` |
| decay | `0.2` |
| maximum hops | `3` |
| maximum visited nodes | `1000` |
| graph weight | `0.1`, with source contribution separately reported |
| output budget | existing top-k and context-token budget, recorded per observation |
| scopes | exact profile/project/workspace context from the case |

Record vector seed IDs, direct candidate IDs, lexical IDs, relevance scores,
semantic graph energy, source relation contribution, and final ranks. A
target being reachable is not enough to call it retrieved. A target already
in the direct candidate union is a direct control, not associative utility.
The measurement must retain negative, ineligible, and missing-path rows.

## Source relation derivation

For each case and arm, take a consistent SQLite snapshot and derive source
relations using a bounded SQL-backed procedure:

1. Join current eligible fact evidence to its exact `source_event_id` and
   validate that the source event exists. Do not use an evidence ID as a
   source event ID unless the database itself contains that relationship.
2. Validate source role/type, current fact version, profile, project,
   workspace, and source-event scope. Apply the same permitted scope rule in
   both directions. Foreign profile relations are rejected; project/workspace
   relations are accepted only under the existing scope contract.
3. Group by source event and apply a fixed fanout bound. Record groups that
   are empty, foreign, stale, malformed, or over the bound. Do not silently
   drop a rejected relation or turn an over-limit group into a clique.
4. Emit deterministic relation IDs and retain the source event ID, fact IDs,
   source role, versions, and scope in relation provenance. Source relations
   must be stable after a close and reopen of the database.

The source arm must never use `seed_id`, `bridge_id`, `target_id`, authored
content similarity, insertion order, or an inferred canonical relation to
create an edge. The semantic arm remains the only arm that uses semantic
nearest-neighbor graph construction.

## Lifecycle and correctness controls

Use a fresh disposable SQLite database for each case, arm, and repetition.
Where a comparison requires it, make byte-identical copies before queries and
record the source and copy SHA-256 digests. Keep the following controls in
the observation artifact rather than summarizing them away:

* close and reopen the store, then compare relation manifests, edge manifests,
  candidate pools, and results;
* insert facts in forward and reverse order and require the same source
  relation set and equivalent combined results;
* remove semantic edges in an edge-cut control and show whether the source
  path remains, then remove source relations and show whether the semantic
  path remains;
* correct a fact and verify that stale versions cannot retain a relation;
* forget one member of a shared source and preserve the source and remaining
  member; forgetting the final dependent removes the relation;
* replay a forgotten source event and restart, verifying suppression does not
  recreate the relation;
* exercise foreign profile, foreign project, and permitted workspace/project
  boundaries;
* expose graph readiness as pending until all required relation sources are
  derived. A partial relation view must not be labelled ready.

The actual selected vector seed and the source-derived relation seed must be
recorded separately. If they differ, preserve both facts in the report. A
query-selected seed is an observation; it is never replaced with the
authored seed to make a path eligible.

## Provenance and artifact schema

The run must be reproducible offline and must bind the implementation before
the first observation. At minimum, record the full source commit, clean state,
hashes of all runnable source files imported by the helper, protocol hash,
case-manifest hash, frozen-vector hash, Python/runtime version, SQLite
version, and all effective retrieval and relation options. Capture source
hashes again after execution and fail the provenance check if they changed.

Use one JSON artifact with this shape. Additional fields are allowed, but
these fields must retain all negatives and controls:

```json
{
  "schema_version": "graph-association-development-v1",
  "status": "complete|failed|blocked",
  "execution_complete": true,
  "coverage_pass": false,
  "source": {
    "commit": "full git hash",
    "clean_before": true,
    "clean_after": true,
    "files": {"path": "sha256"}
  },
  "protocol_sha256": "sha256",
  "case_manifest_sha256": "sha256",
  "frozen_embeddings_sha256": "sha256",
  "options": {},
  "arms": ["semantic", "source_association", "combined"],
  "cases": [],
  "observations": []
}
```

Each observation should include the following fields:

```json
{
  "case_id": "shared_capture_triplet",
  "repetition": 0,
  "arm": "source_association",
  "context": {},
  "query": {"text_digest": "sha256", "embedding_digest": "sha256"},
  "candidate_pool": {
    "vector_ids": [],
    "lexical_ids": [],
    "union_ids": [],
    "actual_selected_seed_id": null
  },
  "source_relations": [
    {
      "relation_id": "...",
      "source_event_id": "...",
      "fact_ids": [],
      "scope": {},
      "status": "used|rejected|overflow|stale|missing"
    }
  ],
  "edges": {
    "semantic": {},
    "source_association": {},
    "combined": {}
  },
  "target": {
    "id": "...",
    "direct_candidate": false,
    "structural_path": false,
    "activation_path": false,
    "energy": 0.0,
    "rank": null,
    "in_top_k": false
  },
  "controls": {
    "restart_pass": true,
    "insertion_order_pass": true,
    "scope_pass": true,
    "edge_cut_pass": true,
    "correction_pass": true,
    "forget_pass": true,
    "replay_suppression_pass": true
  },
  "artifacts": {
    "sqlite_sha256": "sha256",
    "relation_manifest": "path",
    "edge_manifest": "path"
  }
}
```

Record raw score components and ranks for every candidate, not just the
target. Include explicit reasons for no path, no activation, ineligible
source, direct candidate, scope rejection, stale evidence, and overflow.
Keep `fixture_label`, if present, separate from measured graph fields.

## Execution, clustering, and interpretation gates

Run all five cases in all three arms for three deterministic repetitions.
Repeated deterministic rows are clustered by fixed `case_id`; they are not
150 or 45 independent cases. Summaries must retain all repetitions and report
case-level and arm-level counts. If bootstrap intervals are used later, use
case clusters and report the template/case limits; do not use repetitions as
the denominator.

The minimum execution gate is 5 cases x 3 arms x 3 repetitions with all
negative and control rows retained. `execution_complete` means the helper
finished writing the artifact. `coverage_pass` is separate and may be false.
Require all five cases to have complete observations and all lifecycle and
scope controls to pass before interpreting the arms. Require at least three
cases with genuine shared source-event evidence and at least one
distinct-event semantic-chain case. Otherwise report association coverage
failed and stop.

A future typed-channel design is justified only if, on the predeclared cases,
source-backed paths are reproducible after restart, correction, forget, replay,
and scope controls; source provenance survives every transition; and the
source arm changes reachability for at least one target outside the direct
candidate union. Rank and top-k effects must also be reported, including no
effect. A positive structural path alone is insufficient, and no result here
authorizes a production default, threshold, graph-builder version, or
held-out dataset change.

If the source arm finds no relations because the cases lack genuine shared
events, report a fixture/contract coverage failure. If relations exist but do
not change reachability or ranking, report no demonstrated utility. If
relations change results but fail lifecycle, scope, or provenance controls,
stop at the diagnostic result. Only a durable, bounded, scope-safe and
behaviorally distinct result permits a separate design review for a typed
association channel such as `EVIDENCE_ASSOCIATED`; it does not permit
silently folding those edges into `SEMANTIC_SIMILAR`.

No observation from this protocol may be used to relabel or reinterpret the
original held-out zero-coverage result. Any production implementation and any
new model-backed evaluation require a separately reviewed protocol, fresh
source and vector provenance, and independent regression coverage.
