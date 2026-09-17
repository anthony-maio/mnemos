# Graph retrieval evaluation after the coverage failure

The completed sourceb7 held-out run has 150 observations across 50 cases.
Every case meets the semantic floor and keeps the target outside the direct
retrieval pool, but none has a persisted path from the query-selected seed
to the target. The coverage gate fails at zero eligible cases against the
required 40. Graph correctness controls pass; graph retrieval utility remains
unmeasured. Keep the [original results](../validation/2026-09-16/real-model-v3/mechanism-heldout/REVIEW.md)
unchanged.

The raw rows show that the query selects a competitor as its seed in 117 of
150 observations. The authored seed label does not determine query seeding.
This is consistent with the mutual top-k graph separating the dense query
neighborhood from the bridge and target. It does not establish that this
graph design improves retrieval on realistic memories.

## Next experiment

1. Add an offline development preflight that uses the same seed selection,
   candidate union, persisted graph, semantic floor, and bounded path test as
   the actual measurement. Report each eligibility condition separately.
2. Use development cases to test whether the unchanged production graph can
   retain useful bridge paths among plausible distractors. Keep deterministic
   manually linked traversal controls separate from semantic graph utility.
3. Freeze the next dataset version, protocol, source revision, model digests,
   thresholds, eligible-denominator requirement, and template clustering
   before collecting a fresh held-out result. Do not relabel these observed
   held-out cases as a new independent test or tune against their answers.
4. Report ineligible cases alongside eligible results. A failed denominator
   remains a failed coverage gate, even if individual correctness checks pass.
   Compare graph-assisted retrieval against the same scoped semantic baseline.

No additional Ollama workload should overlap the active seven-profile
lifecycle run. That run stays bound to sourceb7 and its existing hashes.
Any future mechanism or dataset change needs its own recorded run. Baseline
retrieval remains the default while module benefit is unproved.

## Development preflight result

The [pinned offline preflight](../validation/2026-09-16/graph-preflight-development/README.md)
now calls the production measurement helper from the unchanged sourceb7
snapshot using five cached development cases. None is structurally eligible:
all have a persisted graph, but none has a valid path from the query-selected
seed to the target. No new model or network calls were made. The two earlier
diagnostic attempts are preserved with their protocol limitations. This
completes the preflight work in step 1; it does not establish graph utility
or satisfy the held-out coverage gate.
