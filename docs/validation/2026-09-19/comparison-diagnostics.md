# Comparison limits and next validation work

These are read-only interpretations of preserved results. No production
threshold, evaluation case, or frozen model setting was changed.

## Terminal encoding comparison

The [independent three-arm comparison](encoding-terminal-local-v1/README.md)
completed execution with all three arms still marked incomplete. It covers
600 observations, 2,220 source events, and 1,320 completed automatic policy
decisions per arm. Partial observations total 34 across the arms, all with a
missing correction target. Surprisal filters 120 events; length/noise filters
96 different events. Decision coverage establishes that both gates actually
ran, without establishing a quality benefit.

Among applicable complete paired observations, length/noise minus full has
an answerable hit-rate difference of -12.41 percentage points, with a
session-clustered 95% interval of [-19.35, -6.38], over 282 pairs. Eighteen
incomplete pairs remain excluded. The result is conditional on that subset
and uses exact-text fixture judgments; it cannot establish general semantic
quality or complete-run acceptance. The ungated comparison is also conditional
and does not establish a clear benefit from surprisal. No defaults or frozen
criteria change on the basis of these results.

## Terminal retrieval comparison

The [retrieval terminal archive](retrieval-terminal-local-v1/README.md)
preserves an incomplete four-strategy comparison. It contains 900 query rows
and 3,600 strategy results: 300 answerable, 300 no-answer, and 150 each for
the semantic and state state-pair kinds. The run has 589 complete and 11
partial session observations. The 11 partial rows all report a missing
accepted correction target; the 19 unverified quality rows remain excluded.

Hybrid, lexical, and vector are each compared with full as treatment minus
full. Every paired quality estimate is 0.0 with a 95% interval of [0.0, 0.0].
Eligible counts are 289/300 answerable, 300/300 no-answer, and 146/150 for
each state-pair kind. The underlying hit rates are 106/289, 278/300, and
83/146 respectively. These are conditional quality diagnostics. Coverage is
incomplete, the quality gate is NOT RUN, and human acceptance remains not run.
Latency is descriptive only under the protocol; no latency gate or confidence
interval is claimed.

## Legacy synthetic replacement benchmark

The [checkpoint archive](embedding-client-checkpoint-v1/README.md) retains
the same accuracy results on `fc72cc2` and `39b9c79`, across both stores and
both repetitions:

| Dataset | Baseline MRR | Engine MRR | Relative lift |
| --- | ---: | ---: | ---: |
| contradiction-update-v1 | 0.500 | 0.375 | -25% |
| preference-drift-v1 | 0.375 | 0.375 | 0% |
| cross-project-scope-v1 | 0.714 | 0.429 | -40% |

All pairs fail the unchanged +15% lift requirement. This is not evidence of
a regression caused by client pooling: the accuracy values match the prior
snapshot exactly. It is also not yet a controlled retrieval comparison.

The baseline batches document embeddings, while engine ingestion embeds one
document at a time. `SimpleEmbeddingProvider` updates IDF as documents arrive,
so those paths receive different vectors. Baseline scope filtering is also
disabled by default, while the engine applies scope. The ingestion pipelines
differ in affective tags, graph edges, and pending episodes as well.

A separate diagnostic should reuse identical stateless document/query vectors
and the same scope policy for both retrieval paths. Keep datasets, top-k,
repetitions, and acceptance thresholds unchanged. Preserve the original
benchmark failures; a controlled diagnostic must not replace them silently.
The separately queued real-model retrieval comparison already uses identical
accepted snapshots and remains a distinct experiment.

That [offline control has now completed](fair-comparison-diagnostic-v1/README.md)
with a stateless token-hash embedder injected into both paths and scope-aware
baseline retrieval. It records 24 runs and 184 per-query outcomes. Cross-project
MRR is 1.0 for both paths on both stores in both repetitions. Contradiction and
preference outcomes still differ by store and path; identical vectors alone
do not establish equivalent candidate selection or a production quality claim.
The replacement gate remains failed overall in both repetitions. The initial
non-injected attempt and a subsequent rejected internal prediction string are
preserved separately from the valid run. Subsequent diagnosis found that the
26 differing baseline pairs were exact cosine ties. The in-memory store used
insertion order, whereas both SQLite paths used ascending memory ID. The
[retrieval parity repair](retrieval-parity-checkpoint-v1/README.md) aligns these
contracts and rejects empty/nonpositive in-memory requests consistently. Its
repeated control now has zero baseline identity mismatches across 46 pairs.
Engine results and both replacement gates remain negative; this storage fix
does not establish semantic quality or resolve every engine-path difference.

The remaining engine/store differences in that control have another limitation:
baseline chunks use stable benchmark IDs, while engine ingestion creates UUID4
identities independently for each run. Exact semantic ties therefore resolve
to different facts when the final ID tie-break runs. Default baseline retrieval
does not use graph or affective contributions. All fixture documents fit in
the vector candidate pool, so the lexical union adds no new candidates. Raw
engine UUID differences are not evidence of a storage mismatch, and tied old/new
vectors cannot establish which assertion is semantically current. A subsequent
development diagnostic should record score tie groups and stable content IDs,
or use prospectively fixed non-tied vectors. Production identity generation and
held-out fixtures must not be changed to improve this synthetic metric.

## Held-out graph eligibility

The [completed mechanism archive](mechanism-local-v1/README.md) has 150 complete
observations across 50 cases, but zero eligible cases. Eligibility requires a
target outside the direct candidate pool, above the semantic relevance floor,
and reachable along a persisted path from the actual selected seed.

Read-only inspection found target cosine ranks 33 or 34 among 64 primary
facts, outside the 30-item vector pool. Similarities were approximately
0.584 to 0.704, above the unchanged 0.2 floor. The actual selected seed matched
the authored seed label in only 10 of 50 cases. No target had a persisted
path from the actual selected seed; only five had a path from the authored
seed. Authored seed/bridge labels do not establish a production graph path.

The benchmark correctly records these cases as ineligible. Structural checks
passing does not supply missing utility evidence, and an empty eligible-miss
list cannot be interpreted as successful recall. This observation does not
identify a traversal bug; it shows that the intended associative retrieval
condition was not produced by these texts and frozen embeddings.

Further work should start with a separately frozen development preflight that
records actual seed choice, direct target rank, scope, persisted path, energy,
and result. Keep production parameters unchanged and retain both positive and
negative controls. Do not repair, selectively discard, or re-label the failed
held-out cases to manufacture coverage. Any new held-out evaluation requires
an independently fixed protocol and must remain separate from this result.
