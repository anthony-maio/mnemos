# Comparison limits and next validation work

These are read-only interpretations of preserved results. No production
threshold, evaluation case, or frozen model setting was changed.

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
