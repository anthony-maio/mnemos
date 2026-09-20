# T18 contract-repair evaluation protocol v1

Status at preparation: NOT RUN. September 20, 2026.

This prospective rerun follows the
[frozen fixture audit](../validation/2026-09-20/retrieval-fixture-audit-v1/README.md).
The old v3 dataset and every old result remain unchanged. V4 contains
previously observed synthetic fixtures. It is not new unseen heldout data,
human acceptance, or evidence of daily host use.

## Frozen input and permitted repair

Input: `benchmarks/local_memory/datasets/local-memory-heldout-v4.jsonl`.
SHA-256: `0c1f74e24d485a37889a18a65463b85c4486824975de8283dda50754d6e434aa`.
Its exact parent bytes have SHA-256
`036fa51ee13271a85a14a62a2fe4ef65d13eaeb290331fe325aa5bf32ce893fc`.

The deterministic generator preserves all source event text, order,
timestamps, roles, evidence identities, correction and forget targets,
replacement bodies, and existing protocol values. The permitted differences
are:

- Forty supported explicit-memory gold bodies equal the existing API body.
  No model output selects the replacement text. Sixty explicit-memory facts
  receive unscored annotations retaining their original gold and API body.
- Twenty state pairs use the amended supported body as `matched_content`.
  Query text, pair identities, state metadata, and relevant identities stay
  unchanged.
- Twenty foreign-profile/project queries become no-answer scope canaries.
  Each gains one uniquely identified local answerable query. The foreign
  query runs first; its target remains outside both the source profile and
  project. The local query uses the original session context.
- All twenty forgotten and twenty superseded gold bodies stay unchanged.
  In particular, unprefixed stale replay remains detectable by the unchanged
  stale-text judge. The exact remembered body is not substituted for these
  historical detection patterns.

There are 200 heldout-labeled sessions and 20 disjoint development sessions,
240 total facts, 100 critical heldout facts, 100 answerable heldout queries,
120 no-answer heldout queries, and 50 state pairs. All existing minimums
remain unchanged. The extra no-answer queries change the composition of
aggregate retrieval metrics; an old-v3/new-v4 difference is not a product
improvement estimate.

V4 validation retains the v3 lifecycle-target checks and rejects explicit
body mismatches, answerable source-scope mismatches, contradictory no-answer
expectations, inconsistent state targets, and missing or localized foreign
canaries. All original scoring functions and thresholds remain unchanged.

## Execution binding

Use a clean source snapshot of the committed candidate. Before inference,
write a preparation record binding the commit, complete code provenance,
this protocol's bytes, dataset bytes, environment, and live model metadata.
Only new isolated output and database directories are permitted. No live
host settings or memory database may be used.

Endpoint: `http://127.0.0.1:11434`, as authorized after the lab machine failed.
The local tags were read again during preparation. Required model bindings:

| Role | Model | Digest |
| --- | --- | --- |
| Generation | `qwen3.5:9b` | `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` |
| Embeddings | `nomic-embed-text:latest` | `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f` |

Keep the earlier lifecycle generation settings: temperature 0, at most
512 generated tokens, 120-second provider timeout, and `think=false`.
Keep top-k 5, a 1,200-token retrieval-context limit, and empty query,
document, and symmetric embedding prefixes. Record the model's inherited
parameters separately; do not infer a context length from the model name.
Record the RTX 4080 hardware and observed Ollama residency without treating
a preflight GPU listing as proof that every request used the GPU.

Run all seven existing profiles in their existing order, three repetitions
per session: baseline, full, and one ablation for each of surprisal gating,
mutable RAG, affective routing, sleep consolidation, and spreading activation.
This is 600 observations per profile and 4,200 in the complete matrix.
Use `run_heldout` with a new run namespace and an `experiment_binding`
containing the preparation record's immutable identifiers.

The existing comparison methods remain Wilson 95% intervals and paired
session-cluster bootstrap intervals with 10,000 resamples and seed 20260916.
Report all original extraction, retention, retrieval, stale, no-answer,
state, byte-growth, token, latency, and lifecycle metrics with their actual
denominators. Scope, correction, forgetting, and replay failures remain
failures even when other metrics improve.

## Failure handling and interpretation

Complete the planned profile/session population. Retain partial observations
and missing target failures; do not rerun only failed rows or impute outcomes.
A terminal incomplete matrix remains incomplete. Process interruption may
resume only against identical recorded bindings and existing checkpoints;
the runner must retain completed rows and use fresh attempt databases for
unfinished work. Never restart because a polling call timed out. Preserve
the terminal report and logs before a separately versioned new attempt.

Do not pool v3 and v4, claim unseen generalization, or interpret fixture
repair as better model quality. Exact text remains a deterministic proxy;
these runs do not establish semantic equivalence or human usefulness.
The known historical-extraction issue remains unchanged, so missing
correction targets may recur. Their presence does not justify rewriting
those inputs or silently changing historical-memory policy.

The four-strategy retrieval comparison and applied encoding comparator need
separate successor bindings after the lifecycle matrix is inspected. This
protocol does not silently reuse their v1 protocols, which explicitly bind
v3. Graph-mechanism eligibility, classifier fidelity, complete native capture,
Desktop, real-provider failure recovery, and the pilot remain separate gates.
