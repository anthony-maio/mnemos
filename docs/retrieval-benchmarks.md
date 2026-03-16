# Retrieval Benchmarking

Mnemos ships a benchmark harness via `mnemos-benchmark` to measure retrieval quality and latency across retrieval modes and local store implementations.

## Metrics

- `Recall@k`: fraction of queries where at least one relevant chunk appears in top-k results.
- `MRR`: mean reciprocal rank of the first relevant result.
- `p95 latency (ms)`: 95th percentile retrieval latency for retrieval path execution.

## Quick Start

```bash
mnemos-benchmark --stores memory,sqlite --retrievers baseline,engine --top-k 5
```

Copy-paste-safe contributor command for the shipped trust gate:

```bash
MNEMOS_LLM_PROVIDER=mock MNEMOS_EMBEDDING_PROVIDER=simple \
mnemos-benchmark --stores memory,sqlite --retrievers baseline,engine \
  --dataset-pack claim-driving --top-k 1 --repetitions 2 --enforce-production-gate
```

Retriever modes:
- `baseline`: direct vector retrieval from the store (`store.retrieve`)
- `engine`: full Mnemos retrieval path (`MnemosEngine.retrieve`)

`baseline` is intentionally scope-agnostic by default so scoped retrieval value is visible in cross-project benchmarks. To make baseline scope-aware for strict apples-to-apples runs, pass `--baseline-scope-aware`.

The report includes a `comparisons` section with per-store deltas:
- `delta_engine_minus_baseline.recall_at_k`
- `delta_engine_minus_baseline.mrr`
- `delta_engine_minus_baseline.latency_p95_ms`

When `--repetitions` is greater than `1`, the top-level report shape changes:
- `runs[]`: each full benchmark run with its own `results`, `comparisons`, and `gates`
- `summary.runs`
- `summary.passed_runs`
- `summary.failed_runs`
- `summary.all_passed`
- `summary.stores.<store>.latency_p95_ratio.min|max`
- `summary.stores.<store>.mrr_lift_ratio.min|max`

This command uses the built-in benchmark dataset.

## Claim-Driving Dataset Pack

Mnemos includes fixed-version datasets for replacement-claim evaluation:

- `benchmarks/datasets/contradiction-update-v1.jsonl`
- `benchmarks/datasets/preference-drift-v1.jsonl`
- `benchmarks/datasets/cross-project-scope-v1.jsonl`

Run:

```bash
mnemos-benchmark --stores memory --retrievers baseline,engine --dataset-pack claim-driving --top-k 1
```

## Production Replacement Gate

The benchmark report includes:

- `gates.production_replacement.required_mrr_lift_ratio`
- `gates.production_replacement.max_latency_p95_ratio`
- `gates.production_replacement.latency_ratio_floor_ms`
- `gates.production_replacement.passed`
- `gates.production_replacement.details[]`

Enforce gate (non-zero exit on failure):

```bash
mnemos-benchmark --stores memory --retrievers baseline,engine --dataset-pack claim-driving --top-k 1 --enforce-production-gate
```

Repeat the full benchmark and require every run to pass:

```bash
mnemos-benchmark --stores memory,sqlite --retrievers baseline,engine --dataset-pack claim-driving --top-k 1 --repetitions 2 --enforce-production-gate
```

Default gate thresholds:

- MRR lift ratio: `>= 0.15`
- p95 latency ratio: `<= 2.0`
- latency floor for ratio denominator: `1.0 ms`

CI profile-specific production gates:

- `memory` profile: `MRR lift >= 0.15`, `p95 ratio <= 2.0`
- `sqlite` default profile: `MRR lift >= 0.15`, `p95 ratio <= 4.0`

Current trust-release benchmark summary on the shipped SQLite path:

- repeated `claim-driving` gate: `2/2` runs passed
- SQLite `mrr_lift_ratio` range: `0.40` to `1.67`
- SQLite `latency_p95_ratio` range: `1.05` to `2.49`

## Custom Dataset Format

Use `.json` (array) or `.jsonl` (one object per line). Each item must include:

- `id`: stable document ID
- `content`: stored memory text
- `queries`: list of query variants expected to retrieve that document (can be empty for distractor documents)

Optional scoped-memory fields:
- document-level: `scope` (`project|workspace|global`), `scope_id`
- query object fields: `text`, `current_scope`, `scope_id`, `allowed_scopes`, `relevant_ids`

Example:

```json
[
  {
    "id": "aws-ecs",
    "scope": "project",
    "scope_id": "repo-alpha",
    "content": "We deploy microservices on AWS ECS with blue-green releases.",
    "queries": [
      {
        "text": "blue green ecs deployment",
        "current_scope": "project",
        "scope_id": "repo-alpha",
        "allowed_scopes": ["project", "global"]
      }
    ]
  }
]
```

Run:

```bash
mnemos-benchmark --stores sqlite --retrievers baseline,engine --dataset ./benchmarks/retrieval.json --top-k 10
```
