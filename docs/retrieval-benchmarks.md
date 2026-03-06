# Retrieval Benchmarking

Mnemos ships a benchmark harness via `mnemos-benchmark` to measure retrieval quality and latency across storage backends.

## Metrics

- `Recall@k`: fraction of queries where at least one relevant chunk appears in top-k results.
- `MRR`: mean reciprocal rank of the first relevant result.
- `p95 latency (ms)`: 95th percentile retrieval latency for `store.retrieve(...)`.

## Quick Start

```bash
mnemos-benchmark --stores memory,sqlite,qdrant --top-k 5
```

This command uses the built-in benchmark dataset.

## Custom Dataset Format

Use `.json` (array) or `.jsonl` (one object per line). Each item must include:

- `id`: stable document ID
- `content`: stored memory text
- `queries`: list of query variants expected to retrieve that document

Example:

```json
[
  {
    "id": "aws-ecs",
    "content": "We deploy microservices on AWS ECS with blue-green releases.",
    "queries": ["blue green ecs deployment", "microservices platform on aws ecs"]
  }
]
```

Run:

```bash
mnemos-benchmark --stores qdrant --dataset ./benchmarks/retrieval.json --top-k 10
```

## Qdrant Notes

- For local embedded mode, pass `--qdrant-path`.
- For remote mode, pass `--qdrant-url` and optionally `--qdrant-api-key`.
- Use `--qdrant-collection` to isolate repeated runs.
