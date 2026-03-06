"""
tests/test_benchmark.py — Retrieval benchmark metric tests.
"""

from __future__ import annotations

from mnemos.benchmark import BenchmarkDocument, compute_retrieval_metrics, run_retrieval_benchmark


def test_compute_retrieval_metrics_recall_mrr_and_p95() -> None:
    retrieved_ids = [
        ["a", "b", "c"],  # hit at rank 1
        ["x", "y", "z"],  # miss
        ["m", "k", "n"],  # hit at rank 2
    ]
    relevant_ids = [
        {"a"},
        {"q"},
        {"k"},
    ]
    latencies_ms = [10.0, 20.0, 30.0]

    metrics = compute_retrieval_metrics(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        latencies_ms=latencies_ms,
        top_k=3,
    )

    assert metrics["query_count"] == 3
    assert metrics["recall_at_k"] == 2 / 3
    assert metrics["mrr"] == (1.0 + 0.0 + 0.5) / 3
    assert metrics["latency_p95_ms"] >= 20.0


def test_run_retrieval_benchmark_memory_backend() -> None:
    documents = [
        BenchmarkDocument(
            id="aws",
            content="We deploy services on AWS ECS.",
            queries=("aws ecs deployment",),
        ),
        BenchmarkDocument(
            id="postgres",
            content="We use PostgreSQL backups every night.",
            queries=("postgres backup schedule",),
        ),
    ]

    result = run_retrieval_benchmark(
        store_type="memory",
        top_k=1,
        documents=documents,
    )

    assert result["store_type"] == "memory"
    assert result["query_count"] == 2
    assert "recall_at_k" in result
    assert "mrr" in result
    assert "latency_p95_ms" in result
