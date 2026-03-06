"""
tests/test_benchmark.py — Retrieval benchmark metric tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnemos.benchmark import (
    BenchmarkDocument,
    _build_comparisons,
    compute_retrieval_metrics,
    evaluate_production_replacement_gate,
    load_documents,
    run_retrieval_benchmark,
)


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
    assert result["retriever"] == "baseline"
    assert result["query_count"] == 2
    assert "recall_at_k" in result
    assert "mrr" in result
    assert "latency_p95_ms" in result


def test_run_retrieval_benchmark_engine_mode() -> None:
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
        retriever="engine",
        top_k=1,
        documents=documents,
    )

    assert result["store_type"] == "memory"
    assert result["retriever"] == "engine"
    assert result["query_count"] == 2
    assert "recall_at_k" in result
    assert "mrr" in result
    assert "latency_p95_ms" in result


def test_build_comparisons_includes_delta() -> None:
    results = [
        {
            "store_type": "sqlite",
            "retriever": "baseline",
            "recall_at_k": 0.5,
            "mrr": 0.4,
            "latency_p95_ms": 10.0,
        },
        {
            "store_type": "sqlite",
            "retriever": "engine",
            "recall_at_k": 0.7,
            "mrr": 0.6,
            "latency_p95_ms": 12.0,
        },
    ]

    comparisons = _build_comparisons(results)

    assert len(comparisons) == 1
    comp = comparisons[0]
    assert comp["store_type"] == "sqlite"
    assert comp["delta_engine_minus_baseline"]["recall_at_k"] == pytest.approx(0.2)
    assert comp["delta_engine_minus_baseline"]["mrr"] == pytest.approx(0.2)
    assert comp["delta_engine_minus_baseline"]["latency_p95_ms"] == pytest.approx(2.0)


def test_evaluate_production_replacement_gate_uses_floor_and_passes() -> None:
    comparisons = [
        {
            "dataset": "claim-driving-a",
            "store_type": "sqlite",
            "baseline": {"mrr": 0.5, "latency_p95_ms": 0.2},
            "engine": {"mrr": 0.7, "latency_p95_ms": 1.0},
        }
    ]

    gate = evaluate_production_replacement_gate(
        comparisons,
        min_mrr_lift=0.15,
        max_latency_ratio=2.0,
        latency_floor_ms=1.0,
    )

    assert gate["passed"] is True
    assert gate["failed_pairs"] == 0
    assert gate["details"][0]["latency_p95_ratio"] == pytest.approx(1.0)


def test_evaluate_production_replacement_gate_fails_when_lift_too_low() -> None:
    comparisons = [
        {
            "dataset": "claim-driving-b",
            "store_type": "sqlite",
            "baseline": {"mrr": 0.8, "latency_p95_ms": 5.0},
            "engine": {"mrr": 0.84, "latency_p95_ms": 6.0},
        }
    ]
    gate = evaluate_production_replacement_gate(comparisons, min_mrr_lift=0.15)
    assert gate["passed"] is False
    assert gate["failed_pairs"] == 1


def test_load_documents_allows_empty_query_lists(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    payload = [
        {"id": "d1", "content": "distractor without benchmark queries", "queries": []},
        {"id": "d2", "content": "active benchmark doc", "queries": ["active query"]},
    ]
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")

    docs = load_documents(dataset_path)

    assert len(docs) == 2
    assert docs[0].queries == ()
    assert docs[1].queries == ("active query",)
