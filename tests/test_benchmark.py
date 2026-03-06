"""
tests/test_benchmark.py — Retrieval benchmark metric tests.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from mnemos.benchmark import (
    BenchmarkDocument,
    BenchmarkQuery,
    _benchmark_store_id,
    _build_comparisons,
    _load_queries,
    _scope_filter_for_query,
    compute_retrieval_metrics,
    evaluate_production_replacement_gate,
    load_documents,
    run_retrieval_benchmark,
)
from mnemos.types import MemoryChunk


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


def test_benchmark_store_id_is_stable_uuid() -> None:
    first = _benchmark_store_id("editor-old")
    second = _benchmark_store_id("editor-old")
    assert first == second
    assert str(uuid.UUID(first)) == first


def test_load_queries_supports_scoped_query_objects(tmp_path: Path) -> None:
    dataset_path = tmp_path / "scope-pack.jsonl"
    dataset_path.write_text(
        (
            '{"id":"alpha-ci","scope":"project","scope_id":"alpha","content":"alpha content",'
            '"queries":[{"text":"which ci here","current_scope":"project","scope_id":"alpha",'
            '"allowed_scopes":["project","global"]}]}\n'
            '{"id":"global-style","scope":"global","content":"global defaults",'
            '"queries":[{"text":"global defaults","current_scope":"project","scope_id":"alpha",'
            '"allowed_scopes":["project","global"],"relevant_ids":["global-style"]}]}\n'
        ),
        encoding="utf-8",
    )

    docs = load_documents(dataset_path)
    queries = _load_queries(dataset_path, docs)

    assert len(queries) == 2
    assert queries[0].text == "which ci here"
    assert queries[0].current_scope == "project"
    assert queries[0].scope_id == "alpha"
    assert queries[0].allowed_scopes == ("project", "global")
    assert queries[1].relevant_ids == {"global-style"}


def test_scope_filter_for_query_honors_scope_boundaries() -> None:
    query = BenchmarkQuery(
        text="ci system",
        relevant_ids={"alpha-ci"},
        current_scope="project",
        scope_id="alpha",
        allowed_scopes=("project", "global"),
    )
    filter_fn = _scope_filter_for_query(query)

    alpha_chunk = MemoryChunk(
        content="alpha",
        metadata={"scope": "project", "scope_id": "alpha"},
    )
    beta_chunk = MemoryChunk(
        content="beta",
        metadata={"scope": "project", "scope_id": "beta"},
    )
    global_chunk = MemoryChunk(
        content="global",
        metadata={"scope": "global"},
    )
    workspace_chunk = MemoryChunk(
        content="workspace",
        metadata={"scope": "workspace", "scope_id": "consulting"},
    )

    assert filter_fn(alpha_chunk) is True
    assert filter_fn(global_chunk) is True
    assert filter_fn(beta_chunk) is False
    assert filter_fn(workspace_chunk) is False


def test_run_retrieval_benchmark_accepts_explicit_queries() -> None:
    documents = [
        BenchmarkDocument(
            id="alpha",
            content="Project alpha uses ECS.",
            queries=("query alpha",),
            scope="project",
            scope_id="alpha",
        ),
        BenchmarkDocument(
            id="beta",
            content="Project beta uses GKE.",
            queries=("query beta",),
            scope="project",
            scope_id="beta",
        ),
    ]
    queries = [
        BenchmarkQuery(
            text="query beta",
            relevant_ids={"beta"},
            current_scope="project",
            scope_id="beta",
            allowed_scopes=("project", "global"),
        )
    ]

    result = run_retrieval_benchmark(
        store_type="memory",
        retriever="baseline",
        top_k=1,
        documents=documents,
        queries=queries,
    )

    assert result["query_count"] == 1


def test_run_retrieval_benchmark_supports_scope_aware_baseline_mode() -> None:
    documents = [
        BenchmarkDocument(
            id="alpha",
            content="Project alpha uses ECS.",
            queries=("deployment target",),
            scope="project",
            scope_id="alpha",
        ),
        BenchmarkDocument(
            id="beta",
            content="Project beta uses GKE.",
            queries=("deployment target",),
            scope="project",
            scope_id="beta",
        ),
    ]
    queries = [
        BenchmarkQuery(
            text="deployment target",
            relevant_ids={"alpha"},
            current_scope="project",
            scope_id="alpha",
            allowed_scopes=("project", "global"),
        )
    ]

    result = run_retrieval_benchmark(
        store_type="memory",
        retriever="baseline",
        top_k=1,
        documents=documents,
        queries=queries,
        baseline_scope_aware=True,
    )

    assert result["query_count"] == 1
