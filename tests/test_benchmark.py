"""
tests/test_benchmark.py — Retrieval benchmark metric tests.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import mnemos.types as types_module
import mnemos.benchmark as benchmark_module

from mnemos.benchmark import (
    BenchmarkDocument,
    BenchmarkQuery,
    _benchmark_store_id,
    _build_comparisons,
    _load_queries,
    _scope_filter_for_query,
    _sqlite_path_for_run,
    summarize_repeat_runs,
    compute_retrieval_metrics,
    evaluate_production_replacement_gate,
    load_documents,
    run_retrieval_benchmark,
)
from mnemos.types import MemoryChunk


@pytest.mark.parametrize("store_type", ["memory", "sqlite"])
@pytest.mark.parametrize("retriever", ["baseline", "engine"])
def test_chronology_does_not_depend_on_clock_resolution(
    monkeypatch, tmp_path, store_type, retriever
):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(types_module, "datetime", FrozenDateTime)
    captured = []
    original_close = benchmark_module._close_store

    def capture_and_close(store):
        captured.extend(store.get_all())
        original_close(store)

    monkeypatch.setattr(benchmark_module, "_close_store", capture_and_close)
    path = Path("benchmarks/datasets/contradiction-update-v1.jsonl")
    documents = load_documents(path)
    result = run_retrieval_benchmark(
        store_type=store_type,
        retriever=retriever,
        top_k=1,
        documents=documents,
        queries=_load_queries(path, documents),
        sqlite_path=tmp_path / "chronology.sqlite",
    )
    by_content = {chunk.content: chunk for chunk in captured}
    assert len(by_content) == len(documents)
    first = by_content[documents[0].content].created_at
    for index, doc in enumerate(documents):
        expected = first + timedelta(seconds=index)
        chunk = by_content[doc.content]
        assert chunk.created_at == chunk.updated_at == chunk.last_observed_at == expected
        if retriever == "engine":
            assert chunk.metadata["original_timestamp"] == expected.isoformat()
    assert result["chronology"]["policy"] == "dataset_order_1s_v1"
    end = datetime.fromisoformat(result["chronology"]["observation_end_exclusive"])
    assert end == first + timedelta(seconds=len(documents))


@pytest.mark.parametrize("retriever", ["baseline", "engine"])
def test_ingest_measurement_excludes_retrieval(monkeypatch, retriever):
    ticks = iter([0.0, 1.0, 10.0, 15.0])
    monkeypatch.setattr(benchmark_module, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    result = run_retrieval_benchmark(
        store_type="memory",
        retriever=retriever,
        top_k=1,
        documents=[
            BenchmarkDocument(
                id="db", content="We use PostgreSQL backups nightly.", queries=("postgres backups",)
            )
        ],
    )
    assert result["ingest_seconds"] == 1.0
    assert result["latency_mean_ms"] == 5000.0


def test_explicit_observation_anchor_is_shared_by_retrievers():
    end = datetime(2026, 11, 8, 12, tzinfo=timezone.utc)
    documents = [
        BenchmarkDocument(
            id="db", content="We use PostgreSQL backups nightly.", queries=("postgres backups",)
        )
    ]
    reports = [
        run_retrieval_benchmark(
            store_type="memory", retriever=mode, top_k=1, documents=documents, observation_end=end
        )
        for mode in ("baseline", "engine")
    ]
    assert reports[0]["chronology"] == reports[1]["chronology"]
    assert reports[0]["chronology"]["observation_end_exclusive"] == end.isoformat()


def test_naive_observation_anchor_is_rejected_before_storage(monkeypatch):
    def unexpected_storage(*args, **kwargs):
        pytest.fail("invalid chronology must not create or clean a store")

    monkeypatch.setattr(benchmark_module, "_cleanup_local_store_artifacts", unexpected_storage)
    with pytest.raises(ValueError, match="timezone"):
        run_retrieval_benchmark(
            store_type="memory",
            top_k=1,
            documents=[BenchmarkDocument(id="db", content="Database policy.", queries=("policy",))],
            observation_end=datetime(2026, 11, 8),
        )


@pytest.mark.parametrize("store_type", ["memory", "sqlite"])
def test_repeated_observations_preserve_first_creation(monkeypatch, tmp_path, store_type):
    captured = []
    original_close = benchmark_module._close_store

    def capture_and_close(store):
        captured.extend(store.get_all())
        original_close(store)

    monkeypatch.setattr(benchmark_module, "_close_store", capture_and_close)
    end = datetime(2026, 11, 8, 12, tzinfo=timezone.utc)
    run_retrieval_benchmark(
        store_type=store_type,
        retriever="engine",
        top_k=1,
        sqlite_path=tmp_path / "repeated.sqlite",
        observation_end=end,
        documents=[
            BenchmarkDocument(
                id=key, content="We use PostgreSQL backups nightly.", queries=("postgres backups",)
            )
            for key in ("first", "second")
        ],
    )
    assert len(captured) == 1
    assert captured[0].created_at == end - timedelta(seconds=2)
    assert captured[0].updated_at == captured[0].last_observed_at == end - timedelta(seconds=1)


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


def test_evaluate_production_replacement_gate_uses_store_specific_latency_defaults() -> None:
    comparisons = [
        {
            "dataset": "claim-driving-c",
            "store_type": "sqlite",
            "baseline": {"mrr": 0.375, "latency_p95_ms": 0.623},
            "engine": {"mrr": 1.0, "latency_p95_ms": 2.012},
        }
    ]

    gate = evaluate_production_replacement_gate(comparisons, min_mrr_lift=0.15)

    assert gate["passed"] is True
    assert gate["details"][0]["max_latency_p95_ratio"] == pytest.approx(4.0)
    assert gate["details"][0]["latency_ratio_floor_ms"] == pytest.approx(2.0)


def test_evaluate_production_replacement_gate_uses_sqlite_latency_floor() -> None:
    comparisons = [
        {
            "dataset": "claim-driving-d",
            "store_type": "sqlite",
            "baseline": {"mrr": 0.7142857142857143, "latency_p95_ms": 0.8},
            "engine": {"mrr": 1.0, "latency_p95_ms": 5.6},
        }
    ]

    gate = evaluate_production_replacement_gate(comparisons, min_mrr_lift=0.15)

    assert gate["passed"] is True
    assert gate["details"][0]["latency_p95_ratio"] == pytest.approx(2.8)


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


def test_summarize_repeat_runs_reports_pass_counts_and_ratio_ranges() -> None:
    reports = [
        {
            "gates": {
                "production_replacement": {
                    "passed": True,
                    "details": [
                        {
                            "store_type": "sqlite",
                            "dataset": "claim-driving",
                            "latency_p95_ratio": 1.2,
                            "mrr_lift_ratio": 0.8,
                        }
                    ],
                }
            }
        },
        {
            "gates": {
                "production_replacement": {
                    "passed": False,
                    "details": [
                        {
                            "store_type": "sqlite",
                            "dataset": "claim-driving",
                            "latency_p95_ratio": 1.8,
                            "mrr_lift_ratio": 0.7,
                        }
                    ],
                }
            }
        },
    ]

    summary = summarize_repeat_runs(reports)

    assert summary["runs"] == 2
    assert summary["passed_runs"] == 1
    assert summary["failed_runs"] == 1
    assert summary["all_passed"] is False
    assert summary["stores"]["sqlite"]["latency_p95_ratio"]["min"] == pytest.approx(1.2)
    assert summary["stores"]["sqlite"]["latency_p95_ratio"]["max"] == pytest.approx(1.8)


def test_sqlite_path_for_run_adds_unique_suffixes(tmp_path: Path) -> None:
    base = tmp_path / "mnemos.sqlite"

    first = _sqlite_path_for_run(base, repetition=1, store_type="sqlite", retriever="baseline")
    second = _sqlite_path_for_run(base, repetition=2, store_type="sqlite", retriever="engine")

    assert first != second
    assert first.name == "mnemos-r1-sqlite-baseline.sqlite"
    assert second.name == "mnemos-r2-sqlite-engine.sqlite"
