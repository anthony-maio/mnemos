import hashlib
import json
from datetime import datetime, timezone

import pytest

from mnemos import benchmark
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider


@pytest.mark.parametrize("store_type", ["memory", "sqlite"])
@pytest.mark.parametrize("retriever", ["baseline", "engine"])
def test_saved_query_evidence_reconstructs_metrics(store_type, retriever, tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "build_embedder_from_env", lambda **_: SimpleEmbeddingProvider())
    documents = [
        benchmark.BenchmarkDocument(
            "cedar", "Cedar backups use amber storage.", ("amber storage",)
        ),
        benchmark.BenchmarkDocument(
            "willow", "Willow builds use cobalt workers.", ("cobalt workers",)
        ),
    ]
    queries = [
        benchmark.BenchmarkQuery("amber storage", {"cedar"}),
        benchmark.BenchmarkQuery("cobalt workers", {"willow"}),
        benchmark.BenchmarkQuery("amber storage", {"cedar"}),
    ]
    result = benchmark.run_retrieval_benchmark(
        store_type=store_type,
        retriever=retriever,
        top_k=1,
        documents=documents,
        queries=queries,
        sqlite_path=tmp_path / "evidence.sqlite",
    )
    saved = json.loads(json.dumps(result))
    evidence = saved["retrieval_evidence"]
    rows = evidence["queries"]
    assert len(rows) == 3
    assert rows[0]["query_sha256"] == rows[2]["query_sha256"]
    assert [row["latency_ms"] for row in rows] == result["query_latencies_ms"]
    stored = {row["chunk_id"]: row for row in evidence["stored_before_queries"]}
    for doc in documents:
        row = stored[evidence["document_chunk_ids"][doc.id]]
        assert row["content_sha256"] == hashlib.sha256(doc.content.encode()).hexdigest()
        assert datetime.fromisoformat(row["created_at"]).tzinfo is not None
    for query, row in zip(queries, rows):
        assert row["relevant_chunk_ids"] == sorted(
            evidence["document_chunk_ids"][i] for i in query.relevant_ids
        )
        assert row["returned_chunk_ids"] == [c["chunk_id"] for c in row["returned_chunks"]]
        assert datetime.fromisoformat(row["started_at"]).tzinfo is not None
    rebuilt = benchmark.compute_retrieval_metrics(
        retrieved_ids=[r["returned_chunk_ids"] for r in rows],
        relevant_ids=[set(r["relevant_chunk_ids"]) for r in rows],
        latencies_ms=[r["latency_ms"] for r in rows],
        top_k=1,
    )
    for name in ("recall_at_k", "mrr", "latency_p95_ms"):
        assert rebuilt[name] == result[name]


def test_evidence_copies_observed_timestamps_before_later_access():
    chunk = MemoryChunk(content="Synthetic evidence")
    recorded = benchmark._chunk_evidence(chunk)
    original = dict(recorded)
    chunk.updated_at = datetime(2030, 1, 1, tzinfo=timezone.utc)
    chunk.access_count += 1
    assert recorded == original
    assert recorded["updated_at"] != chunk.updated_at.isoformat()
    assert recorded["content_updated_at"] is None
