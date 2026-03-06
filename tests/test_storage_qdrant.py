"""
tests/test_storage_qdrant.py — QdrantStore behavior with a fake client.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import cosine_similarity
from mnemos.utils.storage import QdrantStore


@pytest.fixture
def fake_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    class Distance:
        COSINE = "cosine"

    class VectorParams:
        def __init__(self, size: int, distance: str) -> None:
            self.size = size
            self.distance = distance

    class PointStruct:
        def __init__(self, id: str, vector: list[float], payload: dict[str, Any]) -> None:
            self.id = id
            self.vector = vector
            self.payload = payload

    class PointIdsList:
        def __init__(self, points: list[str]) -> None:
            self.points = points

    class _Vectors:
        def __init__(self, size: int) -> None:
            self.size = size

    class _Params:
        def __init__(self, size: int) -> None:
            self.vectors = _Vectors(size)

    class _Config:
        def __init__(self, size: int) -> None:
            self.params = _Params(size)

    class _CollectionInfo:
        def __init__(self, size: int) -> None:
            self.config = _Config(size)

    class _PointView:
        def __init__(self, id: str, vector: list[float], payload: dict[str, Any]) -> None:
            self.id = id
            self.vector = vector
            self.payload = payload

    class _QueryResult:
        def __init__(self, points: list[_PointView]) -> None:
            self.points = points

    class FakeQdrantClient:
        def __init__(
            self,
            *,
            path: str | None = None,
            url: str | None = None,
            api_key: str | None = None,
        ) -> None:
            self.path = path
            self.url = url
            self.api_key = api_key
            self._collections: dict[str, dict[str, Any]] = {}

        def collection_exists(self, collection_name: str) -> bool:
            return collection_name in self._collections

        def create_collection(self, collection_name: str, vectors_config: VectorParams) -> None:
            self._collections[collection_name] = {
                "size": vectors_config.size,
                "points": {},
            }

        def get_collection(self, collection_name: str) -> _CollectionInfo:
            collection = self._collections.get(collection_name)
            if collection is None:
                raise KeyError(collection_name)
            return _CollectionInfo(collection["size"])

        def upsert(
            self, collection_name: str, points: list[PointStruct], wait: bool = True
        ) -> None:
            collection = self._collections[collection_name]
            for point in points:
                collection["points"][point.id] = {
                    "vector": list(point.vector),
                    "payload": dict(point.payload),
                }

        def search(
            self,
            collection_name: str,
            query_vector: list[float],
            limit: int,
            with_payload: bool = True,
            with_vectors: bool = True,
        ) -> list[_PointView]:
            collection = self._collections[collection_name]
            scored: list[tuple[float, str, dict[str, Any]]] = []
            for point_id, point_data in collection["points"].items():
                score = cosine_similarity(query_vector, point_data["vector"])
                scored.append((score, point_id, point_data))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                _PointView(pid, data["vector"], data["payload"]) for _, pid, data in scored[:limit]
            ]

        def query_points(
            self,
            collection_name: str,
            query: list[float],
            limit: int,
            with_payload: bool = True,
            with_vectors: bool = True,
        ) -> _QueryResult:
            collection = self._collections[collection_name]
            scored: list[tuple[float, str, dict[str, Any]]] = []
            for point_id, point_data in collection["points"].items():
                score = cosine_similarity(query, point_data["vector"])
                scored.append((score, point_id, point_data))
            scored.sort(key=lambda x: x[0], reverse=True)
            points = [
                _PointView(pid, data["vector"], data["payload"]) for _, pid, data in scored[:limit]
            ]
            return _QueryResult(points)

        def retrieve(
            self,
            collection_name: str,
            ids: list[str],
            with_payload: bool = True,
            with_vectors: bool = True,
        ) -> list[_PointView]:
            collection = self._collections[collection_name]
            results: list[_PointView] = []
            for point_id in ids:
                point_data = collection["points"].get(point_id)
                if point_data is None:
                    continue
                results.append(_PointView(point_id, point_data["vector"], point_data["payload"]))
            return results

        def scroll(
            self,
            collection_name: str,
            limit: int,
            offset: int | None = None,
            with_payload: bool = True,
            with_vectors: bool = True,
        ) -> tuple[list[_PointView], int | None]:
            collection = self._collections[collection_name]
            all_ids = list(collection["points"].keys())
            start = offset or 0
            end = min(start + limit, len(all_ids))
            batch_ids = all_ids[start:end]
            points = [
                _PointView(
                    point_id,
                    collection["points"][point_id]["vector"],
                    collection["points"][point_id]["payload"],
                )
                for point_id in batch_ids
            ]
            next_offset = end if end < len(all_ids) else None
            return points, next_offset

        def delete(
            self, collection_name: str, points_selector: PointIdsList, wait: bool = True
        ) -> None:
            collection = self._collections[collection_name]
            for point_id in points_selector.points:
                collection["points"].pop(point_id, None)

        def close(self) -> None:
            return None

    fake_module = types.ModuleType("qdrant_client")
    fake_module.QdrantClient = FakeQdrantClient  # type: ignore[attr-defined]
    fake_module.models = types.SimpleNamespace(
        Distance=Distance,
        VectorParams=VectorParams,
        PointStruct=PointStruct,
        PointIdsList=PointIdsList,
    )

    monkeypatch.setitem(sys.modules, "qdrant_client", fake_module)


def test_qdrant_store_crud_and_retrieve(fake_qdrant: None) -> None:
    store = QdrantStore(path=":memory:", collection_name="mnemos_test")

    chunk_a = MemoryChunk(
        id="a",
        content="AWS ECS blue green deployment",
        embedding=[1.0, 0.0, 0.0],
        metadata={"team": "platform"},
    )
    chunk_b = MemoryChunk(
        id="b",
        content="PostgreSQL backup and PITR",
        embedding=[0.0, 1.0, 0.0],
        metadata={"team": "data"},
    )

    store.store(chunk_a)
    store.store(chunk_b)

    hits = store.retrieve([1.0, 0.0, 0.0], top_k=1)
    assert len(hits) == 1
    assert hits[0].id == "a"

    fetched = store.get("a")
    assert fetched is not None
    assert fetched.content == chunk_a.content

    updated = chunk_a.reconsolidate("AWS ECS uses canary and blue-green deployment")
    updated.embedding = [1.0, 0.0, 0.0]
    assert store.update("a", updated) is True
    assert store.get("a") is not None
    assert store.get("a").version == 2  # type: ignore[union-attr]

    assert store.delete("b") is True
    assert store.delete("does-not-exist") is False
    assert len(store.get_all()) == 1

    stats = store.get_stats()
    assert stats["backend"] == "QdrantStore"
    assert stats["total_chunks"] == 1


def test_qdrant_store_filter_fn(fake_qdrant: None) -> None:
    store = QdrantStore(path=":memory:", collection_name="mnemos_test_filters")
    store.store(
        MemoryChunk(
            id="infra-1",
            content="Kubernetes alerting and Grafana dashboards",
            embedding=[0.9, 0.1, 0.0],
            metadata={"team": "infra"},
        )
    )
    store.store(
        MemoryChunk(
            id="marketing-1",
            content="Launch copy tests and pricing page edits",
            embedding=[0.2, 0.8, 0.0],
            metadata={"team": "marketing"},
        )
    )

    filtered = store.retrieve(
        [1.0, 0.0, 0.0],
        top_k=5,
        filter_fn=lambda chunk: chunk.metadata.get("team") == "infra",
    )

    assert len(filtered) == 1
    assert filtered[0].id == "infra-1"


def test_qdrant_store_retrieve_uses_query_points_fallback(fake_qdrant: None) -> None:
    store = QdrantStore(path=":memory:", collection_name="mnemos_test_query_points")
    store.store(
        MemoryChunk(
            id="infra-1",
            content="Kubernetes alerting and Grafana dashboards",
            embedding=[0.9, 0.1, 0.0],
            metadata={"team": "infra"},
        )
    )
    store.store(
        MemoryChunk(
            id="data-1",
            content="PostgreSQL backup and PITR",
            embedding=[0.0, 1.0, 0.0],
            metadata={"team": "data"},
        )
    )

    # Force codepath for clients that expose query_points instead of search.
    delattr(type(store._client), "search")

    hits = store.retrieve([1.0, 0.0, 0.0], top_k=1)
    assert len(hits) == 1
    assert hits[0].id == "infra-1"
