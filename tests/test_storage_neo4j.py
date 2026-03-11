"""
tests/test_storage_neo4j.py — Neo4jStore behavior with a fake driver.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from mnemos.types import MemoryChunk
from mnemos.utils.storage import Neo4jStore


@pytest.fixture
def fake_neo4j(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResult:
        def __init__(self, records: list[dict[str, Any]] | None = None, counters: Any = None) -> None:
            self._records = records or []
            self._counters = counters

        def __iter__(self):
            return iter(self._records)

        def single(self) -> dict[str, Any] | None:
            return self._records[0] if self._records else None

        def consume(self) -> Any:
            return self._counters

    class FakeCounters:
        def __init__(self, *, nodes_deleted: int = 0, properties_set: int = 0) -> None:
            self.nodes_deleted = nodes_deleted
            self.properties_set = properties_set

    class FakeSession:
        def __init__(self, driver: "FakeDriver", database: str | None = None) -> None:
            self._driver = driver
            self._database = database

        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def run(self, query: str, **params: Any) -> FakeResult:
            normalized = " ".join(query.split())
            if "CREATE CONSTRAINT" in normalized:
                return FakeResult()

            if "MERGE (chunk:" in normalized and "SET chunk.content" in normalized:
                chunk = dict(params)
                self._driver.nodes[chunk["id"]] = chunk
                return FakeResult(counters=FakeCounters(properties_set=1))

            if "SET chunk.access_count" in normalized:
                node = self._driver.nodes.get(params["chunk_id"])
                if node is None:
                    return FakeResult(counters=FakeCounters(properties_set=0))
                node["access_count"] = params["access_count"]
                node["updated_at"] = params["updated_at"]
                return FakeResult(counters=FakeCounters(properties_set=2))

            if "RETURN chunk.id AS id" in normalized and "WHERE chunk.id = $chunk_id" in normalized:
                node = self._driver.nodes.get(params["chunk_id"])
                if node is None:
                    return FakeResult()
                return FakeResult([dict(node)])

            if (
                "RETURN chunk.id AS id" in normalized
                and "MATCH (chunk:" in normalized
                and "WHERE chunk.id = $chunk_id" not in normalized
            ):
                return FakeResult([dict(node) for node in self._driver.nodes.values()])

            if "DELETE chunk" in normalized:
                deleted = 1 if self._driver.nodes.pop(params["chunk_id"], None) is not None else 0
                return FakeResult(counters=FakeCounters(nodes_deleted=deleted))

            if "RETURN count(chunk) AS total_chunks" in normalized:
                total = len(self._driver.nodes)
                with_embedding = sum(1 for node in self._driver.nodes.values() if node["embedding"] is not None)
                avg_salience = (
                    sum(float(node["salience"]) for node in self._driver.nodes.values()) / total
                    if total
                    else 0.0
                )
                avg_access = (
                    sum(int(node["access_count"]) for node in self._driver.nodes.values()) / total
                    if total
                    else 0.0
                )
                return FakeResult(
                    [
                        {
                            "total_chunks": total,
                            "chunks_with_embeddings": with_embedding,
                            "average_salience": avg_salience,
                            "average_access_count": avg_access,
                        }
                    ]
                )

            raise AssertionError(f"Unexpected Cypher query: {normalized}")

    class FakeDriver:
        def __init__(self, uri: str, auth: tuple[str, str]) -> None:
            self.uri = uri
            self.auth = auth
            self.nodes: dict[str, dict[str, Any]] = {}

        def verify_connectivity(self) -> None:
            return None

        def session(self, *, database: str | None = None) -> FakeSession:
            return FakeSession(self, database=database)

        def close(self) -> None:
            return None

    class FakeGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str]) -> FakeDriver:
            return FakeDriver(uri, auth)

    fake_module = types.ModuleType("neo4j")
    fake_module.GraphDatabase = FakeGraphDatabase  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "neo4j", fake_module)


def test_neo4j_store_crud_and_retrieve(fake_neo4j: None) -> None:
    store = Neo4jStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="mnemos",
        label="MnemosMemoryChunk",
    )

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

    assert store.touch("a", access_count=4) is True
    touched = store.get("a")
    assert touched is not None
    assert touched.access_count == 4

    assert store.delete("b") is True
    assert store.delete("does-not-exist") is False
    assert len(store.get_all()) == 1

    stats = store.get_stats()
    assert stats["backend"] == "Neo4jStore"
    assert stats["total_chunks"] == 1
