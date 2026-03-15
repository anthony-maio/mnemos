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
        def __init__(
            self, records: list[dict[str, Any]] | None = None, counters: Any = None
        ) -> None:
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
            if normalized.startswith("SHOW CONSTRAINTS"):
                return FakeResult([{"total": 0}])
            if "CREATE CONSTRAINT" in normalized:
                return FakeResult()

            if "MERGE (chunk:" in normalized and "SET chunk.content" in normalized:
                chunk = dict(params)
                self._driver.nodes[chunk["id"]] = chunk
                return FakeResult(counters=FakeCounters(properties_set=1))

            if (
                "OPTIONAL MATCH (source)-[rel]->() WHERE type(rel) = $edge_type DELETE rel"
                in normalized
            ):
                source_id = params["chunk_id"]
                self._driver.edges.pop(source_id, None)
                return FakeResult(counters=FakeCounters())

            if (
                "UNWIND $edges AS edge" in normalized
                and "MERGE (source)-[rel:RELATED_TO]->(target)" in normalized
            ):
                source_id = params["chunk_id"]
                source_edges = self._driver.edges.setdefault(source_id, {})
                for edge in params["edges"]:
                    source_edges[edge["target_id"]] = edge["weight"]
                return FakeResult(counters=FakeCounters(properties_set=len(params["edges"])))

            if "SET chunk.access_count" in normalized:
                node = self._driver.nodes.get(params["chunk_id"])
                if node is None:
                    return FakeResult(counters=FakeCounters(properties_set=0))
                node["access_count"] = params["access_count"]
                node["updated_at"] = params["updated_at"]
                return FakeResult(counters=FakeCounters(properties_set=2))

            if (
                "RETURN properties(chunk)['id'] AS id" in normalized
                and "WHERE chunk.id = $chunk_id" in normalized
            ):
                node = self._driver.nodes.get(params["chunk_id"])
                if node is None:
                    return FakeResult()
                return FakeResult([dict(node)])

            if (
                "RETURN properties(chunk)['id'] AS id" in normalized
                and "MATCH (chunk:" in normalized
                and "WHERE chunk.id = $chunk_id" not in normalized
            ):
                return FakeResult([dict(node) for node in self._driver.nodes.values()])

            if (
                "RETURN source.id AS source_id" in normalized
                and "rel.weight AS weight" in normalized
                and "WHERE type(rel) = $edge_type" in normalized
            ):
                records: list[dict[str, Any]] = []
                allowed_ids = set(params.get("chunk_ids", [])) if "chunk_ids" in params else None
                for source_id, neighbors in self._driver.edges.items():
                    if allowed_ids is not None and source_id not in allowed_ids:
                        continue
                    for target_id, weight in neighbors.items():
                        if allowed_ids is not None and target_id not in allowed_ids:
                            continue
                        records.append(
                            {
                                "source_id": source_id,
                                "target_id": target_id,
                                "weight": weight,
                            }
                        )
                return FakeResult(records)

            if "DELETE chunk" in normalized:
                chunk_id = params["chunk_id"]
                deleted = 1 if self._driver.nodes.pop(chunk_id, None) is not None else 0
                self._driver.edges.pop(chunk_id, None)
                for neighbors in self._driver.edges.values():
                    neighbors.pop(chunk_id, None)
                return FakeResult(counters=FakeCounters(nodes_deleted=deleted))

            if "RETURN count(chunk) AS total_chunks" in normalized:
                total = len(self._driver.nodes)
                with_embedding = sum(
                    1 for node in self._driver.nodes.values() if node["embedding"] is not None
                )
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

            if (
                "RETURN count(rel) AS directed_related_edges" in normalized
                and "WHERE type(rel) = $edge_type" in normalized
            ):
                directed = sum(len(neighbors) for neighbors in self._driver.edges.values())
                return FakeResult([{"directed_related_edges": directed}])

            raise AssertionError(f"Unexpected Cypher query: {normalized}")

    class FakeDriver:
        def __init__(self, uri: str, auth: tuple[str, str]) -> None:
            self.uri = uri
            self.auth = auth
            self.nodes: dict[str, dict[str, Any]] = {}
            self.edges: dict[str, dict[str, float]] = {}

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


@pytest.fixture
def strict_fake_neo4j(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResult:
        def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
            self._records = records or []
            self._closed = False

        def close(self) -> None:
            self._closed = True

        def _ensure_open(self) -> None:
            if self._closed:
                raise RuntimeError("Neo4j result accessed after session close")

        def __iter__(self):
            self._ensure_open()
            return iter(self._records)

        def single(self) -> dict[str, Any] | None:
            self._ensure_open()
            return self._records[0] if self._records else None

        def consume(self) -> None:
            self._ensure_open()
            return None

    class FakeSession:
        def __init__(self) -> None:
            self._results: list[FakeResult] = []

        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            for result in self._results:
                result.close()
            return False

        def run(self, query: str, **params: Any) -> FakeResult:
            normalized = " ".join(query.split())
            if normalized.startswith("SHOW CONSTRAINTS"):
                result = FakeResult([{"total": 0}])
            elif "CREATE CONSTRAINT" in normalized:
                result = FakeResult()
            elif "RETURN count(chunk) AS total_chunks" in normalized:
                result = FakeResult(
                    [
                        {
                            "total_chunks": 0,
                            "chunks_with_embeddings": 0,
                            "average_salience": None,
                            "average_access_count": None,
                        }
                    ]
                )
            elif (
                "RETURN count(rel) AS directed_related_edges" in normalized
                and "WHERE type(rel) = $edge_type" in normalized
            ):
                result = FakeResult([{"directed_related_edges": 0}])
            else:
                raise AssertionError(f"Unexpected Cypher query: {normalized}")
            self._results.append(result)
            return result

    class FakeDriver:
        def verify_connectivity(self) -> None:
            return None

        def session(self, *, database: str | None = None) -> FakeSession:
            return FakeSession()

        def close(self) -> None:
            return None

    class FakeGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str]) -> FakeDriver:
            return FakeDriver()

    fake_module = types.ModuleType("neo4j")
    fake_module.GraphDatabase = FakeGraphDatabase  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "neo4j", fake_module)


@pytest.fixture
def warning_safe_fake_neo4j(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_projection = (
        "RETURN properties(chunk)['id'] AS id, "
        "properties(chunk)['content'] AS content, "
        "properties(chunk)['embedding'] AS embedding, "
        "properties(chunk)['metadata_json'] AS metadata_json, "
        "properties(chunk)['salience'] AS salience, "
        "properties(chunk)['cognitive_state_json'] AS cognitive_state_json, "
        "properties(chunk)['created_at'] AS created_at, "
        "properties(chunk)['updated_at'] AS updated_at, "
        "properties(chunk)['access_count'] AS access_count, "
        "properties(chunk)['version'] AS version"
    )
    expected_stats = (
        "RETURN count(chunk) AS total_chunks, "
        "count(properties(chunk)['embedding']) AS chunks_with_embeddings, "
        "avg(properties(chunk)['salience']) AS average_salience, "
        "avg(properties(chunk)['access_count']) AS average_access_count"
    )

    class FakeResult:
        def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
            self._records = records or []

        def __iter__(self):
            return iter(self._records)

        def single(self) -> dict[str, Any] | None:
            return self._records[0] if self._records else None

        def consume(self) -> None:
            return None

    class FakeSession:
        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def run(self, query: str, **params: Any) -> FakeResult:
            normalized = " ".join(query.split())
            if normalized.startswith("SHOW CONSTRAINTS"):
                return FakeResult([{"total": 0}])
            if "CREATE CONSTRAINT" in normalized:
                return FakeResult()
            if expected_projection in normalized and "WHERE chunk.id = $chunk_id" in normalized:
                return FakeResult(
                    [
                        {
                            "id": params["chunk_id"],
                            "content": "neo4j warning-safe get",
                            "embedding": [1.0, 0.0, 0.0],
                            "metadata_json": "{}",
                            "salience": 0.5,
                            "cognitive_state_json": None,
                            "created_at": "2026-03-12T00:00:00+00:00",
                            "updated_at": "2026-03-12T00:00:00+00:00",
                            "access_count": 0,
                            "version": 1,
                        }
                    ]
                )
            if expected_projection in normalized and "WHERE chunk.id = $chunk_id" not in normalized:
                return FakeResult(
                    [
                        {
                            "id": "all",
                            "content": "neo4j warning-safe list",
                            "embedding": [1.0, 0.0, 0.0],
                            "metadata_json": "{}",
                            "salience": 0.5,
                            "cognitive_state_json": None,
                            "created_at": "2026-03-12T00:00:00+00:00",
                            "updated_at": "2026-03-12T00:00:00+00:00",
                            "access_count": 0,
                            "version": 1,
                        }
                    ]
                )
            if expected_stats in normalized:
                return FakeResult(
                    [
                        {
                            "total_chunks": 1,
                            "chunks_with_embeddings": 1,
                            "average_salience": 0.5,
                            "average_access_count": 0.0,
                        }
                    ]
                )
            if (
                "RETURN count(rel) AS directed_related_edges" in normalized
                and "WHERE type(rel) = $edge_type" in normalized
            ):
                return FakeResult([{"directed_related_edges": 0}])
            raise AssertionError(f"Unexpected Cypher query: {normalized}")

    class FakeDriver:
        def verify_connectivity(self) -> None:
            return None

        def session(self, *, database: str | None = None) -> FakeSession:
            return FakeSession()

        def close(self) -> None:
            return None

    class FakeGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str]) -> FakeDriver:
            return FakeDriver()

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


def test_neo4j_store_persists_related_edges(fake_neo4j: None) -> None:
    store = Neo4jStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="mnemos",
        label="MnemosMemoryChunk",
    )

    chunk_a = MemoryChunk(
        id="a",
        content="alpha memory",
        embedding=[1.0, 0.0, 0.0],
        metadata={"scope": "project", "scope_id": "repo-alpha"},
    )
    chunk_b = MemoryChunk(
        id="b",
        content="global memory",
        embedding=[0.9, 0.1, 0.0],
        metadata={"scope": "global"},
    )
    store.store(chunk_a)
    store.store(chunk_b)

    store.upsert_graph_edge("a", "b", 0.91)

    assert store.get_graph_edges(["a", "b"]) == {
        "a": {"b": pytest.approx(0.91)},
        "b": {"a": pytest.approx(0.91)},
    }

    stats = store.get_stats()
    assert stats["related_edges"] == 1


def test_neo4j_store_materializes_results_before_session_close(strict_fake_neo4j: None) -> None:
    store = Neo4jStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="mnemos",
        label="MnemosMemoryChunk",
    )

    stats = store.get_stats()

    assert stats["total_chunks"] == 0


def test_neo4j_store_uses_warning_safe_property_access(
    warning_safe_fake_neo4j: None,
) -> None:
    store = Neo4jStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="mnemos",
        label="MnemosMemoryChunk",
    )

    fetched = store.get("chunk-1")
    all_chunks = store.get_all()
    stats = store.get_stats()

    assert fetched is not None
    assert fetched.content == "neo4j warning-safe get"
    assert len(all_chunks) == 1
    assert stats["chunks_with_embeddings"] == 1


def test_neo4j_store_uses_label_specific_constraint_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_queries: list[str] = []

    class FakeResult:
        def __iter__(self):
            return iter([])

        def single(self) -> None:
            return None

        def consume(self) -> None:
            return None

    class FakeSession:
        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def run(self, query: str, **params: Any) -> FakeResult:
            seen_queries.append(" ".join(query.split()))
            return FakeResult()

    class FakeDriver:
        def verify_connectivity(self) -> None:
            return None

        def session(self, *, database: str | None = None) -> FakeSession:
            return FakeSession()

        def close(self) -> None:
            return None

    class FakeGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str]) -> FakeDriver:
            return FakeDriver()

    fake_module = types.ModuleType("neo4j")
    fake_module.GraphDatabase = FakeGraphDatabase  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "neo4j", fake_module)

    Neo4jStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="mnemos",
        label="MnemosMigrationSmoke",
    )

    assert seen_queries
    assert any(
        "CREATE CONSTRAINT mnemos_memory_chunk_id_mnemosmigrationsmoke" in query
        for query in seen_queries
    )


def test_neo4j_store_preserves_legacy_constraint_name_for_default_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_queries: list[str] = []

    class FakeResult:
        def __iter__(self):
            return iter([])

        def single(self) -> None:
            return None

        def consume(self) -> None:
            return None

    class FakeSession:
        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def run(self, query: str, **params: Any) -> FakeResult:
            seen_queries.append(" ".join(query.split()))
            return FakeResult()

    class FakeDriver:
        def verify_connectivity(self) -> None:
            return None

        def session(self, *, database: str | None = None) -> FakeSession:
            return FakeSession()

        def close(self) -> None:
            return None

    class FakeGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str]) -> FakeDriver:
            return FakeDriver()

    fake_module = types.ModuleType("neo4j")
    fake_module.GraphDatabase = FakeGraphDatabase  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "neo4j", fake_module)

    Neo4jStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="mnemos",
        label="MnemosMemoryChunk",
    )

    assert seen_queries
    assert any(
        "CREATE CONSTRAINT mnemos_memory_chunk_id IF NOT EXISTS" in query for query in seen_queries
    )


def test_neo4j_store_skips_constraint_create_when_schema_already_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_queries: list[str] = []

    class FakeResult:
        def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
            self._records = records or []

        def __iter__(self):
            return iter(self._records)

        def single(self) -> dict[str, Any] | None:
            return self._records[0] if self._records else None

        def consume(self) -> None:
            return None

    class FakeSession:
        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def run(self, query: str, **params: Any) -> FakeResult:
            normalized = " ".join(query.split())
            seen_queries.append(normalized)
            if normalized.startswith("SHOW CONSTRAINTS"):
                return FakeResult([{"total": 1}])
            if normalized.startswith("CREATE CONSTRAINT"):
                return FakeResult()
            return FakeResult()

    class FakeDriver:
        def verify_connectivity(self) -> None:
            return None

        def session(self, *, database: str | None = None) -> FakeSession:
            return FakeSession()

        def close(self) -> None:
            return None

    class FakeGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str]) -> FakeDriver:
            return FakeDriver()

    fake_module = types.ModuleType("neo4j")
    fake_module.GraphDatabase = FakeGraphDatabase  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "neo4j", fake_module)

    Neo4jStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="mnemos",
        label="MnemosMemoryChunk",
    )

    assert any(query.startswith("SHOW CONSTRAINTS") for query in seen_queries)
    assert not any(query.startswith("CREATE CONSTRAINT") for query in seen_queries)
