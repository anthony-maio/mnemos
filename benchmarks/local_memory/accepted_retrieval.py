"""Retrieval on declared accepted state, independent of automatic extraction.

The snapshot is a benchmark input, not proof that ingestion produced these facts.
Answer labels are supplied separately and never enter stored records or embeddings.
Only the baseline vector, lexical and hybrid strategies run here; this is not the
full-module ablation or the native-host latency benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import random
import time
from typing import Any

from pydantic import BaseModel, ConfigDict

from mnemos.config import MnemosConfig
from mnemos.contracts import RetrievalRequest
from mnemos.engine import MnemosEngine
from mnemos.jobs import FactEvidence
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeContext, ScopeRegistry, context_metadata, scope_allows
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.embeddings import EmbeddingProvider, embed_text_async
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore

from .retrieval_comparison import comparison_snapshots

STRATEGIES = ("vector", "lexical", "hybrid")
CONTEXT_KEYS = {
    "user_profile_id",
    "project_id",
    "workspace_id",
    "session_id",
    "host",
    "scope",
    "scope_id",
}


class AcceptedSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: str
    as_of: datetime
    registry: ScopeRegistry
    facts: tuple[MemoryChunk, ...]
    sources: tuple[EpisodicEvent, ...]
    evidence: tuple[FactEvidence, ...]


@dataclass(frozen=True)
class PreparedSnapshot:
    service: LocalMemoryService
    fixture_sha256: str
    state_sha256: str
    document_vectors_sha256: str
    dimension: int


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _unique(values: list[str], label: str) -> None:
    if any(not value.strip() for value in values) or len(set(values)) != len(values):
        raise ValueError(f"Blank or duplicate {label}")


def _time(value: datetime, as_of: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None or value > as_of:
        raise ValueError("Snapshot timestamps must be aware and no later than as_of")


def _partition(metadata: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(metadata.get(key) for key in ("user_profile_id", "scope", "scope_id"))


def _validate(snapshot: AcceptedSnapshot) -> AcceptedSnapshot:
    # Pydantic objects are mutable. Reconstruct before validating and embedding.
    data = AcceptedSnapshot.model_validate(snapshot.model_dump(mode="python"))
    if not data.snapshot_id.strip():
        raise ValueError("Snapshot identity is required")
    _time(data.as_of, data.as_of)
    _unique([fact.id for fact in data.facts], "fact IDs")
    _unique([source.id for source in data.sources], "source IDs")
    _unique([source.event_fingerprint for source in data.sources], "source fingerprints")
    _unique([item.evidence_id for item in data.evidence], "evidence IDs")
    canonical = [fact.canonical_key for fact in data.facts if fact.canonical_key is not None]
    _unique(canonical, "canonical keys")
    records: tuple[MemoryChunk | EpisodicEvent, ...] = (*data.facts, *data.sources)
    for item in records:
        allowed = CONTEXT_KEYS | ({"assertion_status"} if isinstance(item, MemoryChunk) else set())
        if set(item.metadata) - allowed:
            raise ValueError("Snapshot metadata contains undeclared fields or evaluation labels")
        context = data.registry.validate_context(ScopeContext.model_validate(item.metadata))
        if any(
            item.metadata.get(key) != value
            for key, value in context_metadata(context, str(item.metadata.get("scope"))).items()
        ):
            raise ValueError("Snapshot context and scope metadata disagree")
        _time(item.created_at, data.as_of)
        if not item.content.strip():
            raise ValueError("Snapshot content cannot be blank")
        if isinstance(item, MemoryChunk):
            if item.metadata.get("assertion_status") not in {"asserted", "conflict"}:
                raise ValueError("Snapshot facts require declared accepted or conflict status")
            if item.embedding is not None:
                raise ValueError("Document embeddings must be computed by the bound provider")
            _time(item.updated_at, data.as_of)
            if item.last_observed_at is not None:
                _time(item.last_observed_at, data.as_of)
            if item.last_retrieved_at is not None:
                _time(item.last_retrieved_at, data.as_of)
        else:
            _time(item.occurred_at, data.as_of)
            if (item.scope, item.scope_id) != (item.metadata["scope"], item.metadata["scope_id"]):
                raise ValueError("Source scope fields disagree")
    facts = {fact.id: fact for fact in data.facts}
    sources = {source.id: source for source in data.sources}
    for link in data.evidence:
        _time(link.created_at, data.as_of)
        fact, source = facts.get(link.fact_id), sources.get(link.source_event_id or "")
        if fact is None or source is None:
            raise ValueError("Every evidence link must resolve to a fact and retained source")
        if not link.excerpt.strip() or link.excerpt not in source.content:
            raise ValueError("Evidence excerpt must be literal source text")
        if _partition(fact.metadata) != _partition(source.metadata):
            raise ValueError("Fact and source evidence must share the same scope partition")
    if set(facts) != {item.fact_id for item in data.evidence} or set(sources) != {
        item.source_event_id for item in data.evidence
    }:
        raise ValueError("Snapshot contains an unsupported fact or an unreferenced source")
    return data


def _vector(values: list[float], dimension: int) -> list[float]:
    if len(values) != dimension or any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("Embedding dimension or numeric values are invalid")
    vector = [float(value) for value in values]
    norm = math.fsum(value * value for value in vector)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding norm must be positive and finite")
    return vector


def _state(service: LocalMemoryService) -> str:
    store = service.engine.store
    facts = sorted(store.get_all(), key=lambda fact: fact.id)
    return _digest(
        {
            "facts": [fact.model_dump(mode="json") for fact in facts],
            "sources": [
                source.model_dump(mode="json")
                for source in sorted(store.list_episodic_events(), key=lambda source: source.id)
            ],
            "evidence": [
                item.model_dump(mode="json")
                for fact in facts
                for item in sorted(
                    store.list_fact_evidence(fact.id), key=lambda item: item.evidence_id
                )
            ],
            "jobs": [job.model_dump(mode="json") for job in store.list_jobs()],
            "registry": service.registry.model_dump(mode="json"),
            "config": service.engine.config.model_dump(mode="json"),
        }
    )


def build_accepted_snapshot(
    path: Path, snapshot: AcceptedSnapshot, embedder: EmbeddingProvider, *, dimension: int
) -> PreparedSnapshot:
    """Build only a new database. No extraction, maintenance or live runtime runs."""
    path = path.resolve()
    if any(
        candidate.exists()
        for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm"))
    ):
        raise FileExistsError("Snapshot destination or SQLite sidecar already exists")
    if not path.parent.is_dir():
        raise ValueError("Create the owned snapshot parent before building")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("Embedding dimension must be positive")
    data = _validate(snapshot)
    fixture_digest = _digest(data.model_dump(mode="json"))
    vectors = {
        fact.id: _vector(embedder.embed_document(fact.content), dimension) for fact in data.facts
    }
    if any(
        candidate.exists() for candidate in (Path(str(path) + "-wal"), Path(str(path) + "-shm"))
    ):
        raise FileExistsError("SQLite sidecar appeared during embedding")
    # Exclusive creation refuses a destination created during provider calls.
    with path.open("xb"):
        pass
    store = SQLiteStore(str(path))
    try:
        for source in data.sources:
            if store.append_episodic_event(source).id != source.id:
                raise ValueError("Snapshot source identity collapsed during persistence")
        for fact in data.facts:
            fact.embedding = vectors[fact.id]
            fact.metadata["source"] = "accepted_snapshot_fixture"
            links = [item for item in data.evidence if item.fact_id == fact.id]
            fact.metadata["source_event_ids"] = [item.source_event_id for item in links]
            fact.metadata["source_roles"] = sorted(
                {
                    source.role
                    for source in data.sources
                    if source.id in fact.metadata["source_event_ids"]
                }
            )
            store.store(fact)
        store._conn.executemany(
            "INSERT INTO fact_evidence (evidence_id,fact_id,source_event_id,excerpt,source_ref,created_at) VALUES (?,?,?,?,?,?)",
            [
                (
                    item.evidence_id,
                    item.fact_id,
                    item.source_event_id,
                    item.excerpt,
                    item.source_ref,
                    item.created_at.isoformat(),
                )
                for item in data.evidence
            ],
        )
        store._conn.commit()
        if (
            store._conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
            or store._conn.execute("PRAGMA foreign_key_check").fetchall()
        ):
            raise RuntimeError("Snapshot integrity check failed")
        config = MnemosConfig()
        for name in type(config.modules).model_fields:
            setattr(config.modules, name, False)
        config.retrieval.profile = "baseline"
        config.retrieval.affective_enabled = False
        config.retrieval.spreading_enabled = False
        config.retrieval.query_state_source = "none"
        service = LocalMemoryService(
            MnemosEngine(
                config=config,
                embedder=embedder,
                llm=UnavailableLLMProvider(),
                store=store,
                scope_registry=data.registry,
                require_scope_context=True,
            ),
            data.registry,
        )
        return PreparedSnapshot(
            service, fixture_digest, _state(service), _digest(vectors), dimension
        )
    except BaseException:
        store.close()
        raise


class _QueryVector(EmbeddingProvider):
    def __init__(self, text: str, vector: list[float]):
        self.text, self.vector = text, vector

    @property
    def dim(self) -> int:
        return len(self.vector)

    def embed(self, text: str) -> list[float]:
        raise RuntimeError("Only the frozen query embedding is available")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Only the frozen query embedding is available")

    def embed_query(self, text: str) -> list[float]:
        if text != self.text:
            raise RuntimeError("Query text changed after embedding")
        return list(self.vector)


async def compare_accepted_query(
    prepared: PreparedSnapshot,
    request: RetrievalRequest,
    *,
    query_id: str,
    expected_fact_ids: list[str],
    repetition: int,
) -> list[dict[str, Any]]:
    """Compare retrieval only. A provider failure never earns abstention credit."""
    request = RetrievalRequest.model_validate(request.model_dump(mode="python"))
    expected_fact_ids = list(expected_fact_ids)
    if (
        not query_id.strip()
        or repetition < 1
        or request.view != "memory"
        or request.response_format != "full"
        or request.top_k <= 0
        or request.max_context_tokens <= 0
    ):
        raise ValueError("Comparison requires a named, positive-budget full memory query")
    _unique(expected_fact_ids, "answer labels")
    service = prepared.service
    service.registry.validate_context(request.context)
    if _state(service) != prepared.state_sha256:
        raise ValueError("Prepared accepted state changed")
    expected = set(expected_fact_ids)
    scoped = {
        fact.id: fact
        for fact in service.engine.store.get_all()
        if scope_allows(fact.metadata, request.context, request.allowed_scopes)
    }
    visible = {
        key for key, fact in scoped.items() if fact.metadata.get("assertion_status") == "asserted"
    }
    if not expected <= visible:
        raise ValueError("Expected answers must identify visible accepted facts")
    started = time.perf_counter()
    vector, embedding_error = None, None
    try:
        vector = _vector(
            await embed_text_async(service.engine.embedder, request.query, role="query"),
            prepared.dimension,
        )
    except Exception as error:
        embedding_error = type(error).__name__
    embedding_ms = (time.perf_counter() - started) * 1000
    if _state(service) != prepared.state_sha256:
        raise ValueError("Accepted state changed during query embedding")
    rows = []
    order = list(STRATEGIES)
    random.Random(f"{prepared.fixture_sha256}:{query_id}:{repetition}").shuffle(order)
    with comparison_snapshots(service) as (digest, variants):
        for strategy in order:
            result, error_type = None, None
            started = time.perf_counter()
            try:
                if strategy != "lexical" and vector is None:
                    error_type = embedding_error
                else:
                    if vector is not None:
                        variants[strategy].engine._embedder = _QueryVector(request.query, vector)
                    result = await variants[strategy].retrieve(request)
                    if result.get("degraded"):
                        error_type = "DegradedRetrieval"
            except Exception as error:
                error_type = type(error).__name__
            retrieval_ms = (time.perf_counter() - started) * 1000
            selected = [fact["id"] for fact in result["facts"]] if result is not None else []
            overlap = expected & set(selected)
            success = result is not None and error_type is None
            rows.append(
                {
                    "query_id": query_id,
                    "repetition": repetition,
                    "strategy": strategy,
                    "fixture_sha256": prepared.fixture_sha256,
                    "state_sha256": prepared.state_sha256,
                    "document_vectors_sha256": prepared.document_vectors_sha256,
                    "sqlite_snapshot_sha256": digest,
                    "query_vector_sha256": (
                        _digest(vector) if strategy != "lexical" and vector is not None else None
                    ),
                    "request": request.model_dump(mode="json"),
                    "expected_fact_ids": list(expected_fact_ids),
                    "selected_ids": selected,
                    "result": result,
                    "error_type": error_type,
                    "query_embedding_ms": embedding_ms if strategy != "lexical" else 0.0,
                    "retrieval_ms": retrieval_ms,
                    "grade": {
                        "operational_success": success,
                        "answerable": bool(expected),
                        "answerable_hit": success and bool(overlap),
                        "correct_abstention": success and not expected and not selected,
                        "relevant_count": len(overlap) if success else 0,
                        "nonrelevant_count": len(set(selected) - expected),
                        "scope_leak_count": len(set(selected) - set(scoped)),
                        "expected_count": len(expected),
                        "precision": (
                            len(overlap) / len(set(selected)) if success and selected else None
                        ),
                        "recall": (
                            len(overlap) / len(expected)
                            if success and expected
                            else 0.0 if expected else None
                        ),
                    },
                }
            )
    if _state(service) != prepared.state_sha256:
        raise ValueError("Source accepted state changed during comparison")
    return rows
