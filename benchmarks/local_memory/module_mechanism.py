"""Offline mechanism evaluation for the graph and affective modules.

The fixture is deliberately explicit.  Embeddings in the fixture are frozen
inputs, so repeated runs do not call an embedding model or silently move the
candidate pool. A real-model runner can provide a provider when preparing a
fixture. Classifier calls occur only when an explicit LLM provider is passed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from mnemos.config import MnemosConfig, ModuleConfig, SpreadingConfig
from mnemos.engine import MnemosEngine
from mnemos.modules.affective import AffectiveRouter
from mnemos.retrieval import RetrievalConfig, score_candidate
from mnemos.scopes import ProjectIdentity, ScopeContext, ScopeRegistry, WorkspaceIdentity
from mnemos.types import CognitiveState, Interaction, MemoryChunk
from mnemos.jobs import FactEvidence
from mnemos.utils.embeddings import EmbeddingProvider, cosine_similarity, embed_batch_for_role
from mnemos.utils.llm import LLMProvider
from mnemos.utils.storage import InMemoryStore, SQLiteStore
from .scoring import paired_bootstrap_interval, wilson_interval


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


SCHEMA_VERSION = "local-memory-mechanism-v1"
MIN_REPETITIONS = 3
GRAPH_FACT_COUNT = 64
EXECUTION_SEED = 20260916


def _alias(value: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in value:
            return value[name]
    return default


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _state(value: Any, *, allow_none: bool = True) -> CognitiveState | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("state must be an object or null")
    return CognitiveState.model_validate(value)


def _vector(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty embedding list")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains a non-numeric value") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


@dataclass(frozen=True)
class FactSpec:
    id: str
    content: str
    embedding: list[float] | None
    evidence_id: str
    scope: str
    user_profile_id: str
    project_id: str | None = None
    workspace_id: str | None = None
    state: CognitiveState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: Mapping[str, Any], *, fallback_context: Mapping[str, Any]) -> FactSpec:
        fact_id = _require_text(_alias(raw, "id", "fact_id"), "fact id")
        content = _require_text(_alias(raw, "content"), f"fact {fact_id} content")
        raw_embedding = _alias(raw, "embedding")
        embedding = (
            _vector(raw_embedding, f"fact {fact_id} embedding")
            if raw_embedding is not None
            else None
        )
        evidence_id = _require_text(
            _alias(raw, "evidence_id", "source_event_id"),
            f"fact {fact_id} evidence_id",
        )
        context = {**fallback_context, **dict(raw.get("context", {}))}
        scope = str(raw.get("scope", "project")).strip().lower()
        if scope not in {"project", "workspace", "global"}:
            raise ValueError(f"fact {fact_id} has invalid scope {scope!r}")
        profile = _require_text(
            raw.get("user_profile_id", context.get("user_profile_id")),
            f"fact {fact_id} user_profile_id",
        )
        project = raw.get("project_id", context.get("project_id"))
        workspace = raw.get("workspace_id", context.get("workspace_id"))
        if scope == "project" and not project:
            raise ValueError(f"fact {fact_id} project scope needs project_id")
        if scope == "workspace" and not workspace:
            raise ValueError(f"fact {fact_id} workspace scope needs workspace_id")
        supplied_scope_id = raw.get("scope_id")
        expected_scope_id = (
            None if scope == "global" else (project if scope == "project" else workspace)
        )
        if supplied_scope_id is not None and str(supplied_scope_id) != str(expected_scope_id):
            raise ValueError(f"fact {fact_id} scope_id does not match its scope identity")
        metadata = dict(raw.get("metadata", {}))
        metadata.update(
            {
                "scope": scope,
                "user_profile_id": profile,
                "scope_id": (
                    None if scope == "global" else (project if scope == "project" else workspace)
                ),
            }
        )
        if project:
            metadata["project_id"] = project
        if workspace:
            metadata["workspace_id"] = workspace
        return cls(
            id=fact_id,
            content=content,
            embedding=embedding,
            evidence_id=evidence_id,
            scope=scope,
            user_profile_id=profile,
            project_id=str(project) if project is not None else None,
            workspace_id=str(workspace) if workspace is not None else None,
            state=_state(_alias(raw, "state", "cognitive_state")),
            metadata=metadata,
        )

    def chunk(self) -> MemoryChunk:
        if self.embedding is None:
            raise ValueError(f"fact {self.id} has no frozen embedding")
        return MemoryChunk(
            id=self.id,
            content=self.content,
            embedding=list(self.embedding),
            metadata={**self.metadata, "evidence_id": self.evidence_id},
            cognitive_state=self.state,
            cognitive_state_status="frozen_fixture" if self.state is not None else "unknown",
        )


@dataclass(frozen=True)
class GraphCase:
    id: str
    split: str
    template_id: str | None
    query: str
    query_embedding: list[float] | None
    top_k: int
    target_id: str
    context: dict[str, str | None]
    facts: tuple[FactSpec, ...]
    scope_controls: tuple[FactSpec, ...] = ()
    seed_id: str | None = None
    bridge_id: str | None = None
    graph_expectations: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AffectiveCondition:
    query: str
    query_embedding: list[float] | None
    state: CognitiveState
    target_id: str


@dataclass(frozen=True)
class AffectivePair:
    id: str
    split: str
    template_id: str | None
    context: dict[str, str | None]
    semantic_query: str
    query_embedding: list[float] | None
    facts: tuple[FactSpec, ...]
    conditions: tuple[AffectiveCondition, ...]
    no_state_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MechanismDataset:
    manifest: Mapping[str, Any]
    graph_cases: tuple[GraphCase, ...]
    affective_pairs: tuple[AffectivePair, ...]
    digest: str


def _context(raw: Mapping[str, Any]) -> dict[str, str | None]:
    value = raw.get("context", raw)
    if not isinstance(value, Mapping):
        raise ValueError("context must be an object")
    profile = _require_text(value.get("user_profile_id"), "context user_profile_id")
    project = value.get("project_id")
    workspace = value.get("workspace_id")
    return {
        "user_profile_id": profile,
        "project_id": str(project) if project is not None else None,
        "workspace_id": str(workspace) if workspace is not None else None,
    }


def _validate_split(value: Any) -> str:
    split = str(value or "heldout").strip().lower()
    if split not in {"development", "heldout"}:
        raise ValueError("split must be development or heldout")
    return split


def _validate_primary_context(facts: Sequence[FactSpec], context: Mapping[str, str | None]) -> None:
    for fact in facts:
        if (
            fact.user_profile_id != context["user_profile_id"]
            or fact.project_id != context.get("project_id")
            or fact.workspace_id != context.get("workspace_id")
        ):
            raise ValueError(
                "primary facts must share the case context; use scope_controls for controls"
            )


def _parse_graph_case(raw: Mapping[str, Any]) -> GraphCase:
    context = _context(raw)
    query_raw = raw.get("query")
    if isinstance(query_raw, str):
        query = _require_text(query_raw, "graph query text")
        query_raw = {"text": query}
    elif isinstance(query_raw, Mapping):
        query = _require_text(_alias(query_raw, "text", "query"), "graph query text")
    else:
        raise ValueError("graph query must be a string or object")
    raw_query_embedding = _alias(query_raw, "embedding")
    query_embedding = (
        _vector(raw_query_embedding, "graph query embedding")
        if raw_query_embedding is not None
        else None
    )
    target_id = _require_text(
        _alias(
            raw,
            "target_id",
            "target_fact_id",
            default=_alias(query_raw, "target_id", "target_fact_id"),
        ),
        "graph target_id",
    )
    raw_facts = raw.get("facts")
    if not isinstance(raw_facts, list) or len(raw_facts) != GRAPH_FACT_COUNT:
        raise ValueError(f"graph case must contain exactly {GRAPH_FACT_COUNT} same-context facts")
    facts = tuple(FactSpec.from_json(item, fallback_context=context) for item in raw_facts)
    _validate_primary_context(facts, context)
    if len({fact.id for fact in facts}) != len(facts):
        raise ValueError("graph facts must have unique IDs")
    if len({fact.evidence_id for fact in facts}) != len(facts):
        raise ValueError("graph facts must have unique evidence IDs")
    if target_id not in {fact.id for fact in facts}:
        raise ValueError("graph target_id is absent from same-context facts")
    seed_id = raw.get("seed_id")
    bridge_id = raw.get("bridge_id")
    known_fact_ids = {fact.id for fact in facts}
    if seed_id is not None and str(seed_id) not in known_fact_ids:
        raise ValueError("graph seed_id is absent from same-context facts")
    if bridge_id is not None and str(bridge_id) not in known_fact_ids:
        raise ValueError("graph bridge_id is absent from same-context facts")
    if len({str(item) for item in (seed_id, bridge_id, target_id) if item is not None}) != len(
        [item for item in (seed_id, bridge_id, target_id) if item is not None]
    ):
        raise ValueError("graph seed, bridge, and target labels must be distinct")
    answer = raw.get("answer")
    if answer is not None:
        answer_text = _require_text(answer, "graph answer")
        target_fact = next(fact for fact in facts if fact.id == target_id)
        if answer_text.lower() not in target_fact.content.lower():
            raise ValueError("graph answer must be present in the target fact")
        if answer_text.lower() in query.lower():
            raise ValueError("graph query must not copy the declared answer")
    controls_raw = raw.get("scope_controls", [])
    if not isinstance(controls_raw, list):
        raise ValueError("scope_controls must be a list")
    controls = tuple(FactSpec.from_json(item, fallback_context=context) for item in controls_raw)
    return GraphCase(
        id=_require_text(_alias(raw, "id", "case_id"), "graph case id"),
        split=_validate_split(raw.get("split", "heldout")),
        template_id=(str(raw["template_id"]) if raw.get("template_id") is not None else None),
        query=query,
        query_embedding=query_embedding,
        top_k=int(query_raw.get("top_k", 5)),
        target_id=target_id,
        context=context,
        facts=facts,
        scope_controls=controls,
        seed_id=(str(raw["seed_id"]) if raw.get("seed_id") is not None else None),
        bridge_id=(str(raw["bridge_id"]) if raw.get("bridge_id") is not None else None),
        graph_expectations=dict(raw.get("graph_expectations", {})),
    )


def _parse_affective_pair(raw: Mapping[str, Any]) -> AffectivePair:
    context = _context(raw)
    query_raw = raw.get("semantic_query", raw.get("query"))
    if isinstance(query_raw, str):
        semantic_query = _require_text(query_raw, "semantic query")
        query_raw = {"text": semantic_query}
    elif isinstance(query_raw, Mapping):
        semantic_query = _require_text(
            _alias(query_raw, "text", "semantic_text", "query"), "semantic query"
        )
    else:
        raise ValueError("affective semantic_query must be a string or object")
    raw_query_embedding = _alias(query_raw, "embedding")
    query_embedding = (
        _vector(raw_query_embedding, "affective query embedding")
        if raw_query_embedding is not None
        else None
    )
    raw_facts = raw.get("facts", raw.get("candidates"))
    if not isinstance(raw_facts, list) or not raw_facts:
        raise ValueError("affective pair needs candidates")
    facts = tuple(FactSpec.from_json(item, fallback_context=context) for item in raw_facts)
    _validate_primary_context(facts, context)
    if len({fact.id for fact in facts}) != len(facts):
        raise ValueError("affective facts must have unique IDs")
    if len({fact.evidence_id for fact in facts}) != len(facts):
        raise ValueError("affective facts must have unique evidence IDs")
    conditions_raw = raw.get("conditions", [])
    if not isinstance(conditions_raw, list) or not conditions_raw:
        raise ValueError("affective pair needs at least one condition")
    conditions: list[AffectiveCondition] = []
    for condition in conditions_raw:
        if not isinstance(condition, Mapping):
            raise ValueError("affective condition must be an object")
        cond_query = _require_text(_alias(condition, "query", "text"), "affective condition query")
        raw_condition_embedding = _alias(condition, "embedding")
        if raw_condition_embedding is not None and query_embedding is None:
            raise ValueError("affective conditions require the pair semantic embedding")
        if raw_condition_embedding is not None and query_embedding is not None:
            condition_vector = _vector(raw_condition_embedding, "affective condition embedding")
            if condition_vector != query_embedding:
                raise ValueError("affective conditions must reuse the pair semantic embedding")
        # Condition wording is for classifier fidelity. Ranking uses the pair
        # semantic vector, so no separately embedded condition is retained.
        cond_embedding = None
        cond_state = _state(condition.get("state"), allow_none=False)
        if cond_state is None:
            raise ValueError("affective condition state cannot be missing")
        target_id = _require_text(
            _alias(condition, "target_id", "target_fact_id"), "affective target_id"
        )
        if target_id not in {fact.id for fact in facts}:
            raise ValueError(f"affective target {target_id} is absent")
        conditions.append(AffectiveCondition(cond_query, cond_embedding, cond_state, target_id))
    if len({condition.target_id for condition in conditions}) != len(conditions):
        raise ValueError("affective conditions must have distinct target IDs")
    null_state_ids = tuple(sorted(fact.id for fact in facts if fact.state is None))
    supplied_present = "no_state_fact_ids" in raw
    supplied_value = raw.get("no_state_fact_ids", [])
    if not isinstance(supplied_value, list):
        raise ValueError("no_state_fact_ids must be a list")
    supplied_no_state = tuple(str(item) for item in supplied_value)
    if len(set(supplied_no_state)) != len(supplied_no_state):
        raise ValueError("no_state_fact_ids must be unique")
    if supplied_present and set(supplied_no_state) != set(null_state_ids):
        raise ValueError("no_state_fact_ids does not match null candidate states")
    return AffectivePair(
        id=_require_text(_alias(raw, "id", "pair_id"), "affective pair id"),
        split=_validate_split(raw.get("split", "heldout")),
        template_id=(str(raw["template_id"]) if raw.get("template_id") is not None else None),
        context=context,
        semantic_query=semantic_query,
        query_embedding=query_embedding,
        facts=facts,
        conditions=tuple(conditions),
        no_state_ids=null_state_ids,
    )


def load_mechanism_dataset(path: Path) -> MechanismDataset:
    """Load and validate the frozen object fixture shape."""
    payload = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    try:
        root = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("mechanism dataset must be one JSON object") from exc
    else:
        if not isinstance(root, Mapping):
            raise ValueError("mechanism dataset root must be an object")
        manifest = root.get("manifest", {})
        graph_rows = root.get("graph_cases", [])
        affective_rows = root.get("affective_pairs", [])
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be an object")
    version = str(
        root.get("schema_version", root.get("version", manifest.get("schema_version", "")))
    )
    if version not in {SCHEMA_VERSION, "module-mechanism-v1"}:
        raise ValueError(f"unsupported mechanism schema {version!r}")
    if not isinstance(graph_rows, list) or not isinstance(affective_rows, list):
        raise ValueError("graph_cases and affective_pairs must be lists")
    graph_cases = tuple(_parse_graph_case(row) for row in graph_rows)
    pairs = tuple(_parse_affective_pair(row) for row in affective_rows)
    if len({case.id for case in graph_cases}) != len(graph_cases):
        raise ValueError("graph case IDs must be unique")
    if len({pair.id for pair in pairs}) != len(pairs):
        raise ValueError("affective pair IDs must be unique")
    expected_graph = manifest.get("graph_cases")
    if expected_graph is not None and int(expected_graph) != len(graph_cases):
        raise ValueError("manifest graph_cases count does not match rows")
    expected_pairs = manifest.get("affective_pairs")
    if expected_pairs is not None and int(expected_pairs) != len(pairs):
        raise ValueError("manifest affective_pairs count does not match rows")
    return MechanismDataset(manifest, graph_cases, pairs, digest)


def prepare_frozen_embeddings(
    source: Path,
    destination: Path,
    provider: EmbeddingProvider,
) -> Path:
    """Materialize one embedding generation into a new fixture file.

    This is the only helper in this module that calls an embedding provider.
    It refuses to replace an existing destination, making the generated
    vectors and their digest an auditable input to later repetitions.
    """
    if destination.exists():
        raise FileExistsError(f"refusing to replace frozen fixture: {destination}")
    root = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise ValueError("embedding preparation requires the object fixture shape")
    document_texts: list[str] = []
    query_texts: list[str] = []

    def add_fact(fact: dict[str, Any]) -> None:
        if "embedding" not in fact:
            document_texts.append(_require_text(fact.get("content"), "fact content"))

    for case in root.get("graph_cases", []):
        for fact in [*case.get("facts", []), *case.get("scope_controls", [])]:
            add_fact(fact)
        query = case.get("query")
        if isinstance(query, str):
            query_texts.append(query)
        elif isinstance(query, dict) and "embedding" not in query:
            query_texts.append(_require_text(query.get("text", query.get("query")), "graph query"))
    for pair in root.get("affective_pairs", []):
        for fact in pair.get("facts", pair.get("candidates", [])):
            add_fact(fact)
        query = pair.get("semantic_query", pair.get("query"))
        if isinstance(query, str):
            query_texts.append(query)
        elif isinstance(query, dict) and "embedding" not in query:
            query_texts.append(
                _require_text(query.get("text", query.get("query")), "semantic query")
            )
    unique_documents = list(dict.fromkeys(document_texts))
    unique_queries = list(dict.fromkeys(query_texts))
    document_vectors = dict(
        zip(
            unique_documents,
            embed_batch_for_role(provider, unique_documents, role="document"),
        )
    )
    query_vectors = dict(
        zip(unique_queries, embed_batch_for_role(provider, unique_queries, role="query"))
    )

    def fill_fact(fact: dict[str, Any]) -> None:
        if "embedding" not in fact:
            fact["embedding"] = document_vectors[_require_text(fact.get("content"), "fact content")]

    for case in root.get("graph_cases", []):
        for fact in [*case.get("facts", []), *case.get("scope_controls", [])]:
            fill_fact(fact)
        query = case.get("query")
        if isinstance(query, str):
            case["query"] = {"text": query, "embedding": query_vectors[query]}
        elif isinstance(query, dict) and "embedding" not in query:
            query["embedding"] = query_vectors[
                _require_text(query.get("text", query.get("query")), "graph query")
            ]
    for pair in root.get("affective_pairs", []):
        for fact in pair.get("facts", pair.get("candidates", [])):
            fill_fact(fact)
        query = pair.get("semantic_query", pair.get("query"))
        if isinstance(query, str):
            pair["semantic_query"] = {"text": query, "embedding": query_vectors[query]}
        elif isinstance(query, dict) and "embedding" not in query:
            query["embedding"] = query_vectors[
                _require_text(query.get("text", query.get("query")), "semantic query")
            ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(root, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return destination


class _FrozenEmbedder(EmbeddingProvider):
    def __init__(
        self,
        vectors: Mapping[str, Sequence[float]],
        *,
        query_vectors: Mapping[str, Sequence[float]] | None = None,
        document_vectors: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self.vectors = {str(key): list(map(float, value)) for key, value in vectors.items()}
        self.query_vectors = {
            str(key): list(map(float, value)) for key, value in (query_vectors or {}).items()
        }
        self.document_vectors = {
            str(key): list(map(float, value)) for key, value in (document_vectors or {}).items()
        }
        self._roleful = bool(self.query_vectors or self.document_vectors)
        all_vectors = {**self.vectors, **self.query_vectors, **self.document_vectors}
        self._dim = len(next(iter(all_vectors.values()))) if all_vectors else 1

    def embed(self, text: str) -> list[float]:
        vector = self.vectors.get(text)
        if vector is None:
            raise ValueError(f"query embedding missing from frozen fixture: {text!r}")
        return list(vector)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = self.query_vectors.get(text) or self.vectors.get(text)
        if vector is None:
            raise ValueError(f"query embedding missing from frozen fixture: {text!r}")
        return list(vector)

    def embed_document(self, text: str) -> list[float]:
        vector = self.document_vectors.get(text) or self.vectors.get(text)
        if vector is None:
            raise ValueError(f"document embedding missing from frozen fixture: {text!r}")
        return list(vector)

    def embed_symmetric(self, text: str) -> list[float]:
        if self._roleful and text not in self.vectors:
            raise ValueError(f"symmetric embedding missing from frozen fixture: {text!r}")
        return self.embed(text)

    def embed_batch_queries(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_batch_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_document(text) for text in texts]

    def embed_batch_symmetric(self, texts: list[str]) -> list[list[float]]:
        return self.embed_batch(texts)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def should_offload_in_async(self) -> bool:
        return False


class _GraphStore(InMemoryStore):
    """Small graph-capable store used to exercise the engine refresh path."""

    def __init__(self) -> None:
        super().__init__(name="module-mechanism")
        self._graph: dict[str, dict[str, float]] = {}
        self.graph_contract: dict[str, str] = {}

    def bind_graph_contract(self, contract: Mapping[str, str]) -> None:
        self.graph_contract = dict(contract)

    def replace_graph_neighbors(self, chunk_id: str, neighbors: dict[str, float]) -> None:
        self._graph[chunk_id] = {str(key): float(value) for key, value in neighbors.items()}

    def get_graph_neighbors(
        self, chunk_id: str, filter_fn: Any = None, limit: int = 32
    ) -> list[tuple[MemoryChunk, float]]:
        rows: list[tuple[MemoryChunk, float]] = []
        for neighbor_id, weight in sorted(
            self._graph.get(chunk_id, {}).items(), key=lambda item: (-item[1], item[0])
        ):
            chunk = self.get(neighbor_id)
            if chunk is not None and (filter_fn is None or filter_fn(chunk)):
                rows.append((chunk, weight))
        return rows[: max(0, limit)]

    def get_graph_edges(self, chunk_ids: list[str] | None = None) -> dict[str, dict[str, float]]:
        if chunk_ids is None:
            return {key: dict(value) for key, value in self._graph.items()}
        selected = set(chunk_ids)
        return {
            key: {neighbor: weight for neighbor, weight in value.items() if neighbor in selected}
            for key, value in self._graph.items()
            if key in selected
        }

    def retrieve_lexical(
        self, query: str, top_k: int = 20, filter_fn: Any = None
    ) -> list[MemoryChunk]:
        tokens = {token.lower() for token in query.split() if token.strip()}
        candidates = [chunk for chunk in self.get_all() if filter_fn is None or filter_fn(chunk)]
        scored = [
            (sum(token in chunk.content.lower().split() for token in tokens), chunk)
            for chunk in candidates
        ]
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [chunk for score, chunk in scored if score > 0][:top_k]


def _context_model(case_context: Mapping[str, str | None]) -> ScopeContext:
    return ScopeContext(
        user_profile_id=str(case_context["user_profile_id"]),
        project_id=case_context.get("project_id"),
        workspace_id=case_context.get("workspace_id"),
        session_id="module-mechanism",
        host="manual",
    )


def _scope_filter(
    context: ScopeContext, allowed: Sequence[str] = ("project", "workspace", "global")
) -> Any:
    def allowed_chunk(chunk: MemoryChunk) -> bool:
        metadata = chunk.metadata
        if metadata.get("user_profile_id") != context.user_profile_id:
            return False
        scope = metadata.get("scope")
        if scope not in allowed:
            return False
        if scope == "global":
            return True
        wanted = context.project_id if scope == "project" else context.workspace_id
        return wanted is not None and metadata.get("scope_id") == wanted

    return allowed_chunk


def _run(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("module evaluation must run outside an active event loop")


def _engine_for(
    facts: Iterable[FactSpec],
    query_text: str,
    query_embedding: list[float],
    *,
    graph: bool,
    affective: bool,
    store_path: Path | None = None,
) -> tuple[MnemosEngine, Any]:
    materialized = list(facts)
    if any(fact.embedding is None for fact in materialized):
        raise ValueError("evaluation requires frozen embeddings for every fact")
    document_vectors = {
        fact.content: fact.embedding for fact in materialized if fact.embedding is not None
    }
    embedder = _FrozenEmbedder(
        document_vectors,
        query_vectors={query_text: query_embedding},
        document_vectors=document_vectors,
    )
    config = MnemosConfig(
        modules=ModuleConfig(affective=affective, spreading=graph),
        retrieval=RetrievalConfig(
            profile="biomimetic" if graph or affective else "baseline",
            affective_enabled=affective,
            spreading_enabled=graph,
            relevance_floor=0.2,
            graph_weight=0.1,
            max_candidates=100,
        ),
        spreading=SpreadingConfig(max_hops=3, max_visited=1000, max_neighbors_per_node=4),
    )
    store: Any = SQLiteStore(str(store_path)) if store_path is not None else _GraphStore()
    projects: dict[str, ProjectIdentity] = {}
    workspaces: dict[str, WorkspaceIdentity] = {}
    profiles: set[str] = set()
    for fact in materialized:
        profiles.add(fact.user_profile_id)
        if fact.project_id is not None:
            project = projects.setdefault(
                fact.project_id,
                ProjectIdentity(
                    id=fact.project_id,
                    root=str(Path.cwd()),
                    user_profile_id=fact.user_profile_id,
                    enabled=True,
                ),
            )
            if fact.workspace_id is not None and fact.workspace_id not in project.workspace_ids:
                project.workspace_ids.append(fact.workspace_id)
        if fact.workspace_id is not None:
            workspaces.setdefault(
                fact.workspace_id,
                WorkspaceIdentity(
                    id=fact.workspace_id,
                    name=fact.workspace_id,
                    user_profile_id=fact.user_profile_id,
                ),
            )
    registry = ScopeRegistry(projects=projects, workspaces=workspaces, profiles=sorted(profiles))
    return (
        MnemosEngine(config=config, embedder=embedder, store=store, scope_registry=registry),
        store,
    )


def _insert_and_refresh(engine: MnemosEngine, store: Any, facts: Iterable[FactSpec]) -> None:
    chunks = [fact.chunk() for fact in facts]
    for chunk in chunks:
        store.upsert_canonical(chunk)
        if hasattr(store, "_conn"):
            evidence = FactEvidence(
                evidence_id=str(chunk.metadata["evidence_id"]),
                fact_id=chunk.id,
                source_event_id=str(chunk.metadata["evidence_id"]),
                excerpt=chunk.content,
            )
            store._conn.execute(
                "INSERT OR IGNORE INTO fact_evidence "
                "(evidence_id, fact_id, source_event_id, excerpt, source_ref, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    evidence.evidence_id,
                    evidence.fact_id,
                    evidence.source_event_id,
                    evidence.excerpt,
                    None,
                    evidence.created_at.isoformat(),
                ),
            )
            store._conn.commit()
        engine.spreading_activation.add_node_from_chunk(chunk)
    for chunk in chunks:
        engine.refresh_graph_chunk(chunk)


def _direct_pool(
    store: Any, query: str, embedding: list[float], predicate: Any, *, candidate_limit: int = 30
) -> tuple[list[str], list[str], list[str]]:
    vector = store.retrieve(embedding, top_k=candidate_limit, filter_fn=predicate)
    lexical = store.retrieve_lexical(query, top_k=candidate_limit, filter_fn=predicate)
    vector_ids = [item.id for item in vector]
    lexical_ids = [item.id for item in lexical]
    union = sorted(set(vector_ids) | set(lexical_ids))
    return vector_ids, lexical_ids, union


def _actual_seed(
    embedding: list[float], ids: Sequence[str], nodes: Mapping[str, Any]
) -> str | None:
    candidates = [node_id for node_id in ids if node_id in nodes]
    return (
        min(
            candidates,
            key=lambda node_id: (
                -cosine_similarity(embedding, nodes[node_id].embedding or []),
                node_id,
            ),
        )
        if candidates
        else None
    )


def _graph_paths(
    edges: Mapping[str, Mapping[str, float]], seeds: set[str], target: str, *, max_hops: int = 3
) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []

    def visit(node: str, path: list[str], energy: float) -> None:
        if node == target:
            paths.append({"nodes": list(path), "energy": energy, "hops": len(path) - 1})
            return
        if len(path) - 1 >= max_hops:
            return
        for neighbor, weight in sorted(
            edges.get(node, {}).items(), key=lambda item: (-float(item[1]), item[0])
        ):
            if neighbor in path:
                continue
            transfer = energy * 0.8 * float(weight)
            if transfer < 0.3:
                continue
            visit(neighbor, [*path, neighbor], transfer)

    for seed in sorted(seeds):
        visit(seed, [seed], 1.0)
    paths.sort(key=lambda row: (-row["energy"], row["nodes"]))
    return paths


def _target_cutset(
    edges: Mapping[str, Mapping[str, float]], seeds: set[str], target: str
) -> list[tuple[str, str]]:
    paths = _graph_paths(edges, seeds, target)
    if not paths:
        return []
    return sorted({(path["nodes"][-2], target) for path in paths if len(path["nodes"]) > 1})


def _remove_edges(
    edges: Mapping[str, Mapping[str, float]], cutset: Iterable[tuple[str, str]]
) -> dict[str, dict[str, float]]:
    result = {source: dict(neighbors) for source, neighbors in edges.items()}
    for source, target in cutset:
        result.get(source, {}).pop(target, None)
        result.get(target, {}).pop(source, None)
    return result


def evaluate_graph_case(
    case: GraphCase,
    *,
    repetitions: int = 3,
    work_dir: Path | None = None,
    persistent_work_dir: Path | None = None,
    on_row: Callable[[dict[str, Any]], None] | None = None,
    repetition_indices: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """Run all fixed repetitions through the delegated graph measurement helper."""
    if repetitions < MIN_REPETITIONS:
        raise ValueError("mechanism evaluation requires at least three repetitions")
    try:
        from .graph_measurement import measure_graph_repetition
    except ImportError as exc:
        raise RuntimeError("graph_measurement helper is required for persistent runs") from exc
    target_dir = persistent_work_dir or work_dir or Path.cwd() / ".module-mechanism-work"
    target_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    indices = (
        list(range(repetitions)) if repetition_indices is None else sorted(set(repetition_indices))
    )
    if any(index < 0 or index >= repetitions for index in indices):
        raise ValueError("graph repetition index is outside the requested range")
    for repetition in indices:
        row = measure_graph_repetition(
            case,
            work_dir=target_dir,
            repetition=repetition,
        )
        normalized_row = dict(row)
        query_data: Mapping[str, Any] = (
            dict(row["query"]) if isinstance(row.get("query"), Mapping) else {}
        )
        production: Mapping[str, Any] = (
            dict(row["production"]) if isinstance(row.get("production"), Mapping) else {}
        )
        for target, source in (
            ("vector_ids", "vector_ids"),
            ("lexical_ids", "lexical_ids"),
            ("direct_ids", "direct_union_ids"),
            ("actual_selected_seed_id", "actual_selected_seed_id"),
            ("target_id", "target_id"),
            ("target_in_direct_pool", "target_in_direct_pool"),
            ("target_energy", "target_energy"),
            ("target_in_graph_results", "target_in_graph_results"),
            ("valid_path", "valid_path"),
            ("baseline_result_ids", "baseline_result_ids"),
            ("engine_graph_result_ids", "graph_result_ids"),
            ("engine_graph_target_trace", "graph_target_trace"),
        ):
            if target not in normalized_row:
                source_data = query_data if source in query_data else production
                if source in source_data:
                    normalized_row[target] = source_data[source]
        normalized_row.setdefault("case_id", case.id)
        normalized_row.setdefault("template_id", case.template_id)
        normalized_row.setdefault("split", case.split)
        normalized_row.setdefault("repetition", repetition)
        normalized_row.setdefault("target_incoming_edge_cutset", [])
        normalized_row.setdefault("edge_removal", {})
        rows.append(normalized_row)
        if on_row is not None:
            on_row(rows[-1])
    return rows


def _has_cycle(edges: Mapping[str, Mapping[str, float]], allowed: set[str]) -> bool:
    seen: set[str] = set()

    def walk(node: str, parent: str | None) -> bool:
        seen.add(node)
        for neighbor in edges.get(node, {}):
            if neighbor not in allowed:
                continue
            if neighbor not in seen and walk(neighbor, node):
                return True
            if neighbor != parent:
                return True
        return False

    return any(node not in seen and walk(node, None) for node in sorted(allowed))


def evaluate_affective_pair(
    pair: AffectivePair,
    *,
    repetitions: int = 3,
    llm: LLMProvider | None = None,
    on_row: Callable[[dict[str, Any]], None] | None = None,
    skip_repetitions: set[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    if repetitions < MIN_REPETITIONS:
        raise ValueError("mechanism evaluation requires at least three repetitions")
    if pair.query_embedding is None:
        raise ValueError("affective pair has no frozen query embedding")
    engine, store = _engine_for(
        pair.facts, pair.semantic_query, pair.query_embedding, graph=False, affective=True
    )
    _insert_and_refresh(engine, store, pair.facts)
    router = (
        AffectiveRouter(llm, engine._embedder, engine.config.affective) if llm is not None else None
    )
    chunks = {fact.id: fact.chunk() for fact in pair.facts}
    rows: list[dict[str, Any]] = []
    for condition in pair.conditions:
        # Semantic ranking is paired on the one frozen vector from the pair.
        # The wording carrying the state label is used only by the classifier.
        condition_embedding = pair.query_embedding
        semantic_breakdowns = [
            score_candidate(
                chunk, condition_embedding, current_state=None, affective=False, graph_weight=0
            )
            for chunk in chunks.values()
        ]
        state_breakdowns = [
            score_candidate(
                chunk,
                condition_embedding,
                current_state=condition.state,
                affective=True,
                similarity_weight=0.7,
                state_weight=0.3,
                graph_weight=0,
            )
            for chunk in chunks.values()
        ]
        semantic_scores = [breakdown.total for breakdown in semantic_breakdowns]
        state_scores = [breakdown.total for breakdown in state_breakdowns]
        semantic_rank = [
            item[0]
            for item in sorted(zip(chunks, semantic_scores), key=lambda item: (-item[1], item[0]))
        ]
        state_rank = [
            item[0]
            for item in sorted(zip(chunks, state_scores), key=lambda item: (-item[1], item[0]))
        ]
        semantic_by_id = {fact_id: score for fact_id, score in zip(chunks, semantic_scores)}
        state_by_id = {fact_id: score for fact_id, score in zip(chunks, state_scores)}
        for repetition in range(repetitions):
            if (
                skip_repetitions is not None
                and (condition.target_id, repetition) in skip_repetitions
            ):
                continue
            classifier_state: CognitiveState | None = None
            classifier_status = "not_run"
            if router is not None:
                classifier_state = _run(
                    router.classify_state(
                        Interaction(
                            role="user", content=condition.query, metadata=dict(pair.context)
                        )
                    )
                )
                classifier_status = "measured" if classifier_state is not None else "unavailable"
            row = {
                "pair_id": pair.id,
                "template_id": pair.template_id,
                "split": pair.split,
                "condition_query": condition.query,
                "target_id": condition.target_id,
                "repetition": repetition,
                "semantic_top_id": semantic_rank[0] if semantic_rank else None,
                "affective_top_id": state_rank[0] if state_rank else None,
                "semantic_top5_ids": semantic_rank[:5],
                "affective_top5_ids": state_rank[:5],
                "semantic_scores": semantic_by_id,
                "affective_scores": state_by_id,
                "semantic_score_components": {
                    fact_id: breakdown.model_dump(mode="json")
                    for fact_id, breakdown in zip(chunks, semantic_breakdowns)
                },
                "affective_score_components": {
                    fact_id: breakdown.model_dump(mode="json")
                    for fact_id, breakdown in zip(chunks, state_breakdowns)
                },
                "semantic_target_rank": (
                    semantic_rank.index(condition.target_id) + 1
                    if condition.target_id in semantic_rank
                    else None
                ),
                "affective_target_rank": (
                    state_rank.index(condition.target_id) + 1
                    if condition.target_id in state_rank
                    else None
                ),
                "affective_target_in_top_k": condition.target_id in state_rank[:5],
                "classifier_status": classifier_status,
                "classifier_state": (
                    classifier_state.model_dump(mode="json")
                    if classifier_state is not None
                    else None
                ),
                "classifier_distance": (
                    classifier_state.distance(condition.state)
                    if classifier_state is not None
                    else None
                ),
                "no_state_ids_present": sorted(set(pair.no_state_ids) & set(chunks)),
            }
            rows.append(row)
            if on_row is not None:
                on_row(row)
    return rows


def _summary_graph(
    rows: Sequence[Mapping[str, Any]], *, repetitions: int = MIN_REPETITIONS
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    complete = {
        case_id: len({int(item.get("repetition", -1)) for item in items}) == repetitions
        and len(items) == repetitions
        for case_id, items in grouped.items()
    }
    eligible = {
        case_id: items
        for case_id, items in grouped.items()
        if complete[case_id] and all(item.get("status") == "observed" for item in items)
    }
    heldout = {
        case_id: items
        for case_id, items in eligible.items()
        if items and items[0].get("split") == "heldout"
    }
    positive_counts = {
        case_id: sum(
            bool(item.get("target_in_graph_results"))
            and float(item.get("target_energy", 0)) >= 0.3
            and bool(item.get("paths") or item.get("valid_path"))
            for item in items
        )
        for case_id, items in grouped.items()
    }
    positive_ids = [
        case_id for case_id in heldout if positive_counts.get(case_id, 0) == repetitions
    ]
    baseline_ids = [
        case_id
        for case_id, items in heldout.items()
        if any(
            str(item.get("target_id"))
            in {str(value) for value in item.get("baseline_result_ids", [])}
            for item in items
        )
    ]
    interval = wilson_interval(len(positive_ids), len(heldout)) if heldout else None
    templates = sorted({str(items[0].get("template_id")) for items in heldout.values() if items})
    all_heldout = {
        case_id
        for case_id, items in grouped.items()
        if items and items[0].get("split") == "heldout"
    }
    ineligible_ids = sorted(all_heldout - set(eligible))
    miss_ids = sorted(set(heldout) - set(positive_ids))
    paired_baseline = [
        statistics.mean(
            str(row.get("target_id")) in row.get("baseline_result_ids", []) for row in items
        )
        for items in heldout.values()
    ]
    paired_graph = [positive_counts[case_id] / repetitions for case_id in heldout]
    paired_interval = (
        paired_bootstrap_interval(
            paired_baseline,
            paired_graph,
            cluster_ids=[
                str(items[0].get("template_id") or case_id) for case_id, items in heldout.items()
            ],
        ).as_dict()
        if heldout
        else None
    )
    return {
        "unique_cases": len(grouped),
        "eligible_cases": len(eligible),
        "ineligible_case_ids": ineligible_ids,
        "heldout_cases": len(all_heldout),
        "heldout_eligible_cases": len(heldout),
        "positive_cases": len(positive_ids),
        "positive_case_ids": sorted(positive_ids),
        "miss_case_ids": miss_ids,
        "positive_repetitions": sum(positive_counts.values()),
        "observations": len(rows),
        "complete_cases": sum(complete.values()),
        "template_clusters": templates,
        "baseline_target_direct_cases": len(baseline_ids),
        "baseline_target_direct_rate": len(baseline_ids) / len(heldout) if heldout else None,
        "wilson_95": interval.as_dict() if interval else None,
        "paired_graph_top_k_difference_95": paired_interval,
        "wilson_limitation": "Descriptive case interval; repeated templates are not independent. Use the template-clustered paired interval for differences.",
    }


def _summary_affective(
    rows: Sequence[Mapping[str, Any]], *, repetitions: int = MIN_REPETITIONS
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["pair_id"]), str(row.get("condition_query", row.get("target_id")))), []
        ).append(row)
    unique_rows = [items for items in grouped.values() if items and len(items) == repetitions]
    baseline = [
        statistics.mean(1.0 if item.get("semantic_target_rank") == 1 else 0.0 for item in items)
        for items in unique_rows
    ]
    treatment = [
        statistics.mean(1.0 if item.get("affective_target_rank") == 1 else 0.0 for item in items)
        for items in unique_rows
    ]
    group_items = [
        (key, items) for key, items in grouped.items() if items and len(items) == repetitions
    ]
    first_rows = [items[0] for _, items in group_items]
    cluster_ids = [
        str(first_rows[index].get("template_id") or key[0])
        for index, (key, _) in enumerate(group_items)
    ]
    interval = (
        paired_bootstrap_interval(baseline, treatment, cluster_ids=cluster_ids)
        if unique_rows
        else None
    )
    parsed = [row for row in rows if row.get("classifier_status") == "measured"]
    fidelity = [
        row
        for row in parsed
        if row.get("classifier_distance") is not None and float(row["classifier_distance"]) <= 0.2
    ]
    pair_ids = sorted({key[0] for key in grouped})
    discriminating_pairs = {
        key[0]
        for key, items in grouped.items()
        if items
        and any(item.get("semantic_top_id") != item.get("affective_top_id") for item in items)
    }
    no_state_rows = [row for row in rows if row.get("no_state_ids_present")]
    baseline_top5 = [
        statistics.mean((item.get("semantic_target_rank") or 999) <= 5 for item in items)
        for items in unique_rows
    ]
    treatment_top5 = [
        statistics.mean((item.get("affective_target_rank") or 999) <= 5 for item in items)
        for items in unique_rows
    ]
    top5_interval = (
        paired_bootstrap_interval(baseline_top5, treatment_top5, cluster_ids=cluster_ids)
        if unique_rows
        else None
    )
    return {
        "unique_pairs": len(pair_ids),
        "unique_conditions": len(grouped),
        "complete_conditions": len(unique_rows),
        "incomplete_pair_ids": sorted(
            {key[0] for key, items in grouped.items() if len(items) != repetitions}
        ),
        "observations": len(rows),
        "template_clusters": sorted({str(item.get("template_id")) for item in first_rows}),
        "state_appropriate_top1": sum(treatment),
        "semantic_target_top1_rate": statistics.mean(baseline) if baseline else None,
        "affective_target_top1_rate": statistics.mean(treatment) if treatment else None,
        "non_discriminating_pairs": sorted(set(pair_ids) - discriminating_pairs),
        "paired_top1_difference_95": interval.as_dict() if interval else None,
        "paired_top5_difference_95": top5_interval.as_dict() if top5_interval else None,
        "classifier_calls": sum(
            row.get("classifier_status") in {"measured", "unavailable"} for row in rows
        ),
        "classifier_parsed_states": len(parsed),
        "classifier_fidelity_passes": len(fidelity),
        "classifier_fidelity_missing": len(rows) - len(parsed),
        "no_state_negative_observations": len(no_state_rows),
    }


def _graph_correctness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    cycle_observations = 0
    for row in rows:
        identity = {"case_id": row.get("case_id"), "repetition": row.get("repetition")}
        controls = row.get("correctness_controls", {})
        production = row.get("production", {})
        coverage = row.get("coverage", {})
        cycle_observations += coverage.get("cycle_observed") is True
        checks = {
            "paired_snapshot": controls.get("snapshot", {}).get("byte_identical_before_open"),
            "activation_energy": controls.get("activation_vs_independent_path", {}).get("matches"),
            "insertion_edges": controls.get("insertion_order", {}).get("edges_equal"),
            "insertion_results": controls.get("insertion_order", {}).get("results_equal"),
            "scope_isolation": (
                not production["scope_control_ids_forbidden_observed"]
                if "scope_control_ids_forbidden_observed" in production
                else None
            ),
        }
        if row.get("eligible") is True:
            cut = controls.get("edge_cut", {})
            checks["edge_cut"] = (
                not cut["target_present_after"]
                and production.get("target_id") not in cut.get("result_ids_after_cut", [])
                if cut.get("query_executed") is True and "target_present_after" in cut
                else None
            )
        for check, passed in checks.items():
            if passed is None:
                missing.append({**identity, "check": check})
            elif passed is not True:
                failures.append({**identity, "check": check})
    status = "FAIL" if failures else ("NOT MEASURED" if missing or not rows else "PASS")
    return {
        "status": status,
        "correctness_pass": status == "PASS",
        "failures": failures,
        "missing": missing,
        "cycle_observations": cycle_observations,
        "cycle_coverage": "OBSERVED" if cycle_observations else "NOT OBSERVED",
    }


def evaluate_dataset(
    dataset_path: Path,
    *,
    output_dir: Path | None = None,
    work_dir: Path | None = None,
    repetitions: int | None = None,
    llm: LLMProvider | None = None,
) -> dict[str, Any]:
    dataset = load_mechanism_dataset(dataset_path)
    requested = (
        repetitions
        if repetitions is not None
        else int(dataset.manifest.get("repetitions", MIN_REPETITIONS))
    )
    if requested < MIN_REPETITIONS:
        raise ValueError("mechanism evaluation requires at least three repetitions")
    checkpoint_path = (work_dir or dataset_path.parent) / "module-mechanism.checkpoint.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    binding_digest = _json_digest(dataset.manifest.get("binding", {}))
    known_ids = {case.id for case in dataset.graph_cases} | {
        pair.id for pair in dataset.affective_pairs
    }
    completed: set[str] = set()
    graph_rows: list[dict[str, Any]] = []
    affective_rows: list[dict[str, Any]] = []
    if checkpoint_path.exists():
        try:
            prior = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if not isinstance(prior, Mapping):
                raise ValueError("checkpoint is malformed")
            required_checkpoint_fields = {
                "schema_version",
                "dataset_sha256",
                "binding_sha256",
                "repetitions",
                "execution_seed",
                "completed",
                "graph_rows",
                "affective_rows",
            }
            if not required_checkpoint_fields <= set(prior):
                raise ValueError("checkpoint is incomplete")
            if prior.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("checkpoint schema is incompatible")
            if prior.get("dataset_sha256") != dataset.digest:
                raise ValueError("checkpoint belongs to a different dataset digest")
            if prior.get("binding_sha256") != binding_digest:
                raise ValueError("checkpoint belongs to a different protocol binding")
            if prior.get("repetitions") != requested:
                raise ValueError("checkpoint repetitions do not match this run")
            prior_completed = prior.get("completed", [])
            if not isinstance(prior_completed, list) or len(
                {str(item) for item in prior_completed}
            ) != len(prior_completed):
                raise ValueError("checkpoint completed IDs are malformed or duplicated")
            if prior.get("execution_seed") != EXECUTION_SEED:
                raise ValueError("checkpoint execution seed does not match this run")
            completed = {str(item) for item in prior_completed}
            if not completed <= known_ids:
                raise ValueError("checkpoint contains an unknown case or pair")
            raw_graph_rows = prior.get("graph_rows", [])
            raw_affective_rows = prior.get("affective_rows", [])
            if not isinstance(raw_graph_rows, list) or not all(
                isinstance(item, Mapping) for item in raw_graph_rows
            ):
                raise ValueError("checkpoint graph observations are malformed")
            if not isinstance(raw_affective_rows, list) or not all(
                isinstance(item, Mapping) for item in raw_affective_rows
            ):
                raise ValueError("checkpoint affective observations are malformed")
            graph_rows = [dict(item) for item in raw_graph_rows]
            affective_rows = [dict(item) for item in raw_affective_rows]
            graph_seen: set[tuple[str, int]] = set()
            for row in graph_rows:
                try:
                    gkey = (str(row.get("case_id")), int(row.get("repetition", -1)))
                except (TypeError, ValueError) as exc:
                    raise ValueError("checkpoint contains an invalid graph observation") from exc
                if (
                    gkey[0] not in {case.id for case in dataset.graph_cases}
                    or not 0 <= gkey[1] < requested
                    or gkey in graph_seen
                ):
                    raise ValueError(
                        "checkpoint contains an invalid or duplicate graph observation"
                    )
                graph_seen.add(gkey)
            affective_seen: set[tuple[str, str, int]] = set()
            pair_targets = {
                pair.id: {condition.target_id for condition in pair.conditions}
                for pair in dataset.affective_pairs
            }
            for row in affective_rows:
                pair_id = str(row.get("pair_id"))
                try:
                    akey = (pair_id, str(row.get("target_id")), int(row.get("repetition", -1)))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "checkpoint contains an invalid affective observation"
                    ) from exc
                if (
                    pair_id not in pair_targets
                    or akey[1] not in pair_targets[pair_id]
                    or not 0 <= akey[2] < requested
                    or akey in affective_seen
                ):
                    raise ValueError(
                        "checkpoint contains an invalid or duplicate affective observation"
                    )
                affective_seen.add(akey)
            for case in dataset.graph_cases:
                if (
                    case.id in completed
                    and len(
                        {
                            int(row.get("repetition", -1))
                            for row in graph_rows
                            if row.get("case_id") == case.id
                        }
                    )
                    < requested
                ):
                    raise ValueError("checkpoint marks an incomplete graph case as complete")
            for pair in dataset.affective_pairs:
                if pair.id in completed:
                    observed = {
                        (str(row.get("target_id")), int(row.get("repetition", -1)))
                        for row in affective_rows
                        if row.get("pair_id") == pair.id
                    }
                    expected = {
                        (condition.target_id, repetition)
                        for condition in pair.conditions
                        for repetition in range(requested)
                    }
                    if not expected <= observed:
                        raise ValueError(
                            "checkpoint marks an incomplete affective pair as complete"
                        )
        except OSError as exc:
            raise ValueError("checkpoint cannot be read") from exc
        except json.JSONDecodeError as exc:
            raise ValueError("checkpoint is malformed") from exc

    def save_checkpoint() -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "dataset_sha256": dataset.digest,
            "binding_sha256": binding_digest,
            "repetitions": requested,
            "execution_seed": EXECUTION_SEED,
            "completed": sorted(completed),
            "graph_rows": graph_rows,
            "affective_rows": affective_rows,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, checkpoint_path)

    def record_graph(row: dict[str, Any]) -> None:
        graph_rows.append(row)
        save_checkpoint()

    def record_affective(row: dict[str, Any]) -> None:
        affective_rows.append(row)
        save_checkpoint()

    execution_order = [
        *[f"graph:{case.id}" for case in dataset.graph_cases],
        *[f"affective:{pair.id}" for pair in dataset.affective_pairs],
    ]
    random.Random(EXECUTION_SEED).shuffle(execution_order)
    execution_items: dict[str, tuple[str, Any]] = {
        **{f"graph:{case.id}": ("graph", case) for case in dataset.graph_cases},
        **{f"affective:{pair.id}": ("affective", pair) for pair in dataset.affective_pairs},
    }
    for item_key in execution_order:
        kind, item = execution_items[item_key]
        if kind == "graph":
            case = item
            if case.id not in completed:
                graph_observed = {
                    int(row.get("repetition", -1))
                    for row in graph_rows
                    if row.get("case_id") == case.id
                }
                evaluate_graph_case(
                    case,
                    repetitions=requested,
                    work_dir=work_dir,
                    on_row=record_graph,
                    repetition_indices=[
                        index for index in range(requested) if index not in graph_observed
                    ],
                )
                if (
                    len(
                        {
                            int(row.get("repetition", -1))
                            for row in graph_rows
                            if row.get("case_id") == case.id
                        }
                    )
                    == requested
                ):
                    completed.add(case.id)
                save_checkpoint()
        else:
            pair = item
            if pair.id not in completed:
                affective_observed = {
                    (str(row.get("target_id")), int(row.get("repetition", -1)))
                    for row in affective_rows
                    if row.get("pair_id") == pair.id
                }
                expected = {
                    (condition.target_id, repetition)
                    for condition in pair.conditions
                    for repetition in range(requested)
                }
                evaluate_affective_pair(
                    pair,
                    repetitions=requested,
                    llm=llm,
                    on_row=record_affective,
                    skip_repetitions=affective_observed,
                )
                affective_observed = {
                    (str(row.get("target_id")), int(row.get("repetition", -1)))
                    for row in affective_rows
                    if row.get("pair_id") == pair.id
                }
                if expected <= affective_observed:
                    completed.add(pair.id)
                save_checkpoint()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "interpretation": "observations only; this report does not establish product utility or human acceptance",
        "acceptance": {
            "status": "NOT RUN",
            "reason": "mechanism evaluation has no human-acceptance gate",
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset.digest,
            "graph_cases": len(dataset.graph_cases),
            "affective_pairs": len(dataset.affective_pairs),
        },
        "repetitions": requested,
        "execution_seed": EXECUTION_SEED,
        "execution_order": execution_order,
        "graph": {"rows": graph_rows, "summary": _summary_graph(graph_rows, repetitions=requested)},
        "affective": {
            "rows": affective_rows,
            "summary": _summary_affective(affective_rows, repetitions=requested),
        },
        "statistics": {
            "repetitions_are_clustered_by_fixed_case_id": True,
            "gold_states_are_synthetic_labels": True,
        },
        "checkpoint": str(checkpoint_path),
        "graph_correctness": _graph_correctness(graph_rows),
    }
    graph_summary = report["graph"]["summary"]
    expected_graph = int(dataset.manifest.get("graph_cases", len(dataset.graph_cases)))
    expected_pairs = int(dataset.manifest.get("affective_pairs", len(dataset.affective_pairs)))
    report["execution"] = {
        "complete": len(completed) >= expected_graph + expected_pairs,
        "graph_cases_expected": expected_graph,
        "graph_cases_completed": len({str(row.get("case_id")) for row in graph_rows}),
        "affective_pairs_expected": expected_pairs,
        "affective_pairs_completed": len({str(row.get("pair_id")) for row in affective_rows}),
    }
    report["coverage"] = {
        "minimum_heldout_graph_cases": 50,
        "minimum_eligible_graph_cases": 40,
        "graph_heldout_cases": graph_summary["heldout_cases"],
        "graph_heldout_eligible_cases": graph_summary["heldout_eligible_cases"],
        "minimum_heldout_affective_pairs": 50,
        "heldout_affective_pairs": len(
            {str(row.get("pair_id")) for row in affective_rows if row.get("split") == "heldout"}
        ),
        "affective_pairs_with_all_conditions_and_repetitions": sum(
            1
            for pair in dataset.affective_pairs
            if pair.split == "heldout"
            and len(
                {
                    (str(row.get("target_id")), int(row.get("repetition", -1)))
                    for row in affective_rows
                    if row.get("pair_id") == pair.id
                }
            )
            >= len(pair.conditions) * requested
        ),
        "failed": (
            graph_summary["heldout_cases"] < 50
            or graph_summary["heldout_eligible_cases"] < 40
            or len(
                {str(row.get("pair_id")) for row in affective_rows if row.get("split") == "heldout"}
            )
            < 50
            or requested < MIN_REPETITIONS
        ),
    }
    report["coverage"]["minimum_complete_heldout_affective_pairs"] = 50
    report["coverage"]["failed"] = bool(
        report["coverage"]["failed"]
        or report["coverage"]["affective_pairs_with_all_conditions_and_repetitions"] < 50
    )
    report["correctness_pass"] = report["graph_correctness"]["correctness_pass"]
    report["overall_mechanism_gates_pass"] = bool(
        report["execution"]["complete"]
        and not report["coverage"]["failed"]
        and report["correctness_pass"]
    )
    report["acceptance"] = {
        "status": "NOT RUN",
        "coverage_pass": not report["coverage"]["failed"],
        "graph_correctness": report["graph_correctness"]["status"],
        "correctness_pass": report["correctness_pass"],
        "overall_mechanism_gates_pass": report["overall_mechanism_gates_pass"],
        "reason": "coverage and mechanism observations require separate scientific interpretation",
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "module-mechanism.json"
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        for name, observations in (("graph", graph_rows), ("affective", affective_rows)):
            (output_dir / f"{name}.observations.jsonl").write_text(
                "".join(
                    json.dumps(row, sort_keys=True, default=str) + "\n" for row in observations
                ),
                encoding="utf-8",
            )
        failures = {
            "graph_ineligible_case_ids": graph_summary["ineligible_case_ids"],
            "graph_miss_case_ids": graph_summary["miss_case_ids"],
            "graph_correctness": report["graph_correctness"],
            "coverage": report["coverage"],
            "affective": report["affective"]["summary"],
        }
        (output_dir / "failures-and-coverage.json").write_text(
            json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        lines = [
            "# Module mechanism evaluation",
            "",
            f"Status: {report['status']}",
            "Interpretation: observations only; this does not establish product utility or human acceptance.",
            "Acceptance: NOT RUN (mechanism evaluation has no human-acceptance gate).",
            f"Dataset SHA256: {dataset.digest}",
            f"Execution seed: {EXECUTION_SEED}",
            f"Coverage failed: {report['coverage']['failed']}",
            f"Graph correctness: {report['graph_correctness']['status']}",
            f"Correctness pass: {report['correctness_pass']}",
            f"Overall mechanism gates pass: {report['overall_mechanism_gates_pass']}",
            "Graph observations use production SQLite snapshots. Affective ranking uses synthetic gold states and the production scorer; model classifier fidelity is separate.",
            "Embedding and classifier provider provenance is recorded by the launcher. Offline observations alone do not establish real-model use or semantic quality.",
            "",
            "Repeated rows are clustered by fixed case ID; they are not independent cases.",
            "",
        ]
        lines.append(f"Graph cases: {len(dataset.graph_cases)}; observations: {len(graph_rows)}")
        lines.append(
            f"Affective pairs: {len(dataset.affective_pairs)}; observations: {len(affective_rows)}"
        )
        (output_dir / "module-mechanism.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the offline Mnemos module mechanism evaluation"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=MIN_REPETITIONS)
    args = parser.parse_args(argv)
    report = evaluate_dataset(
        args.dataset,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        repetitions=args.repetitions,
    )
    print(
        json.dumps(
            {"status": report["status"], "dataset_sha256": report["dataset"]["sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
