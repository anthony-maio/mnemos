"""Disposable semantic/source relation views for development measurements.

The view changes only the in-process ``get_graph_neighbors`` method on an
already populated engine. Persisted semantic edges, facts, and source evidence
are never rewritten. Source relations come from the read-only structural
source-association diagnostic, so a new view observes corrections and forgets
after the caller reopens it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
import copy
import math
from typing import Any, Literal

from mnemos.engine import MnemosEngine
from mnemos.types import MemoryChunk

from .source_association_diagnostic import inspect_source_associations

RelationArm = Literal["semantic", "source_association", "combined"]
_ARMS = frozenset({"semantic", "source_association", "combined"})


class SourceAssociationView:
    """A temporary relation projection over one populated SQLite engine."""

    def __init__(
        self,
        engine: MnemosEngine,
        *,
        arm: RelationArm,
        source_edge_weight: float = 1.0,
        fanout_limit: int = 16,
        excluded_pairs: Iterable[tuple[str, str]] | None = None,
        cut_fact_id: str | None = None,
    ) -> None:
        if arm not in _ARMS:
            raise ValueError(f"Unknown relation arm: {arm!r}")
        if (
            isinstance(source_edge_weight, bool)
            or not isinstance(source_edge_weight, (int, float))
            or not math.isfinite(float(source_edge_weight))
            or not 0.0 < float(source_edge_weight) <= 1.0
        ):
            raise ValueError("source_edge_weight must be a finite positive number")
        if isinstance(fanout_limit, bool) or not isinstance(fanout_limit, int) or fanout_limit < 2:
            raise ValueError("fanout_limit must be an integer >= 2")
        self.engine = engine
        self.arm = arm
        self.source_edge_weight = float(source_edge_weight)
        self.fanout_limit = fanout_limit
        self.cut_fact_id = cut_fact_id
        self._configured_excluded_pairs = self._normalize_excluded_pairs(excluded_pairs)
        self._excluded_pairs = set(self._configured_excluded_pairs)
        self._store: Any | None = None
        self._original_neighbors: Callable[..., list[tuple[MemoryChunk, float]]] | None = None
        self._had_instance_method = False
        self._saved_nodes: dict[str, Any] = {}
        self._saved_authoritative: dict[str, Any] = {}
        self._saved_preferences: dict[str, dict[str, float]] = {}
        self._saved_source_policies: dict[str, Any] = {}
        self._saved_preferences_initialized = False
        self._saved_external_snapshot = False
        self._saved_total_activations = 0
        self._semantic_edges: dict[tuple[str, str], float] = {}
        self._source_edges: dict[tuple[str, str], float] = {}
        self._source_adjacency: dict[str, dict[str, float]] = defaultdict(dict)
        self._provenance: dict[tuple[str, str], dict[str, Any]] = {}
        self._semantic_provenance: dict[tuple[str, str], dict[str, Any]] = {}
        self._source_provenance: dict[tuple[str, str], dict[str, Any]] = {}
        self._diagnostic: dict[str, Any] = {}
        self._entered = False

    @classmethod
    def _normalize_excluded_pairs(
        cls, pairs: Iterable[tuple[str, str]] | None
    ) -> set[tuple[str, str]]:
        normalized: set[tuple[str, str]] = set()
        if pairs is None:
            return normalized
        for pair in pairs:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or not all(isinstance(item, str) and item.strip() for item in pair)
            ):
                raise ValueError("excluded_pairs must contain nonempty string pairs")
            first, second = (item.strip() for item in pair)
            if first == second:
                raise ValueError("excluded_pairs cannot contain self-pairs")
            normalized.add(cls._pair(first, second))
        return normalized

    @staticmethod
    def _pair(first: str, second: str) -> tuple[str, str]:
        return (first, second) if first < second else (second, first)

    def _load_relations(self, store: Any) -> None:
        get_edges = getattr(store, "get_graph_edges", None)
        if not callable(get_edges):
            raise TypeError("relation views require a SQLite-like persisted graph store")
        for source_id, targets in get_edges().items():
            for target_id, raw_weight in targets.items():
                weight = float(raw_weight)
                if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
                    raise ValueError("persisted semantic edge has an invalid weight")
                pair = self._pair(str(source_id), str(target_id))
                self._semantic_edges[pair] = max(self._semantic_edges.get(pair, 0.0), weight)

        db_path = getattr(store, "db_path", None)
        if not isinstance(db_path, str) or not db_path or db_path == ":memory:":
            raise TypeError("source-association views require a file-backed SQLite store")
        diagnostic = inspect_source_associations(db_path, fanout_limit=self.fanout_limit)
        self._diagnostic = diagnostic
        for relation in diagnostic["relations"]:
            first_id, second_id = (str(item) for item in relation["fact_ids"])
            pair = self._pair(first_id, second_id)
            self._source_edges[pair] = max(
                self._source_edges.get(pair, 0.0), self.source_edge_weight
            )
            self._source_adjacency[first_id][second_id] = self.source_edge_weight
            self._source_adjacency[second_id][first_id] = self.source_edge_weight

            source_details = {
                "relation_id": relation["relation_id"],
                "source_event_id": relation["source_event_id"],
                "fact_ids": list(relation["fact_ids"]),
                "fact_versions": dict(relation["fact_versions"]),
                "evidence_ids": list(relation["evidence_ids"]),
                "scope": dict(relation["scope"]),
                "fact_scopes": copy.deepcopy(relation["fact_scopes"]),
                "source_scope": dict(relation["source_scope"]),
                "source_role": relation["source_role"],
                "source_type": relation["source_type"],
            }
            self._source_provenance.setdefault(
                pair, {"weight": self.source_edge_weight, "relations": []}
            )["relations"].append(source_details)

        self._rebuild_provenance()

    def _rebuild_provenance(self) -> None:
        self._provenance.clear()
        self._semantic_provenance.clear()
        self._source_provenance = {
            pair: value
            for pair, value in self._source_provenance.items()
            if not self._is_excluded(pair)
        }
        for pair in set(self._semantic_edges) | set(self._source_edges):
            if self._is_excluded(pair):
                continue
            semantic = self._semantic_edges.get(pair)
            source = self._source_edges.get(pair)
            if semantic is not None:
                self._semantic_provenance[pair] = {"weight": semantic}
            components = {
                "semantic": semantic,
                "source_association": source,
            }
            active = {
                name: value
                for name, value in components.items()
                if value is not None and (self.arm == "combined" or name == self.arm)
            }
            if not active:
                continue
            effective = max(active.values())
            self._provenance[pair] = {
                "pair": list(pair),
                "semantic": semantic,
                "source_association": source,
                "effective_weight": effective,
                "available_components": [
                    name for name, value in components.items() if value is not None
                ],
                "components": list(active),
                "source_relations": copy.deepcopy(
                    self._source_provenance.get(pair, {}).get("relations", [])
                ),
            }

    def __enter__(self) -> "SourceAssociationView":
        if self._entered:
            raise RuntimeError("relation view cannot be entered twice")
        store = getattr(self.engine, "store", None)
        original = getattr(store, "get_graph_neighbors", None)
        if store is None or not callable(original):
            raise TypeError("relation views require an engine with get_graph_neighbors")
        self._store = store
        self._original_neighbors = original
        self._had_instance_method = "get_graph_neighbors" in getattr(store, "__dict__", {})
        spreading = self.engine.spreading_activation
        self._saved_nodes = {
            node.id: node.model_copy(deep=True) for node in spreading.get_all_nodes()
        }
        self._saved_authoritative = dict(getattr(spreading, "_authoritative", {}))
        self._saved_preferences = copy.deepcopy(getattr(spreading, "_preferences", {}))
        self._saved_source_policies = copy.deepcopy(getattr(spreading, "_source_policies", {}))
        self._saved_preferences_initialized = bool(
            getattr(spreading, "_preferences_initialized", False)
        )
        self._saved_external_snapshot = bool(
            getattr(spreading, "_has_external_authoritative_snapshot", False)
        )
        self._saved_total_activations = int(getattr(spreading, "_total_activations", 0))
        try:
            self._excluded_pairs = set(self._configured_excluded_pairs)
            self._reset_loaded_relations()
            self._load_relations(store)
            if self.cut_fact_id:
                self._excluded_pairs.update(
                    pair
                    for pair in set(self._semantic_edges) | set(self._source_edges)
                    if self.cut_fact_id in pair
                )
            self._rebuild_provenance()
            if self.arm != "semantic" or self._excluded_pairs:
                for node in self.engine.spreading_activation.get_all_nodes():
                    node.neighbors.clear()

            def view_neighbors(
                source_id: str,
                *,
                filter_fn: Callable[[MemoryChunk], bool],
                limit: int = 32,
            ) -> list[tuple[MemoryChunk, float]]:
                return self.get_graph_neighbors(source_id, filter_fn=filter_fn, limit=limit)

            store.get_graph_neighbors = view_neighbors
            self._entered = True
            return self
        except BaseException:
            self._restore()
            raise

    def _restore(self) -> None:
        if self._store is not None and self._original_neighbors is not None:
            if self._had_instance_method:
                self._store.get_graph_neighbors = self._original_neighbors
            else:
                self._store.__dict__.pop("get_graph_neighbors", None)
        spreading = self.engine.spreading_activation
        spreading._nodes = {
            node_id: node.model_copy(deep=True) for node_id, node in self._saved_nodes.items()
        }
        spreading._authoritative = {
            node_id: spreading._nodes.get(node_id, node)
            for node_id, node in self._saved_authoritative.items()
        }
        spreading._preferences = copy.deepcopy(self._saved_preferences)
        spreading._source_policies = copy.deepcopy(self._saved_source_policies)
        spreading._preferences_initialized = self._saved_preferences_initialized
        spreading._has_external_authoritative_snapshot = self._saved_external_snapshot
        spreading._total_activations = self._saved_total_activations
        self._entered = False

    def _reset_loaded_relations(self) -> None:
        self._semantic_edges.clear()
        self._source_edges.clear()
        self._source_adjacency.clear()
        self._provenance.clear()
        self._semantic_provenance.clear()
        self._source_provenance.clear()
        self._diagnostic.clear()

    def _is_excluded(self, pair: tuple[str, str]) -> bool:
        return pair in self._excluded_pairs

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._restore()
        return None

    @property
    def provenance(self) -> Mapping[tuple[str, str], Mapping[str, Any]]:
        return self._provenance

    def provenance_for(self, first_id: str, second_id: str) -> Mapping[str, Any] | None:
        return self._provenance.get(self._pair(first_id, second_id))

    @property
    def semantic_provenance(self) -> Mapping[tuple[str, str], Mapping[str, Any]]:
        return self._semantic_provenance

    @property
    def source_provenance(self) -> Mapping[tuple[str, str], Mapping[str, Any]]:
        return self._source_provenance

    @property
    def diagnostic(self) -> Mapping[str, Any]:
        return self._diagnostic

    @property
    def excluded_pairs(self) -> frozenset[tuple[str, str]]:
        return frozenset(self._excluded_pairs)

    def get_graph_neighbors(
        self,
        source_id: str,
        *,
        filter_fn: Callable[[MemoryChunk], bool],
        limit: int = 32,
    ) -> list[tuple[MemoryChunk, float]]:
        if not self._entered or self._store is None or self._original_neighbors is None:
            raise RuntimeError("relation view is not active")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            return []
        candidates: dict[str, tuple[MemoryChunk, float]] = {}
        if self.arm in {"semantic", "combined"}:
            for chunk, weight in self._original_neighbors(
                source_id, filter_fn=filter_fn, limit=limit
            ):
                if self._is_excluded(self._pair(source_id, chunk.id)):
                    continue
                candidates[chunk.id] = (chunk, float(weight))
        if self.arm in {"source_association", "combined"}:
            for target_id, weight in self._source_adjacency.get(source_id, {}).items():
                if self._is_excluded(self._pair(source_id, target_id)):
                    continue
                chunk = self._store.get(target_id)
                if chunk is None or not filter_fn(chunk):
                    continue
                existing = candidates.get(target_id)
                if existing is None or weight > existing[1]:
                    candidates[target_id] = (chunk, weight)
        return sorted(candidates.values(), key=lambda item: (-item[1], item[0].id))[:limit]


@contextmanager
def relation_view(
    engine: MnemosEngine,
    *,
    arm: RelationArm,
    source_edge_weight: float = 1.0,
    fanout_limit: int = 16,
    excluded_pairs: Iterable[tuple[str, str]] | None = None,
    cut_fact_id: str | None = None,
) -> Iterator[SourceAssociationView]:
    """Apply one disposable semantic/source/combined relation arm."""

    view = SourceAssociationView(
        engine,
        arm=arm,
        source_edge_weight=source_edge_weight,
        fanout_limit=fanout_limit,
        excluded_pairs=excluded_pairs,
        cut_fact_id=cut_fact_id,
    )
    with view:
        yield view
