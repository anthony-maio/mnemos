"""Scoped, bounded spreading activation for associative memory retrieval.

The current traversal algorithm is version 2. It records the greatest energy
available at each node from one seed to a bounded depth. Energies belong to a
single call, so a previous query cannot prime a later query through mutable node
state.
"""

from __future__ import annotations

import math
import uuid
from collections import deque
from collections.abc import Mapping
from typing import Any, Callable, Iterable

from ..config import SpreadingConfig
from ..types import ActivationNode, MemoryChunk
from ..utils.embeddings import EmbeddingProvider, cosine_similarity, embed_text_for_role

NodeFilter = Callable[[ActivationNode], bool]


class SpreadingActivation:
    """A sparse semantic graph with bounded, scope aware activation."""

    ALGORITHM_VERSION = "2"
    GRAPH_VERSION = "2"
    # Persisted edges are derived state.  Keep the derivation and traversal
    # versions alongside the rows so an incompatible graph can be discarded
    # and rebuilt without changing the authoritative memory facts.
    PERSISTENCE_SCHEMA_VERSION = "1"
    EDGE_BUILDER_VERSION = "4"
    TRAVERSAL_VERSION = ALGORITHM_VERSION

    def __init__(
        self,
        embedder: EmbeddingProvider,
        config: SpreadingConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._config = config or SpreadingConfig()
        self._nodes: dict[str, ActivationNode] = {}
        # Directed nearest-neighbor preferences are the authoritative in
        # memory derivation state. Persisted edges are reciprocal mutual-kNN
        # edges, so keeping preferences separate prevents a greedy insertion
        # from changing an unrelated node's choice.
        self._preferences: dict[str, dict[str, float]] = {}
        # This is a derivation-only view of authoritative facts.  It may
        # contain records which are not hydrated into ``_nodes`` and is never
        # traversed directly.
        self._authoritative: dict[str, ActivationNode] = {}
        self._has_external_authoritative_snapshot = False
        self._source_policies: dict[str, tuple[float, set[str] | None]] = {}
        self._preferences_initialized = False
        self._total_activations = 0

    @staticmethod
    def _metadata(node: ActivationNode) -> dict[str, Any]:
        return node.metadata if isinstance(node.metadata, dict) else {}

    @classmethod
    def _scope_data(cls, node: ActivationNode) -> tuple[str, str | None, str | None, str | None]:
        """Return scope, scope id, profile id, and workspace id."""
        metadata = cls._metadata(node)
        raw_scope = metadata.get("scope")
        scope = str(raw_scope).strip().lower() if isinstance(raw_scope, str) else "global"
        if scope not in {"project", "workspace", "global"}:
            scope = "global"

        def clean(value: Any) -> str | None:
            return value.strip() if isinstance(value, str) and value.strip() else None

        profile = clean(metadata.get("user_profile_id")) or clean(metadata.get("profile_id"))
        project_id = clean(metadata.get("project_id"))
        workspace_id = clean(metadata.get("workspace_id"))
        scope_id = clean(metadata.get("scope_id"))
        if scope == "project":
            return scope, project_id or scope_id, profile, workspace_id
        if scope == "workspace":
            return scope, workspace_id or scope_id, profile, workspace_id
        return scope, None, profile, workspace_id

    @staticmethod
    def _clean_id(value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    def _nodes_can_connect(self, node_a: ActivationNode, node_b: ActivationNode) -> bool:
        scope_a, id_a, profile_a, workspace_a = self._scope_data(node_a)
        scope_b, id_b, profile_b, workspace_b = self._scope_data(node_b)
        # Legacy nodes may connect to legacy nodes. A missing profile cannot be
        # mixed into a profiled graph where it could become a cross-user bridge.
        if profile_a != profile_b:
            return False
        if scope_a == "global" or scope_b == "global":
            return True
        if scope_a == scope_b:
            return id_a is not None and id_a == id_b
        # A project may carry explicit workspace membership.
        if scope_a == "project" and scope_b == "workspace":
            return workspace_a is not None and id_b == workspace_a
        if scope_a == "workspace" and scope_b == "project":
            return workspace_b is not None and id_a == workspace_b
        return False

    def add_node(
        self,
        content: str,
        embedding: list[float] | None = None,
        node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ActivationNode:
        """Add a node, or replace its content while retaining valid neighbors."""
        if embedding is None:
            embedding = embed_text_for_role(self._embedder, content, role="document")
        nid = node_id or str(uuid.uuid4())
        old = self._nodes.get(nid)
        old_neighbors = dict(old.neighbors) if old is not None else {}
        normalized_metadata = dict(metadata or {})
        if old is not None:
            for other in self._nodes.values():
                if other.id != nid:
                    other.neighbors.pop(nid, None)
        node = ActivationNode(
            id=nid,
            content=content,
            energy=0.0,
            neighbors={},
            embedding=embedding,
            metadata=normalized_metadata,
        )
        self._nodes[nid] = node
        for neighbor_id, weight in old_neighbors.items():
            neighbor = self._nodes.get(neighbor_id)
            if neighbor is None or neighbor_id == nid:
                continue
            try:
                numeric_weight = float(weight)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric_weight) or not 0.0 <= numeric_weight <= 1.0:
                continue
            if self._nodes_can_connect(node, neighbor):
                node.neighbors[neighbor_id] = numeric_weight
                neighbor.neighbors[nid] = numeric_weight
        if not self._has_external_authoritative_snapshot and embedding is not None:
            self._authoritative[nid] = node
        return node

    def refresh_node(
        self,
        node_id: str,
        content: str,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ActivationNode:
        """Replace corrected content and preserve still valid graph links."""
        return self.add_node(content, embedding=embedding, node_id=node_id, metadata=metadata)

    def add_node_from_chunk(self, chunk: MemoryChunk) -> ActivationNode:
        embedding = chunk.embedding or embed_text_for_role(
            self._embedder, chunk.content, role="document"
        )
        return self.add_node(
            content=chunk.content,
            embedding=embedding,
            node_id=chunk.id,
            metadata=dict(chunk.metadata),
        )

    def _max_neighbors(self) -> int:
        return max(int(self._config.max_neighbors_per_node), 1)

    def set_authoritative_snapshot(self, records: Iterable[ActivationNode]) -> None:
        """Replace the derivation view without hydrating traversal nodes."""
        snapshot: dict[str, ActivationNode] = {}
        for record in records:
            if record.embedding is None:
                continue
            snapshot[record.id] = record
        self._authoritative = snapshot
        self._has_external_authoritative_snapshot = True

    def _ensure_loaded_authoritative(self) -> None:
        """Use loaded nodes as the authority for standalone graph APIs."""
        if self._has_external_authoritative_snapshot:
            return
        self._authoritative = {
            node_id: node for node_id, node in self._nodes.items() if node.embedding is not None
        }

    def _source_policy(
        self,
        source_id: str,
        *,
        threshold: float | None = None,
        candidate_ids: set[str] | None = None,
    ) -> tuple[float, set[str] | None]:
        if threshold is None and candidate_ids is None and source_id in self._source_policies:
            stored_threshold, stored_candidates = self._source_policies[source_id]
            return stored_threshold, (None if stored_candidates is None else set(stored_candidates))
        return (
            float(self._config.auto_connect_threshold if threshold is None else threshold),
            None if candidate_ids is None else set(candidate_ids),
        )

    def _nearest_authoritative(
        self,
        source: ActivationNode,
        *,
        threshold: float | None = None,
        candidate_ids: set[str] | None = None,
    ) -> dict[str, float]:
        threshold = float(self._config.auto_connect_threshold if threshold is None else threshold)
        candidates: list[tuple[float, str]] = []
        for target_id, target in self._authoritative.items():
            if (
                target_id == source.id
                or (candidate_ids is not None and target_id not in candidate_ids)
                or not self._nodes_can_connect(source, target)
            ):
                continue
            similarity = float(cosine_similarity(source.embedding or [], target.embedding or []))
            if math.isfinite(similarity) and similarity >= threshold:
                candidates.append((similarity, target_id))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return {target_id: score for score, target_id in candidates[: self._max_neighbors()]}

    def hydrate_preferences(self, preference_map: Mapping[str, Any]) -> int:
        """Hydrate directed top-k preferences persisted by the engine."""
        normalized: dict[str, dict[str, float]] = {}
        for source_id, raw_targets in preference_map.items():
            if source_id not in self._authoritative and source_id not in self._nodes:
                continue
            if not isinstance(raw_targets, Mapping):
                continue
            targets: dict[str, float] = {}
            for target_id, raw_weight in raw_targets.items():
                if target_id == source_id:
                    continue
                try:
                    weight = float(raw_weight)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
                    continue
                targets[str(target_id)] = weight
            normalized[source_id] = dict(
                sorted(targets.items(), key=lambda item: (-item[1], item[0]))[
                    : self._max_neighbors()
                ]
            )
        self._preferences = normalized
        self._source_policies = {
            source_id: (float(self._config.auto_connect_threshold), None)
            for source_id in normalized
        }
        self._preferences_initialized = True
        self._rebuild_mutual_edges()
        return self.get_edge_count() // 2

    def get_preferences(self) -> dict[str, dict[str, float]]:
        return {source: dict(targets) for source, targets in self._preferences.items()}

    def derive_authoritative_nodes(self, source_ids: set[str]) -> set[str]:
        """Build a bounded set of sources for an initial rebuild step."""
        preferences = dict(self._preferences)
        computed: set[str] = set()
        for source_id in sorted(source_ids):
            source = self._authoritative.get(source_id)
            if source is not None:
                source_threshold, source_candidates = self._source_policy(source_id)
                self._source_policies[source_id] = (source_threshold, source_candidates)
                preferences[source_id] = self._nearest_authoritative(
                    source,
                    threshold=source_threshold,
                    candidate_ids=source_candidates,
                )
                computed.add(source_id)
        self._preferences = preferences
        self._preferences_initialized = True
        self._rebuild_mutual_edges()
        return computed

    def derive_authoritative_node(
        self,
        node_id: str,
        *,
        deleted_id: str | None = None,
        threshold: float | None = None,
        candidate_ids: set[str] | None = None,
    ) -> set[str]:
        """Update top-k preferences after one authoritative fact change.

        Insertions compare the new fact once with each authoritative record.
        A correction or deletion refills only sources whose preference was
        affected.  The caller persists the returned source rows.
        """
        if deleted_id is not None:
            self._authoritative.pop(deleted_id, None)
            self._preferences.pop(deleted_id, None)
            self._source_policies.pop(deleted_id, None)
            affected = {
                source_id
                for source_id, targets in self._preferences.items()
                if deleted_id in targets
            }
            for targets in self._preferences.values():
                targets.pop(deleted_id, None)
            for source_id in affected:
                source = self._authoritative.get(source_id)
                if source is not None:
                    source_threshold, source_candidates = self._source_policy(source_id)
                    self._preferences[source_id] = self._nearest_authoritative(
                        source,
                        threshold=source_threshold,
                        candidate_ids=source_candidates,
                    )
            self._rebuild_mutual_edges()
            return affected | {deleted_id}

        source = self._authoritative.get(node_id)
        if source is None:
            return set()
        effective_threshold, effective_candidates = self._source_policy(
            node_id,
            threshold=threshold,
            candidate_ids=candidate_ids,
        )
        self._source_policies[node_id] = (effective_threshold, effective_candidates)
        incoming_before = {
            source_id for source_id, targets in self._preferences.items() if node_id in targets
        }
        for targets in self._preferences.values():
            targets.pop(node_id, None)
        self._preferences[node_id] = self._nearest_authoritative(
            source, threshold=effective_threshold, candidate_ids=effective_candidates
        )
        affected = {node_id}
        # A correction can lower an existing preference.  Refill every source
        # that previously selected this target; insertions use the bounded
        # cutoff path below instead.
        for source_id in sorted(incoming_before):
            if source_id == node_id:
                continue
            candidate = self._authoritative.get(source_id)
            if candidate is not None:
                source_threshold, source_candidates = self._source_policy(source_id)
                self._preferences[source_id] = self._nearest_authoritative(
                    candidate,
                    threshold=source_threshold,
                    candidate_ids=source_candidates,
                )
                affected.add(source_id)
        # A new or corrected target can displace a source's current cutoff.
        # This is one comparison per source, rather than a source-by-source
        # full candidate scan. Sources already refilled above are skipped.
        for source_id, candidate in self._authoritative.items():
            if (
                source_id == node_id
                or source_id in incoming_before
                or source_id not in self._preferences
            ):
                continue
            if not self._nodes_can_connect(candidate, source):
                continue
            source_threshold, source_candidates = self._source_policy(source_id)
            if source_candidates is not None and node_id not in source_candidates:
                continue
            similarity = float(cosine_similarity(candidate.embedding or [], source.embedding or []))
            if not math.isfinite(similarity) or similarity < source_threshold:
                continue
            targets = dict(self._preferences[source_id])
            previous = targets.get(node_id)
            if previous is None and len(targets) >= self._max_neighbors():
                cutoff = min(targets.values())
                cutoff_id = max(target for target, weight in targets.items() if weight == cutoff)
                if similarity < cutoff or (similarity == cutoff and node_id >= cutoff_id):
                    continue
                targets.pop(cutoff_id)
            targets[node_id] = similarity
            self._preferences[source_id] = dict(
                sorted(targets.items(), key=lambda item: (-item[1], item[0]))[
                    : self._max_neighbors()
                ]
            )
            affected.add(source_id)
        self._preferences_initialized = True
        self._rebuild_mutual_edges()
        return affected

    def refresh_projected_edges(self) -> int:
        """Project the current directed preferences into loaded neighbors."""
        self._rebuild_mutual_edges()
        return self.get_edge_count() // 2

    def set_graph_pending(self) -> None:
        """Clear derived preferences while an invalidated rebuild is running."""
        self._preferences.clear()
        self._preferences_initialized = False
        self.clear_all_edges()

    def _rebuild_mutual_edges(self) -> None:
        for node in self._nodes.values():
            node.neighbors.clear()
        for source_id in sorted(self._preferences):
            for target_id, weight in sorted(self._preferences[source_id].items()):
                if source_id >= target_id:
                    continue
                if source_id not in self._preferences.get(target_id, {}):
                    continue
                source = self._nodes.get(source_id)
                target = self._nodes.get(target_id)
                if source is None or target is None or not self._nodes_can_connect(source, target):
                    continue
                target_weight = self._preferences[target_id][source_id]
                stable_weight = min(float(weight), float(target_weight))
                source.neighbors[target_id] = stable_weight
                target.neighbors[source_id] = stable_weight

    @classmethod
    def _normalized_scope_data(cls, node: ActivationNode) -> tuple[str, str | None]:
        """Compatibility view of the original scope helper."""
        scope, scope_id, _, _ = cls._scope_data(node)
        return scope, scope_id

    def remove_edge(
        self,
        node_a_id: str,
        node_b_id: str,
        *,
        bidirectional: bool = True,
    ) -> bool:
        if node_a_id not in self._nodes or node_b_id not in self._nodes:
            return False
        removed = self._nodes[node_a_id].neighbors.pop(node_b_id, None) is not None
        if bidirectional:
            removed = self._nodes[node_b_id].neighbors.pop(node_a_id, None) is not None or removed
        return removed

    def clear_edges(self, node_id: str) -> bool:
        node = self._nodes.get(node_id)
        if node is None:
            return False
        for neighbor_id in list(node.neighbors):
            self.remove_edge(node_id, neighbor_id)
        for other in self._nodes.values():
            if other.id != node_id:
                other.neighbors.pop(node_id, None)
        return True

    def clear_all_edges(self) -> None:
        for node in self._nodes.values():
            node.neighbors.clear()

    def add_edge(
        self,
        node_a_id: str,
        node_b_id: str,
        weight: float,
        bidirectional: bool = True,
    ) -> bool:
        if node_a_id not in self._nodes or node_b_id not in self._nodes:
            return False
        if node_a_id == node_b_id or not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            return False
        if not self._nodes_can_connect(self._nodes[node_a_id], self._nodes[node_b_id]):
            return False
        self._nodes[node_a_id].neighbors[node_b_id] = float(weight)
        if bidirectional:
            self._nodes[node_b_id].neighbors[node_a_id] = float(weight)
        return True

    def _apply_candidate_pairs(
        self,
        candidates: list[tuple[float, str, str]],
        *,
        exclude_existing: bool,
    ) -> int:
        limit = self._max_neighbors()
        edges_added = 0
        for similarity, node_a_id, node_b_id in sorted(
            candidates, key=lambda item: (-item[0], item[1], item[2])
        ):
            node_a = self._nodes.get(node_a_id)
            node_b = self._nodes.get(node_b_id)
            if node_a is None or node_b is None:
                continue
            if exclude_existing and node_b_id in node_a.neighbors:
                continue
            if len(node_a.neighbors) >= limit or len(node_b.neighbors) >= limit:
                continue
            if self.add_edge(node_a_id, node_b_id, similarity):
                edges_added += 1
        return edges_added

    @classmethod
    def persistence_contract(cls) -> dict[str, str]:
        """Return the versions required to interpret persisted graph edges."""
        return {
            "schema_version": cls.PERSISTENCE_SCHEMA_VERSION,
            "edge_builder_version": cls.EDGE_BUILDER_VERSION,
            "traversal_version": cls.TRAVERSAL_VERSION,
        }

    @classmethod
    def get_persistence_contract(cls) -> dict[str, str]:
        """Compatibility alias for callers exposing graph metadata."""
        return cls.persistence_contract()

    @classmethod
    def validate_persistence_contract(cls, metadata: Mapping[str, Any]) -> None:
        """Reject derived state from a graph algorithm this class cannot read."""
        if not isinstance(metadata, Mapping):
            raise ValueError("persisted graph metadata must be an object")
        expected = cls.persistence_contract()
        missing = [key for key in expected if key not in metadata]
        incompatible = [
            key
            for key, value in expected.items()
            if key in metadata and str(metadata[key]) != value
        ]
        if missing or incompatible:
            detail = ", ".join(missing + incompatible)
            raise ValueError(f"incompatible persisted graph contract: {detail}")

    def export_edges(self) -> dict[str, Any]:
        """Return a versioned envelope suitable for persisting derived edges."""
        return {
            **self.persistence_contract(),
            "edges": {
                node_id: dict(node.neighbors)
                for node_id, node in self._nodes.items()
                if node.neighbors
            },
        }

    def hydrate_edges(
        self,
        edge_map: Mapping[str, Any],
        *,
        persistence_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Load persisted edges while applying profile and scope policy.

        New callers should pass the versioned envelope returned by
        :meth:`export_edges`, or pass its metadata separately.  A raw edge map
        remains accepted for old storage implementations; it is treated as a
        legacy compatibility path and is never used to validate a versioned
        envelope silently.
        """
        if not isinstance(edge_map, Mapping):
            raise ValueError("persisted graph edges must be an object")
        raw_edges: Mapping[str, Any] = edge_map
        envelope_metadata = persistence_metadata
        if "edges" in edge_map and (
            "schema_version" in edge_map
            or "edge_builder_version" in edge_map
            or "traversal_version" in edge_map
        ):
            envelope_metadata = edge_map
            raw_edges = edge_map.get("edges")  # type: ignore[assignment]
            if not isinstance(raw_edges, Mapping):
                raise ValueError("persisted graph envelope edges must be an object")
        if envelope_metadata is not None:
            self.validate_persistence_contract(envelope_metadata)
        unique_pairs: dict[tuple[str, str], float] = {}
        for source_id, neighbors in raw_edges.items():
            if source_id not in self._nodes or not isinstance(neighbors, dict):
                continue
            for target_id, weight in neighbors.items():
                if target_id not in self._nodes or source_id == target_id:
                    continue
                try:
                    numeric_weight = float(weight)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric_weight) or not 0.0 <= numeric_weight <= 1.0:
                    continue
                pair = tuple(sorted((source_id, target_id)))
                unique_pairs[pair] = max(numeric_weight, unique_pairs.get(pair, 0.0))
        self.clear_all_edges()
        candidates = [(weight, pair[0], pair[1]) for pair, weight in unique_pairs.items()]
        return self._apply_candidate_pairs(candidates, exclude_existing=False)

    def connect_node(
        self, node_id: str, threshold: float | None = None, *, candidate_ids: set[str] | None = None
    ) -> int:
        """Derive one node through the bounded mutual top-k builder.

        ``candidate_ids`` retains the legacy candidate-pool boundary for the
        source's own preferences.  The resulting loaded edges are still
        projected only when both directed preferences agree.
        """
        node = self._nodes.get(node_id)
        if node is None or node.embedding is None:
            return 0
        self._ensure_loaded_authoritative()
        before = {
            tuple(sorted((source_id, target_id)))
            for source_id, source in self._nodes.items()
            for target_id in source.neighbors
            if source_id < target_id
        }
        self.derive_authoritative_node(
            node_id,
            threshold=threshold,
            candidate_ids=candidate_ids,
        )
        after = {
            tuple(sorted((source_id, target_id)))
            for source_id, source in self._nodes.items()
            for target_id in source.neighbors
            if source_id < target_id
        }
        return len(after - before)

    def auto_connect(
        self,
        threshold: float | None = None,
        exclude_existing: bool = True,
    ) -> int:
        """Build a deterministic bounded mutual top-k graph.

        The historical ``exclude_existing`` switch remains accepted for API
        compatibility.  Calling this derived builder is a rebuild boundary,
        so existing derived or manually added adjacency is replaced by the
        mutual projection rather than allowed to create non-mutual edges.
        Explicit ``add_edge`` calls remain available for manual graph use.
        """
        del exclude_existing
        self._ensure_loaded_authoritative()
        before = {
            tuple(sorted((source_id, target_id)))
            for source_id, source in self._nodes.items()
            for target_id in source.neighbors
            if source_id < target_id
        }
        self._preferences = {}
        self.clear_all_edges()
        effective_threshold = float(
            self._config.auto_connect_threshold if threshold is None else threshold
        )
        self._source_policies = {
            source_id: (effective_threshold, None) for source_id in self._authoritative
        }
        for source_id in sorted(self._authoritative):
            self._preferences[source_id] = self._nearest_authoritative(
                self._authoritative[source_id], threshold=effective_threshold
            )
        self._preferences_initialized = True
        self._rebuild_mutual_edges()
        after = {
            tuple(sorted((source_id, target_id)))
            for source_id, source in self._nodes.items()
            for target_id in source.neighbors
            if source_id < target_id
        }
        return len(after - before)

    def _query_policy(
        self,
        *,
        current_scope: str | None,
        scope_id: str | None,
        allowed_scopes: Iterable[str] | None,
        user_profile_id: str | None,
        project_id: str | None,
        workspace_id: str | None,
    ) -> tuple[str | None, str | None, str | None, str | None, tuple[str, ...], bool]:
        profile = self._clean_id(user_profile_id)
        current = current_scope.strip().lower() if isinstance(current_scope, str) else None
        if current is not None and current not in {"project", "workspace", "global"}:
            raise ValueError("current_scope must be project, workspace, or global")
        project = self._clean_id(project_id)
        workspace = self._clean_id(workspace_id)
        sid = self._clean_id(scope_id)
        if current == "project" and project is None:
            project = sid
        if current == "workspace" and workspace is None:
            workspace = sid
        if current == "global":
            project = None
            workspace = None
        permitted: tuple[str, ...]
        if allowed_scopes is None:
            permitted = ("project", "workspace", "global")
        else:
            permitted_list: list[str] = []
            for raw_scope in allowed_scopes:
                normalized = str(raw_scope).strip().lower()
                if normalized not in {"project", "workspace", "global"}:
                    raise ValueError("allowed_scopes contains an invalid scope")
                if normalized not in permitted_list:
                    permitted_list.append(normalized)
            permitted = tuple(permitted_list)
        legacy = (
            profile is None
            and current is None
            and sid is None
            and project is None
            and workspace is None
            and allowed_scopes is None
        )
        return current, project, workspace, profile, permitted, legacy

    def _eligible_node(
        self,
        node: ActivationNode,
        *,
        policy: tuple[str | None, str | None, str | None, str | None, tuple[str, ...], bool],
        filter_fn: NodeFilter | None,
    ) -> bool:
        current, project, workspace, profile, permitted, legacy = policy
        scope, node_id, node_profile, _ = self._scope_data(node)
        if filter_fn is not None and not filter_fn(node):
            return False
        if legacy:
            # A contextless embedded call may use an entirely legacy graph.
            # It must not turn a profiled node into implicitly shared memory.
            # A supplied predicate is the explicit compatibility boundary used
            # by the engine, and is responsible for validating its full scope.
            return node_profile is None or filter_fn is not None
        if profile is None and node_profile is None:
            if scope not in permitted:
                return False
            if scope == "global":
                return True
            if scope == "project":
                return project is not None and node_id == project
            if scope == "workspace":
                return workspace is not None and node_id == workspace
            return current == scope and node_id is not None
        if profile is None or node_profile != profile or scope not in permitted:
            return False
        if scope == "global":
            return True
        if scope == "project":
            return project is not None and node_id == project
        if scope == "workspace":
            return workspace is not None and node_id == workspace
        return current == scope and node_id is not None

    def activate(
        self,
        node_id: str,
        energy: float | None = None,
        *,
        current_scope: str | None = None,
        scope_id: str | None = None,
        allowed_scopes: Iterable[str] | None = None,
        user_profile_id: str | None = None,
        project_id: str | None = None,
        workspace_id: str | None = None,
        filter_fn: NodeFilter | None = None,
        max_hops: int | None = None,
        max_visited: int | None = None,
    ) -> dict[str, float]:
        """Activate one seed, using the configured visit budget unless overridden."""
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found in activation graph.")
        policy = self._query_policy(
            current_scope=current_scope,
            scope_id=scope_id,
            allowed_scopes=allowed_scopes,
            user_profile_id=user_profile_id,
            project_id=project_id,
            workspace_id=workspace_id,
        )
        if not self._eligible_node(self._nodes[node_id], policy=policy, filter_fn=filter_fn):
            raise KeyError(f"Node '{node_id}' is outside the permitted activation graph.")
        self._total_activations += 1
        initial = float(energy if energy is not None else self._config.initial_energy)
        if not math.isfinite(initial) or initial < 0.0:
            raise ValueError("activation energy must be finite and non-negative")
        hop_limit = max(0, int(max_hops if max_hops is not None else self._config.max_hops))
        visit_limit = max_visited
        if visit_limit is None:
            visit_limit = self._config.max_visited
        visit_limit = max(1, int(visit_limit))
        threshold = float(self._config.activation_threshold)
        attenuation = 1.0 - float(self._config.decay_rate)
        best_energy: dict[str, float] = {node_id: initial}
        frontier: deque[tuple[str, float, int]] = deque([(node_id, initial, 0)])
        discovered: set[str] = {node_id}
        expanded_energy: dict[str, float] = {}
        while frontier:
            current_id, current_energy, depth = frontier.popleft()
            if current_energy <= expanded_energy.get(current_id, -1.0):
                continue
            expanded_energy[current_id] = current_energy
            if depth >= hop_limit:
                continue
            current_node = self._nodes.get(current_id)
            if current_node is None:
                continue

            def edge_order(item: tuple[str, float]) -> tuple[float, str]:
                try:
                    weight = float(item[1])
                except (TypeError, ValueError):
                    weight = -1.0
                return (-weight, item[0])

            for neighbor_id, edge_weight in sorted(current_node.neighbors.items(), key=edge_order):
                neighbor = self._nodes.get(neighbor_id)
                if neighbor is None or not self._eligible_node(
                    neighbor, policy=policy, filter_fn=filter_fn
                ):
                    continue
                try:
                    numeric_edge_weight = float(edge_weight)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric_edge_weight) or not 0.0 <= numeric_edge_weight <= 1.0:
                    continue
                transferred = current_energy * attenuation * numeric_edge_weight
                if transferred < threshold and neighbor_id not in best_energy:
                    continue
                previous = best_energy.get(neighbor_id, -1.0)
                if transferred <= previous:
                    continue
                if neighbor_id not in discovered:
                    if len(discovered) >= visit_limit:
                        continue
                    discovered.add(neighbor_id)
                best_energy[neighbor_id] = transferred
                frontier.append((neighbor_id, transferred, depth + 1))
        include_subthreshold_seed = policy[3] is None
        return {
            nid: value
            for nid, value in best_energy.items()
            if (nid == node_id and (include_subthreshold_seed or value >= threshold))
            or value >= threshold
        }

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        *,
        current_scope: str | None = None,
        scope_id: str | None = None,
        allowed_scopes: Iterable[str] | None = None,
        user_profile_id: str | None = None,
        project_id: str | None = None,
        workspace_id: str | None = None,
        filter_fn: NodeFilter | None = None,
        seed_ids: Iterable[str] | None = None,
        max_hops: int | None = None,
        max_visited: int | None = None,
    ) -> list[ActivationNode]:
        """Retrieve a bounded associative neighborhood for a query embedding."""
        if top_k <= 0 or not self._nodes:
            return []
        policy = self._query_policy(
            current_scope=current_scope,
            scope_id=scope_id,
            allowed_scopes=allowed_scopes,
            user_profile_id=user_profile_id,
            project_id=project_id,
            workspace_id=workspace_id,
        )
        permitted_ids = {
            nid
            for nid, node in self._nodes.items()
            if self._eligible_node(node, policy=policy, filter_fn=filter_fn)
        }
        if not permitted_ids:
            return []
        seed_candidates = permitted_ids
        if seed_ids is not None:
            seed_candidates = permitted_ids.intersection({str(seed_id) for seed_id in seed_ids})
            if not seed_candidates:
                return []
        seed_id = min(
            seed_candidates,
            key=lambda nid: (
                -cosine_similarity(query_embedding, self._nodes[nid].embedding or []),
                nid,
            ),
        )
        activated = self.activate(
            seed_id,
            current_scope=current_scope,
            scope_id=scope_id,
            allowed_scopes=allowed_scopes,
            user_profile_id=user_profile_id,
            project_id=project_id,
            workspace_id=workspace_id,
            filter_fn=filter_fn,
            max_hops=max_hops,
            max_visited=max_visited,
        )
        result = [
            self._nodes[nid].model_copy(update={"energy": value})
            for nid, value in activated.items()
            if nid in permitted_ids
        ]
        result.sort(key=lambda node: (-node.energy, node.id))
        return result[:top_k]

    def decay_all(self, rate: float | None = None) -> None:
        """Decay explicit standalone node state; retrieval never uses this state."""
        decay = rate if rate is not None else self._config.natural_decay_rate
        for node in self._nodes.values():
            node.energy = max(0.0, node.energy * (1.0 - decay))

    def get_node(self, node_id: str) -> ActivationNode | None:
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> list[ActivationNode]:
        return list(self._nodes.values())

    def get_node_count(self) -> int:
        return len(self._nodes)

    def get_edge_count(self) -> int:
        return sum(len(node.neighbors) for node in self._nodes.values())

    def clear_energies(self) -> None:
        for node in self._nodes.values():
            node.energy = 0.0

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self._nodes:
            return False
        affected = {
            source_id for source_id, targets in self._preferences.items() if node_id in targets
        }
        del self._nodes[node_id]
        self._authoritative.pop(node_id, None)
        self._preferences.pop(node_id, None)
        self._source_policies.pop(node_id, None)
        for targets in self._preferences.values():
            targets.pop(node_id, None)
        if self._preferences_initialized:
            for source_id in affected:
                source = self._authoritative.get(source_id)
                if source is not None:
                    source_threshold, source_candidates = self._source_policy(source_id)
                    self._preferences[source_id] = self._nearest_authoritative(
                        source,
                        threshold=source_threshold,
                        candidate_ids=source_candidates,
                    )
            self._rebuild_mutual_edges()
        for node in self._nodes.values():
            node.neighbors.pop(node_id, None)
        return True

    def get_stats(self) -> dict[str, Any]:
        energized = sum(1 for node in self._nodes.values() if node.energy > 0)
        average_degree = (
            sum(len(node.neighbors) for node in self._nodes.values()) / len(self._nodes)
            if self._nodes
            else 0.0
        )
        return {
            "module": "SpreadingActivation",
            "algorithm_version": self.ALGORITHM_VERSION,
            "graph_version": self.GRAPH_VERSION,
            "persistence_schema_version": self.PERSISTENCE_SCHEMA_VERSION,
            "edge_builder_version": self.EDGE_BUILDER_VERSION,
            "traversal_version": self.TRAVERSAL_VERSION,
            "total_nodes": len(self._nodes),
            "total_edges": self.get_edge_count() // 2,
            "energized_nodes": energized,
            "average_degree": round(average_degree, 2),
            "total_activations": self._total_activations,
            "graph_derivation_status": "ready" if self._preferences_initialized else "pending",
            "config": {
                "initial_energy": self._config.initial_energy,
                "decay_rate": self._config.decay_rate,
                "activation_threshold": self._config.activation_threshold,
                "max_hops": self._config.max_hops,
                "max_neighbors_per_node": self._config.max_neighbors_per_node,
            },
        }
