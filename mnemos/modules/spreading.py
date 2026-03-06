"""
mnemos/modules/spreading.py — Spreading Activation (Energy-Based Graph RAG).

Neuroscience Basis:
    Collins & Loftus (1975) proposed that semantic memory is organized as a
    network of concepts linked by typed associations. Memory retrieval works
    via 'spreading activation': when a concept is activated (recalled), that
    activation energy propagates along semantic links to related concepts,
    pre-activating (priming) them for rapid subsequent recall.

    For example, hearing "doctor" activates "hospital", "nurse", "medicine",
    "illness" — even without explicitly querying for them. This associative
    priming is why human recall has a 'train of thought' quality that pure
    vector search lacks.

    Key properties of neural spreading activation:
    - Energy decays with each synaptic hop (typically 20-50% per hop)
    - Propagation is limited in depth (only 3-5 meaningful hops in humans)
    - Nodes below a threshold are not 'primed' (threshold prevents noise)
    - Energy decays naturally over time (forgetting; cortical inhibition)

Mnemos Implementation:
    Memory chunks are nodes in an activation graph. Edges connect semantically
    similar nodes (cosine similarity above a threshold). When a query arrives:
    1. Find the closest node by embedding similarity (initial lookup)
    2. Inject activation energy at that node
    3. Let energy spread BFS-style through edges, decaying by decay_rate each hop
    4. Return all nodes with energy above activation_threshold

    This creates a fluid 'spotlight' of context — the LLM gets not just the
    exact match, but the full associative neighborhood, enabling human-like
    contextual recall without requiring multi-million token windows.
"""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from typing import Any

from ..config import SpreadingConfig
from ..types import ActivationNode, MemoryChunk
from ..utils.embeddings import EmbeddingProvider, cosine_similarity


class SpreadingActivation:
    """
    Graph-based memory retrieval with energy-based spreading activation.

    Maintains a graph of ActivationNodes where edges represent semantic
    associations. Energy injected at a seed node propagates through the
    graph, with each hop attenuating the signal — just as neural activation
    energy dissipates across synapses.

    Args:
        embedder: Embedding provider for similarity-based edge construction.
        config: SpreadingConfig controlling energy propagation parameters.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        config: SpreadingConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._config = config or SpreadingConfig()
        # Node storage: node_id → ActivationNode
        self._nodes: dict[str, ActivationNode] = {}
        # Stats
        self._total_activations: int = 0

    def add_node(
        self,
        content: str,
        embedding: list[float] | None = None,
        node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ActivationNode:
        """
        Add a new concept node to the activation graph.

        If embedding is not provided, it is computed from the content.
        After adding, call auto_connect() to build edges based on similarity.

        Args:
            content: The concept/memory content for this node.
            embedding: Optional pre-computed embedding. Computed if not provided.
            node_id: Optional explicit ID. Auto-generated UUID if not provided.
            metadata: Optional metadata dict (stored in ActivationNode, not in the type
                      but tracked in the internal metadata dict).

        Returns:
            The created ActivationNode.
        """
        if embedding is None:
            embedding = self._embedder.embed(content)

        nid = node_id or str(uuid.uuid4())
        node = ActivationNode(
            id=nid,
            content=content,
            energy=0.0,
            neighbors={},
            embedding=embedding,
        )
        self._nodes[nid] = node
        return node

    def add_node_from_chunk(self, chunk: MemoryChunk) -> ActivationNode:
        """
        Add a MemoryChunk as a node in the activation graph.

        Uses the chunk's existing embedding (or computes one if absent).
        The chunk's ID becomes the node's ID, enabling cross-referencing
        between the activation graph and the memory store.

        Args:
            chunk: The MemoryChunk to add as a graph node.

        Returns:
            The created ActivationNode.
        """
        embedding = chunk.embedding or self._embedder.embed(chunk.content)
        return self.add_node(
            content=chunk.content,
            embedding=embedding,
            node_id=chunk.id,
        )

    def add_edge(
        self,
        node_a_id: str,
        node_b_id: str,
        weight: float,
        bidirectional: bool = True,
    ) -> bool:
        """
        Add a weighted edge between two nodes.

        Edges represent semantic associations. Weight controls how much
        activation energy flows across this edge during propagation.
        Higher weight = stronger association = more energy transfer.

        Args:
            node_a_id: ID of the first node.
            node_b_id: ID of the second node.
            weight: Edge weight in [0, 1].
            bidirectional: If True, adds edges in both directions (default).

        Returns:
            True if both nodes exist and edge was added, False otherwise.
        """
        if node_a_id not in self._nodes or node_b_id not in self._nodes:
            return False
        if node_a_id == node_b_id:
            return False

        self._nodes[node_a_id].neighbors[node_b_id] = weight
        if bidirectional:
            self._nodes[node_b_id].neighbors[node_a_id] = weight
        return True

    def auto_connect(
        self,
        threshold: float | None = None,
        exclude_existing: bool = True,
    ) -> int:
        """
        Automatically connect nodes whose embeddings exceed a similarity threshold.

        Scans all pairs of nodes and creates edges where cosine similarity
        exceeds the threshold. This builds the initial semantic graph structure.

        For N nodes this is O(N²) — use sparingly on large graphs or restrict
        to newly-added nodes.

        Args:
            threshold: Minimum cosine similarity for connection.
                       Defaults to config.auto_connect_threshold.
            exclude_existing: Skip pairs that already have an edge (default True).

        Returns:
            Number of new edges created.
        """
        thresh = threshold if threshold is not None else self._config.auto_connect_threshold
        node_list = list(self._nodes.values())
        edges_added = 0

        for i, node_a in enumerate(node_list):
            for node_b in node_list[i + 1 :]:
                if exclude_existing and node_b.id in node_a.neighbors:
                    continue
                if node_a.embedding is None or node_b.embedding is None:
                    continue
                sim = cosine_similarity(node_a.embedding, node_b.embedding)
                if sim >= thresh:
                    node_a.neighbors[node_b.id] = sim
                    node_b.neighbors[node_a.id] = sim
                    edges_added += 1

        return edges_added

    def activate(self, node_id: str, energy: float | None = None) -> dict[str, float]:
        """
        Inject activation energy at a node and propagate through the graph.

        Propagation algorithm (BFS-based):
        1. Set seed node energy = initial_energy
        2. For each neighbor at each hop:
           neighbor_energy += current_energy × (1 - decay_rate) × edge_weight
        3. Only propagate to nodes whose received energy exceeds threshold
        4. Continue until max_hops reached or no nodes above threshold remain

        This is a simplified version of the biologically-inspired spreading
        activation: energy flows from the seed, decaying with each hop,
        creating a gradient of activation across the semantic neighborhood.

        Args:
            node_id: ID of the seed node to activate.
            energy: Initial energy to inject (defaults to config.initial_energy).

        Returns:
            Dict mapping node_id → final energy for all activated nodes.

        Raises:
            KeyError: If node_id is not in the graph.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found in activation graph.")

        self._total_activations += 1
        initial = energy if energy is not None else self._config.initial_energy

        # Reset all energies
        for node in self._nodes.values():
            node.energy = 0.0

        # BFS propagation
        # Queue items: (node_id, energy, hop_depth)
        queue: deque[tuple[str, float, int]] = deque()
        queue.append((node_id, initial, 0))
        self._nodes[node_id].energy = initial

        # Track visited to avoid infinite loops in cyclic graphs
        visited: set[str] = {node_id}
        activated: dict[str, float] = {node_id: initial}

        while queue:
            current_id, current_energy, depth = queue.popleft()

            if depth >= self._config.max_hops:
                continue

            current_node = self._nodes.get(current_id)
            if current_node is None:
                continue

            for neighbor_id, edge_weight in current_node.neighbors.items():
                # Energy transferred to this neighbor
                transferred_energy = current_energy * (1.0 - self._config.decay_rate) * edge_weight

                if transferred_energy < self._config.activation_threshold:
                    continue

                if neighbor_id in self._nodes:
                    # Accumulate (a node can receive energy from multiple paths)
                    self._nodes[neighbor_id].energy += transferred_energy
                    final_energy = self._nodes[neighbor_id].energy

                    if final_energy > self._config.activation_threshold:
                        activated[neighbor_id] = final_energy

                    # Only continue BFS from this node if not yet visited
                    # at this depth (prevents exponential blowup)
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, transferred_energy, depth + 1))

        return activated

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[ActivationNode]:
        """
        Retrieve nodes using spreading activation from the closest match.

        Algorithm:
        1. Find the node with highest cosine similarity to the query embedding
        2. Inject activation energy at that seed node
        3. Return all nodes with energy above threshold, sorted by energy

        Args:
            query_embedding: Embedding of the current query.
            top_k: Maximum number of activated nodes to return.

        Returns:
            List of ActivationNodes sorted by descending activation energy.
            Returns empty list if graph is empty.
        """
        if not self._nodes:
            return []

        # Step 1: Find seed node by embedding similarity
        best_sim = -1.0
        best_node_id = None

        for node in self._nodes.values():
            if node.embedding is None:
                continue
            sim = cosine_similarity(query_embedding, node.embedding)
            if sim > best_sim:
                best_sim = sim
                best_node_id = node.id

        if best_node_id is None:
            return []

        # Step 2: Spread activation from seed node
        activated = self.activate(best_node_id)

        # Step 3: Sort by energy and return top_k
        activated_nodes = [self._nodes[nid] for nid in activated if nid in self._nodes]
        activated_nodes.sort(key=lambda n: n.energy, reverse=True)
        return activated_nodes[:top_k]

    def decay_all(self, rate: float | None = None) -> None:
        """
        Apply natural forgetting by decaying all node energies.

        Analogous to cortical inhibition and passive forgetting:
        without reinforcement, activation fades over time.
        Call this periodically to prevent stale activation states.

        Args:
            rate: Decay rate in [0, 1]. Defaults to config.natural_decay_rate.
                  A rate of 0.05 reduces all energies by 5%.
        """
        decay = rate if rate is not None else self._config.natural_decay_rate
        for node in self._nodes.values():
            node.energy = max(0.0, node.energy * (1.0 - decay))

    def get_node(self, node_id: str) -> ActivationNode | None:
        """Retrieve a node by ID."""
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> list[ActivationNode]:
        """Return all nodes in the graph."""
        return list(self._nodes.values())

    def get_node_count(self) -> int:
        """Return the total number of nodes."""
        return len(self._nodes)

    def get_edge_count(self) -> int:
        """Return the total number of directed edges (bidirectional edges counted twice)."""
        return sum(len(n.neighbors) for n in self._nodes.values())

    def clear_energies(self) -> None:
        """Reset all node energies to zero."""
        for node in self._nodes.values():
            node.energy = 0.0

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and all edges pointing to it.

        Args:
            node_id: ID of the node to remove.

        Returns:
            True if node existed and was removed, False otherwise.
        """
        if node_id not in self._nodes:
            return False

        del self._nodes[node_id]

        # Remove all incoming edges from other nodes
        for node in self._nodes.values():
            node.neighbors.pop(node_id, None)

        return True

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about the spreading activation graph."""
        energized = sum(1 for n in self._nodes.values() if n.energy > 0)
        avg_degree = (
            sum(len(n.neighbors) for n in self._nodes.values()) / len(self._nodes)
            if self._nodes
            else 0.0
        )
        return {
            "module": "SpreadingActivation",
            "total_nodes": len(self._nodes),
            "total_edges": self.get_edge_count() // 2,  # Undirected count
            "energized_nodes": energized,
            "average_degree": round(avg_degree, 2),
            "total_activations": self._total_activations,
            "config": {
                "initial_energy": self._config.initial_energy,
                "decay_rate": self._config.decay_rate,
                "activation_threshold": self._config.activation_threshold,
                "max_hops": self._config.max_hops,
            },
        }
