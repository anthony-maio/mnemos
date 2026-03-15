"""
tests/test_types.py — Tests for Mnemos core types and Pydantic models.
"""

import json
from datetime import datetime, timezone

import pytest

from mnemos.types import (
    ActivationNode,
    CognitiveState,
    ConsolidationResult,
    Interaction,
    MemoryChunk,
    ProcessResult,
)

# ─── CognitiveState tests ─────────────────────────────────────────────────────


class TestCognitiveState:
    def test_default_values(self):
        """CognitiveState should have sensible neutral defaults."""
        state = CognitiveState()
        assert state.valence == 0.0
        assert state.arousal == 0.5
        assert state.complexity == 0.5

    def test_explicit_values(self):
        state = CognitiveState(valence=0.8, arousal=0.9, complexity=0.3)
        assert state.valence == 0.8
        assert state.arousal == 0.9
        assert state.complexity == 0.3

    def test_valence_bounds(self):
        """Valence must be in [-1, 1]."""
        with pytest.raises(Exception):
            CognitiveState(valence=1.5)
        with pytest.raises(Exception):
            CognitiveState(valence=-1.5)

    def test_arousal_bounds(self):
        """Arousal must be in [0, 1]."""
        with pytest.raises(Exception):
            CognitiveState(arousal=1.1)
        with pytest.raises(Exception):
            CognitiveState(arousal=-0.1)

    def test_complexity_bounds(self):
        """Complexity must be in [0, 1]."""
        with pytest.raises(Exception):
            CognitiveState(complexity=2.0)

    def test_distance_identical_states(self):
        """Distance between identical states should be 0."""
        state = CognitiveState(valence=0.5, arousal=0.5, complexity=0.5)
        assert state.distance(state) == pytest.approx(0.0, abs=1e-6)

    def test_distance_symmetric(self):
        """Distance should be symmetric: d(a, b) == d(b, a)."""
        a = CognitiveState(valence=0.8, arousal=0.1, complexity=0.9)
        b = CognitiveState(valence=-0.5, arousal=0.7, complexity=0.2)
        assert a.distance(b) == pytest.approx(b.distance(a), abs=1e-6)

    def test_distance_range(self):
        """Distance should always be in [0, 1]."""
        a = CognitiveState(valence=1.0, arousal=1.0, complexity=1.0)
        b = CognitiveState(valence=-1.0, arousal=0.0, complexity=0.0)
        dist = a.distance(b)
        assert 0.0 <= dist <= 1.0

    def test_distance_max_separation(self):
        """Maximum separation: valence +1 vs -1, arousal 1 vs 0, complexity 1 vs 0."""
        a = CognitiveState(valence=1.0, arousal=1.0, complexity=1.0)
        b = CognitiveState(valence=-1.0, arousal=0.0, complexity=0.0)
        dist = a.distance(b)
        # Maximum possible distance should normalize to 1.0
        assert dist == pytest.approx(1.0, abs=1e-6)

    def test_distance_partial_separation(self):
        """Distance between neutral and positive should be less than max."""
        neutral = CognitiveState(valence=0.0, arousal=0.5, complexity=0.5)
        positive = CognitiveState(valence=1.0, arousal=1.0, complexity=1.0)
        dist = neutral.distance(positive)
        assert 0.0 < dist < 1.0

    def test_serialization(self):
        """CognitiveState should round-trip through JSON."""
        state = CognitiveState(valence=-0.3, arousal=0.8, complexity=0.6)
        json_str = state.model_dump_json()
        restored = CognitiveState.model_validate_json(json_str)
        assert restored.valence == pytest.approx(state.valence)
        assert restored.arousal == pytest.approx(state.arousal)
        assert restored.complexity == pytest.approx(state.complexity)


# ─── MemoryChunk tests ────────────────────────────────────────────────────────


class TestMemoryChunk:
    def test_auto_id(self):
        """MemoryChunk should auto-generate a UUID id."""
        chunk = MemoryChunk(content="test content")
        assert chunk.id
        assert len(chunk.id) == 36  # UUID4 format

    def test_unique_ids(self):
        """Each chunk should get a unique ID."""
        a = MemoryChunk(content="a")
        b = MemoryChunk(content="b")
        assert a.id != b.id

    def test_default_version(self):
        """Default version should be 1."""
        chunk = MemoryChunk(content="test")
        assert chunk.version == 1

    def test_default_access_count(self):
        """Default access_count should be 0."""
        chunk = MemoryChunk(content="test")
        assert chunk.access_count == 0

    def test_salience_bounds(self):
        """Salience must be in [0, 1]."""
        with pytest.raises(Exception):
            MemoryChunk(content="test", salience=1.5)
        with pytest.raises(Exception):
            MemoryChunk(content="test", salience=-0.1)

    def test_touch_increments_access_count(self):
        """touch() should increment access_count."""
        chunk = MemoryChunk(content="test")
        assert chunk.access_count == 0
        chunk.touch()
        assert chunk.access_count == 1
        chunk.touch()
        assert chunk.access_count == 2

    def test_touch_updates_timestamp(self):
        """touch() should update updated_at."""
        chunk = MemoryChunk(content="test")
        original_updated_at = chunk.updated_at
        import time

        time.sleep(0.01)
        chunk.touch()
        assert chunk.updated_at >= original_updated_at

    def test_reconsolidate_preserves_id(self):
        """reconsolidate() should keep the same ID."""
        chunk = MemoryChunk(content="old content", salience=0.8)
        updated = chunk.reconsolidate("new content")
        assert updated.id == chunk.id

    def test_reconsolidate_increments_version(self):
        """reconsolidate() should increment version."""
        chunk = MemoryChunk(content="old content")
        assert chunk.version == 1
        updated = chunk.reconsolidate("new content")
        assert updated.version == 2

    def test_reconsolidate_updates_content(self):
        """reconsolidate() should update content."""
        chunk = MemoryChunk(content="old content")
        updated = chunk.reconsolidate("new content")
        assert updated.content == "new content"

    def test_reconsolidate_clears_embedding(self):
        """reconsolidate() should clear embedding (needs re-embedding)."""
        chunk = MemoryChunk(content="old", embedding=[0.1, 0.2, 0.3])
        updated = chunk.reconsolidate("new")
        assert updated.embedding is None

    def test_reconsolidate_preserves_previous_content_in_metadata(self):
        """reconsolidate() should store previous content in metadata."""
        chunk = MemoryChunk(content="original")
        updated = chunk.reconsolidate("updated")
        assert updated.metadata.get("previous_content") == "original"

    def test_reconsolidate_preserves_cognitive_state(self):
        """reconsolidate() should preserve cognitive state."""
        state = CognitiveState(valence=0.5, arousal=0.8, complexity=0.3)
        chunk = MemoryChunk(content="test", cognitive_state=state)
        updated = chunk.reconsolidate("new test")
        assert updated.cognitive_state is not None
        assert updated.cognitive_state.valence == 0.5

    def test_serialization_roundtrip(self):
        """MemoryChunk should round-trip through JSON."""
        state = CognitiveState(valence=0.3, arousal=0.7, complexity=0.4)
        chunk = MemoryChunk(
            content="test content",
            embedding=[0.1, 0.2, 0.3],
            salience=0.75,
            cognitive_state=state,
            metadata={"source": "test"},
            access_count=3,
            version=2,
        )
        json_str = chunk.model_dump_json()
        restored = MemoryChunk.model_validate_json(json_str)
        assert restored.id == chunk.id
        assert restored.content == chunk.content
        assert restored.salience == chunk.salience
        assert restored.access_count == chunk.access_count
        assert restored.version == chunk.version
        assert restored.embedding == chunk.embedding
        assert restored.cognitive_state.valence == chunk.cognitive_state.valence

    def test_metadata_defaults_empty_dict(self):
        """Metadata should default to an empty dict."""
        chunk = MemoryChunk(content="test")
        assert chunk.metadata == {}

    def test_with_cognitive_state(self):
        """MemoryChunk can hold a CognitiveState."""
        state = CognitiveState(valence=0.9, arousal=0.1, complexity=0.8)
        chunk = MemoryChunk(content="calm and happy", cognitive_state=state)
        assert chunk.cognitive_state.valence == 0.9
        assert chunk.cognitive_state.arousal == 0.1


# ─── Interaction tests ────────────────────────────────────────────────────────


class TestInteraction:
    def test_basic_creation(self):
        """Interaction should be creatable with role and content."""
        interaction = Interaction(role="user", content="Hello, world!")
        assert interaction.role == "user"
        assert interaction.content == "Hello, world!"

    def test_role_normalized_to_lowercase(self):
        """Role should be normalized to lowercase."""
        interaction = Interaction(role="USER", content="test")
        assert interaction.role == "user"

    def test_role_stripped(self):
        """Role should be stripped of whitespace."""
        interaction = Interaction(role="  assistant  ", content="test")
        assert interaction.role == "assistant"

    def test_empty_role_raises(self):
        """Empty role should raise a validation error."""
        with pytest.raises(Exception):
            Interaction(role="", content="test")

    def test_timestamp_is_set(self):
        """Interaction should auto-set timestamp."""
        before = datetime.now(timezone.utc)
        interaction = Interaction(role="user", content="test")
        after = datetime.now(timezone.utc)
        assert before <= interaction.timestamp <= after

    def test_metadata_defaults_empty(self):
        """Metadata should default to empty dict."""
        interaction = Interaction(role="user", content="test")
        assert interaction.metadata == {}


# ─── ProcessResult tests ──────────────────────────────────────────────────────


class TestProcessResult:
    def test_not_stored(self):
        result = ProcessResult(stored=False, salience=0.1, reason="Low surprisal.")
        assert result.stored is False
        assert result.chunk is None
        assert result.salience == 0.1
        assert "Low surprisal" in result.reason

    def test_stored_with_chunk(self):
        chunk = MemoryChunk(content="important fact")
        result = ProcessResult(stored=True, chunk=chunk, salience=0.9, reason="High surprisal.")
        assert result.stored is True
        assert result.chunk is not None
        assert result.chunk.content == "important fact"


# ─── ActivationNode tests ─────────────────────────────────────────────────────


class TestActivationNode:
    def test_auto_id(self):
        node = ActivationNode(content="test node")
        assert node.id
        assert len(node.id) == 36

    def test_default_energy(self):
        node = ActivationNode(content="test")
        assert node.energy == 0.0

    def test_default_neighbors_empty(self):
        node = ActivationNode(content="test")
        assert node.neighbors == {}

    def test_default_metadata_empty(self):
        node = ActivationNode(content="test")
        assert node.metadata == {}

    def test_with_neighbors(self):
        node = ActivationNode(
            content="test",
            neighbors={"node-b": 0.8, "node-c": 0.5},
        )
        assert "node-b" in node.neighbors
        assert node.neighbors["node-b"] == 0.8

    def test_with_metadata(self):
        node = ActivationNode(
            content="test",
            metadata={"scope": "project", "scope_id": "repo-alpha"},
        )
        assert node.metadata["scope"] == "project"
        assert node.metadata["scope_id"] == "repo-alpha"


# ─── ConsolidationResult tests ────────────────────────────────────────────────


class TestConsolidationResult:
    def test_defaults(self):
        result = ConsolidationResult()
        assert result.facts_extracted == []
        assert result.chunks_pruned == 0
        assert result.tools_generated == []
        assert result.duration_seconds == 0.0

    def test_with_data(self):
        result = ConsolidationResult(
            facts_extracted=["User prefers Python", "User works on ML"],
            chunks_pruned=15,
            tools_generated=["def my_tool(): pass"],
            duration_seconds=2.3,
        )
        assert len(result.facts_extracted) == 2
        assert result.chunks_pruned == 15
        assert len(result.tools_generated) == 1
