"""Backend-independent retrieval boundaries and deterministic tied candidates."""

from __future__ import annotations

import pytest

from mnemos.types import MemoryChunk
from mnemos.utils.storage import InMemoryStore, SQLiteStore


@pytest.fixture(params=["memory", "sqlite-native", "sqlite-fallback"])
def store(request, tmp_path):
    if request.param == "memory":
        backend = InMemoryStore()
    else:
        backend = SQLiteStore(str(tmp_path / "retrieval.sqlite"))
        if request.param == "sqlite-native" and not backend._sqlite_vec_enabled:
            backend.close()
            pytest.skip("sqlite-vec is unavailable")
        if request.param == "sqlite-fallback":
            backend._sqlite_vec_enabled = False
    try:
        yield backend
    finally:
        if isinstance(backend, SQLiteStore):
            backend.close()


def test_equal_scores_have_stable_ids_after_reinsertion_and_scope_filtering(store):
    def allowed(chunk):
        return chunk.metadata.get("user_profile_id") == "local"

    allowed._mnemos_scope_filter = {
        "user_profile_id": "local",
        "project_id": "repo",
        "allowed_scopes": ["project"],
    }
    chunks = [
        MemoryChunk(
            id=identity,
            content=identity,
            embedding=[1.0, 0.0],
            metadata={"user_profile_id": profile, "scope": "project", "scope_id": "repo"},
        )
        for identity, profile in [("z", "local"), ("a", "local"), ("0-foreign", "other")]
    ]
    for chunk in chunks:
        store.store(chunk)
    for _ in range(2):
        assert [chunk.id for chunk in store.retrieve([1.0, 0.0], 2)] == ["0-foreign", "a"]
        assert [chunk.id for chunk in store.retrieve([1.0, 0.0], 1, allowed)] == ["a"]
        assert [chunk.id for chunk in store.retrieve([1.0, 0.0], 2, allowed)] == ["a", "z"]
        assert store.delete("a")
        store.store(chunks[1])


@pytest.mark.parametrize("top_k", [0, -1])
def test_nonpositive_limits_return_no_candidates_without_evaluating_predicate(store, top_k):
    for identity in ("a", "b"):
        store.store(MemoryChunk(id=identity, content=identity, embedding=[1.0, 0.0]))

    def unexpected_filter(chunk):
        pytest.fail("An empty retrieval request must not evaluate candidate predicates")

    assert store.retrieve([1.0, 0.0], top_k, unexpected_filter) == []


def test_empty_query_does_not_return_arbitrary_memories(store):
    store.store(MemoryChunk(id="a", content="Keep private context", embedding=[1.0, 0.0]))
    assert store.retrieve([], 1) == []
