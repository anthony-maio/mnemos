from datetime import datetime, timezone
from pathlib import Path
import pytest
from mnemos.config import SleepConfig
from mnemos.modules.sleep import SleepDaemon
from mnemos.types import Interaction
from mnemos.utils.storage import InMemoryStore, SQLiteStore
from mnemos.utils.llm import MockLLMProvider
from mnemos.utils.embeddings import SimpleEmbeddingProvider


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_unprofiled_interaction_ids_cannot_collide_with_content(tmp_path, backend):
    store = InMemoryStore() if backend == "memory" else SQLiteStore(str(tmp_path / "delimiters.db"))
    daemon = SleepDaemon(store, SleepConfig())
    try:
        daemon.add_episode(Interaction(id="one|two", role="user", content="three"))
        daemon.add_episode(Interaction(id="one", role="user", content="two|three"))
        events = store.list_episodic_events(status="pending")
        assert len(events) == 2
        assert {e.content for e in events} == {"three", "two|three"}
    finally:
        if isinstance(store, SQLiteStore):
            store.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("source", ["manual", "claude_hook"])
def test_episode_identity_does_not_cross_profiles(tmp_path, backend, source):
    store = InMemoryStore() if backend == "memory" else SQLiteStore(str(tmp_path / "profiles.db"))
    daemon = SleepDaemon(store, SleepConfig())
    original = Interaction(
        role="user",
        content="A shared fixture",
        metadata={
            "source": source,
            "user_profile_id": "profile-A",
            "scope": "project",
            "scope_id": "repo",
        },
    )
    other = original.model_copy(deep=True)
    other.metadata["user_profile_id"] = "profile-B"
    try:
        daemon.add_episode(original)
        daemon.add_episode(other)
        events = store.list_episodic_events(status="pending")
        assert len(events) == 2
        assert {e.metadata["user_profile_id"] for e in events} == {"profile-A", "profile-B"}
    finally:
        if isinstance(store, SQLiteStore):
            store.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_failed_persistence_does_not_leave_buffered_episode(tmp_path, monkeypatch, backend):
    store = InMemoryStore() if backend == "memory" else SQLiteStore(str(tmp_path / "failed.db"))
    daemon = SleepDaemon(store, SleepConfig())

    def fail(event):
        raise RuntimeError("injected persistence failure")

    monkeypatch.setattr(store, "append_episodic_event", fail)
    try:
        with pytest.raises(RuntimeError, match="injected persistence"):
            daemon.add_episode(Interaction(role="user", content="Synthetic fact"))
        assert daemon.get_episode_count() == 0
    finally:
        if isinstance(store, SQLiteStore):
            store.close()


def test_serialized_retry_after_restart_retains_one_episode(tmp_path):
    path = tmp_path / "restart.db"
    original = Interaction(
        role="user",
        content="Synthetic persisted episode",
        metadata={"user_profile_id": "profile", "scope": "project", "scope_id": "repo"},
    )
    serialized = original.model_dump_json()
    store = SQLiteStore(str(path))
    try:
        SleepDaemon(store, SleepConfig()).add_episode(original)
        event_id = store.list_episodic_events(status="pending")[0].id
    finally:
        store.close()
    reopened = SQLiteStore(str(path))
    try:
        SleepDaemon(reopened, SleepConfig()).add_episode(
            Interaction.model_validate_json(serialized)
        )
        events = reopened.list_episodic_events(status="pending")
        assert len(events) == 1 and events[0].id == event_id
    finally:
        reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("source", ["manual", "claude_hook"])
async def test_episode_identity_preserves_distinct_turns_and_deduplicates_retries(
    tmp_path: Path, backend: str, source: str
) -> None:
    store = InMemoryStore() if backend == "memory" else SQLiteStore(str(tmp_path / "events.db"))
    config = SleepConfig(
        recall_gated_plasticity_enabled=True,
        recall_min_supporting_episodes=2,
        recall_similarity_threshold=0.999,
    )
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = Interaction(
        role="user",
        content="Team uses uv for Python tooling.",
        timestamp=timestamp,
        metadata={"source": source, "hook_event": "UserPromptSubmit"},
    )
    second = Interaction(
        role=first.role, content=first.content, timestamp=timestamp, metadata=first.metadata
    )
    daemon = SleepDaemon(store, config)
    try:
        daemon.add_episode(first)
        daemon.add_episode(Interaction.model_validate_json(first.model_dump_json()))
        daemon.add_episode(second)
        expected_episodes = 2 if source == "manual" else 1
        assert len(store.list_episodic_events(status="pending")) == expected_episodes

        result = await daemon.consolidate(
            MockLLMProvider(responses={"extract permanent facts": "1. " + first.content}),
            SimpleEmbeddingProvider(dim=64),
        )
        assert result.facts_extracted == ([first.content] if source == "manual" else [])
        assert result.chunks_pruned == expected_episodes
        assert daemon.get_episode_count() == 0
        assert SleepDaemon(store, config).get_episode_count() == 0
    finally:
        if isinstance(store, SQLiteStore):
            store.close()
