from datetime import datetime, timedelta, timezone

import pytest

from mnemos.config import MnemosConfig, SurprisalConfig
from mnemos.engine import MnemosEngine
from mnemos.modules.sleep import SleepDaemon
from mnemos.modules.surprisal import SurprisalGate
from mnemos.types import Interaction
from mnemos.utils.embeddings import EmbeddingProvider
from mnemos.utils.llm import MockLLMProvider, UnavailableLLMProvider
from mnemos.utils.storage import InMemoryStore


class OppositeEmbedding(EmbeddingProvider):
    @property
    def dim(self):
        return 2

    def embed(self, text):
        return [1.0, 0.0] if text == "prediction" else [-1.0, 0.0]

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]


def gate(threshold=0.3):
    llm = MockLLMProvider(responses={"predict what": "prediction"}, record_calls=True)
    return SurprisalGate(
        llm,
        OppositeEmbedding(),
        InMemoryStore(),
        SurprisalConfig(threshold=threshold, min_content_length=0),
    )


@pytest.mark.asyncio
async def test_raw_distance_can_reach_two_while_salience_is_clamped():
    result = await gate().process(Interaction(role="user", content="opposite"))
    assert result.stored
    assert result.salience == 1
    assert result.chunk.surprisal_score == 2
    assert result.diagnostics["prediction"] == "prediction"
    boundary = await gate(2).process(Interaction(role="user", content="opposite"))
    assert not boundary.stored
    assert boundary.salience == 1
    assert boundary.diagnostics["surprisal_score"] == 2


@pytest.mark.asyncio
async def test_explicit_remember_survives_low_surprise_and_missing_model():
    engine = MnemosEngine(
        llm=UnavailableLLMProvider(),
        embedder=OppositeEmbedding(),
        store=InMemoryStore(),
        config=MnemosConfig(surprisal=SurprisalConfig(threshold=2)),
    )
    result = await engine.remember("Use uv.")
    assert result.stored
    assert result.chunk.content == "Use uv."
    assert result.chunk.surprisal_score is None
    assert result.diagnostics["prediction_mode"] == "bypassed"
    assert "explicit remember" in result.reason.lower()
    assert not (await engine.remember("api_key=should_never_persist")).stored


@pytest.mark.asyncio
async def test_cached_prediction_excludes_the_scored_event_and_is_auditable():
    instance = gate()
    when = datetime.now(timezone.utc)
    metadata = {"scope": "project", "scope_id": "repo", "session_id": "s"}
    instance.add_to_history(
        Interaction(role="user", content="Earlier evidence", timestamp=when, metadata=metadata)
    )
    incoming = Interaction(
        role="user",
        content="CURRENT_EVENT_CANARY",
        timestamp=when + timedelta(seconds=1),
        metadata=metadata,
    )
    assert await instance.precompute_prediction(incoming)
    result = await instance.process(incoming)
    assert result.diagnostics["prediction_mode"] == "cached"
    assert len(instance._llm.calls) == 1
    assert "Earlier evidence" in instance._llm.calls[0]
    assert "CURRENT_EVENT_CANARY" not in instance._llm.calls[0]


@pytest.mark.asyncio
async def test_standalone_module_firewall_runs_before_models_and_persistence():
    instance = gate()
    dangerous = Interaction(role="user", content="Keep api_key=neverpersist in memory")
    assert not (await instance.process(dangerous)).stored
    instance.add_to_history(dangerous)
    assert not instance._llm.calls
    assert not instance.get_history()
    assert not instance._store.get_all()
    daemon = SleepDaemon(instance._store)
    daemon.add_episode(dangerous)
    assert not instance._store.list_episodic_events()


@pytest.mark.asyncio
async def test_bare_engine_does_not_generate_mock_preferences():
    engine = MnemosEngine()
    await engine.process(Interaction(role="user", content="This project uses a local database"))
    result = await engine.consolidate()
    assert result.facts_extracted == []
    assert engine.store.get_all() == []
