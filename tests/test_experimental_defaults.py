"""Fresh installations require a separate opt-in for experimental modules."""

import pytest

from mnemos.config import MnemosConfig, ModuleConfig, SleepConfig
from mnemos.engine import MnemosEngine
from mnemos.runtime import build_mnemos_config_from_settings
from mnemos.settings import AppSettings, load_settings
from mnemos.types import Interaction
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import MockLLMProvider
from mnemos.utils.storage import InMemoryStore


@pytest.mark.parametrize("provider", ["mock", "ollama", "openai"])
def test_provider_selection_does_not_enable_experiments(provider):
    settings = AppSettings(llm={"provider": provider})
    config = build_mnemos_config_from_settings(settings)
    assert not any(settings.modules.model_dump().values())
    assert not any(config.modules.model_dump().values())
    assert not any(MnemosConfig().modules.model_dump().values())


def test_explicit_module_settings_and_env_overrides_survive_defaults(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text("[modules]\nsleep = true\nspreading = true\n", encoding="utf-8")
    settings = load_settings(
        env={"MNEMOS_MODULE_SPREADING": "false", "MNEMOS_MODULE_AFFECTIVE": "true"},
        cwd=tmp_path,
        global_config_path=config,
    ).settings
    assert settings.modules.model_dump() == {
        "surprisal": False,
        "mutable_rag": False,
        "affective": True,
        "sleep": True,
        "spreading": False,
    }
    assert config.read_text(encoding="utf-8") == "[modules]\nsleep = true\nspreading = true\n"


class RecordingLLM(MockLLMProvider):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def predict(self, prompt, **kwargs):
        self.calls.append("predict")
        return await super().predict(prompt, **kwargs)

    async def classify(self, prompt, labels):
        self.calls.append("classify")
        return await super().classify(prompt, labels)


@pytest.mark.asyncio
async def test_embedded_disabled_modules_never_call_model_or_collect_episodes():
    llm = RecordingLLM()
    engine = MnemosEngine(
        config=MnemosConfig(
            modules=ModuleConfig(**{name: False for name in ModuleConfig.model_fields})
        ),
        llm=llm,
        embedder=SimpleEmbeddingProvider(),
        store=InMemoryStore(),
    )
    automatic = await engine.process(Interaction(role="user", content="Use SQLite for the cache"))
    assert automatic.stored is False
    assert "disabled" in automatic.reason
    remembered = await engine.remember("Use SQLite for the cache")
    assert remembered.stored is True
    assert remembered.chunk.cognitive_state is None
    assert (await engine.consolidate()).facts_extracted == []
    assert engine.sleep_daemon.get_episode_count() == 0
    assert llm.calls == []


@pytest.mark.asyncio
async def test_disabled_consolidation_keeps_preexisting_evidence_without_model_calls():
    llm = RecordingLLM()
    engine = MnemosEngine(
        config=MnemosConfig(
            sleep=SleepConfig(min_episodes_before_consolidation=1, consolidation_interval_seconds=0)
        ),
        llm=llm,
        embedder=SimpleEmbeddingProvider(),
        store=InMemoryStore(),
    )
    engine.sleep_daemon.add_episode(Interaction(role="user", content="Use SQLite for the cache"))
    assert engine.sleep_daemon.get_episode_count() == 1
    result = await engine.consolidate()
    assert result.facts_extracted == []
    assert engine.store.get_all() == []
    assert engine.sleep_daemon.get_episode_count() == 1
    assert llm.calls == []


@pytest.mark.asyncio
async def test_managed_default_explicit_writes_need_no_model(tmp_path):
    from mnemos.memory_service import LocalMemoryService
    from mnemos.scopes import ScopeRegistry
    from mnemos.utils.storage import SQLiteStore

    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="local", enabled=True)
    context = registry.resolve(
        tmp_path, user_profile_id="local", session_id="defaults", host="manual"
    )
    llm = RecordingLLM()
    engine = MnemosEngine(
        llm=llm, embedder=SimpleEmbeddingProvider(), store=SQLiteStore(str(tmp_path / "memory.db"))
    )
    memory = LocalMemoryService(engine, registry)
    try:
        receipt = await memory.dispatch(
            "remember", {"context": context.model_dump(), "content": "Use SQLite for the cache"}
        )
        assert receipt["status"] == "committed"
        assert await memory.maintenance.run_once(force=True) == 1
        fact = engine.store.get_all()[0]
        assert fact.content == "Use SQLite for the cache"
        assert engine.store.list_fact_evidence(fact.id)
        assert fact.cognitive_state is None
        assert llm.calls == []
    finally:
        engine.store.close()
