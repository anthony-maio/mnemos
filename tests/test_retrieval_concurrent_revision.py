"""Retrieval must score the revision that survived asynchronous classification."""

import asyncio
from pathlib import Path

import pytest

from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.types import CognitiveState, Interaction
from mnemos.utils.embeddings import EmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore


class RevisionEmbeddings(EmbeddingProvider):
    @property
    def dim(self) -> int:
        return 2

    def embed(self, text: str) -> list[float]:
        if text.startswith("Unrelated"):
            return [0.0, 1.0]
        if text.startswith("Fallback"):
            return [0.8, 0.6]
        return [1.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


@pytest.mark.asyncio
@pytest.mark.parametrize("spreading", [False, True])
@pytest.mark.parametrize(
    "replacement",
    ["Unrelated gardening convention", "Python " * 300, None],
    ids=["irrelevant", "over-budget", "forgotten"],
)
async def test_revision_during_state_classification_is_ranked_and_budgeted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spreading: bool,
    replacement: str | None,
) -> None:
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="local", enabled=True)
    context = registry.resolve(tmp_path, user_profile_id="local", session_id="s", host="codex")
    store = SQLiteStore(str(tmp_path / "memory.sqlite"))
    engine = MnemosEngine(llm=UnavailableLLMProvider(), embedder=RevisionEmbeddings(), store=store)
    engine.config.retrieval.spreading_enabled = spreading
    memory = LocalMemoryService(engine, registry)
    payload = {"context": context.model_dump(mode="json")}
    entered, release = asyncio.Event(), asyncio.Event()
    task = None
    try:
        for event_id, content in [("target", "Python convention"), ("fallback", "Fallback Python")]:
            await memory.dispatch("remember", {**payload, "event_id": event_id, "content": content})
        await memory.dispatch("maintenance", payload)
        target = next(fact for fact in store.get_all() if fact.content == "Python convention")
        fallback = next(fact for fact in store.get_all() if fact.content == "Fallback Python")

        async def classify(interaction: Interaction) -> CognitiveState:
            if interaction.content == "Python":
                entered.set()
                await release.wait()
            return CognitiveState(valence=0, arousal=0, complexity=0)

        monkeypatch.setattr(engine.affective_router, "classify_state", classify)
        task = asyncio.create_task(
            memory.dispatch(
                "retrieve",
                {
                    **payload,
                    "query": "Python",
                    "profile": "biomimetic",
                    "top_k": 1,
                    "max_context_tokens": 300,
                },
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        if replacement is None:
            forgotten = await memory.dispatch(
                "forget",
                {**payload, "fact_id": target.id, "expected_version": target.version},
            )
            assert forgotten["forgotten"] is True
            revised = None
        else:
            await memory.dispatch(
                "correct",
                {
                    **payload,
                    "event_id": "correction",
                    "fact_id": target.id,
                    "expected_version": target.version,
                    "content": replacement,
                },
            )
            await memory.dispatch("maintenance", payload)
            revised = store.get(target.id)
            assert revised is not None
            assert revised.version == 2
            assert revised.content == replacement
        release.set()
        result = await asyncio.wait_for(task, timeout=5)
        assert result["context_token_upper_bound"] <= 300
        assert [fact["id"] for fact in result["facts"]] == [fallback.id]
        assert result["facts"][0]["scores"]["semantic"] == pytest.approx(0.8)
        if revised is None:
            assert store.get(target.id) is None
        else:
            after = store.get(target.id)
            assert after is not None
            assert after.access_count == revised.access_count
    finally:
        release.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        store.close()
