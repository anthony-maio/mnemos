"""Affect through the judge, and a recall path that never waits on it."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mnemos.config import MnemosConfig, SurprisalConfig
from mnemos.engine import MnemosEngine
from mnemos.judgment import MockJudge, Score, ScoreAnswer, UnavailableJudge
from mnemos.modules.affective import AffectiveRouter
from mnemos.retrieval import RetrievalConfig
from mnemos.runtime import build_configured_engine
from mnemos.types import Interaction
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import MockLLMProvider, UnavailableLLMProvider
from mnemos.utils.storage import InMemoryStore


def _scores(valence: int, arousal: int, complexity: int) -> MockJudge:
    levels = {"valence": valence, "arousal": arousal, "complexity": complexity}

    def script(key, question, state):
        assert isinstance(question, Score)
        index = levels[key]
        return ScoreAnswer(
            value=float(index),
            confidence=0.8,
            probabilities={str(i): (1.0 if i == index else 0.0) for i in range(5)},
        )

    return MockJudge(script)


def test_the_router_asks_the_judge_when_one_is_configured() -> None:
    class ExplodingLLM(UnavailableLLMProvider):
        async def predict(self, prompt: str, **kwargs: object) -> str:
            raise AssertionError("the prompt classifier must not run with a judge")

    judge = _scores(valence=0, arousal=4, complexity=2)
    router = AffectiveRouter(llm=ExplodingLLM(), embedder=SimpleEmbeddingProvider(), judge=judge)

    state = asyncio.run(router.classify_state(Interaction(role="user", content="Prod is down!")))

    assert state is not None
    assert (state.valence, state.arousal, state.complexity) == (-1.0, 1.0, 0.5)
    assert judge.calls[0][2] == "classify_state"
    assert router.get_current_state({}).arousal == 1.0


def test_a_judge_outage_yields_no_state_like_a_provider_outage() -> None:
    router = AffectiveRouter(
        llm=UnavailableLLMProvider(),
        embedder=SimpleEmbeddingProvider(),
        judge=UnavailableJudge(),
    )
    assert asyncio.run(router.classify_state(Interaction(role="user", content="x"))) is None


@pytest.mark.asyncio
async def test_recent_state_recall_makes_no_classification_call() -> None:
    judge = _scores(valence=4, arousal=0, complexity=0)
    engine = MnemosEngine(
        config=MnemosConfig(
            surprisal=SurprisalConfig(threshold=0.0, min_content_length=0),
            retrieval=RetrievalConfig(profile="biomimetic", query_state_source="recent_state"),
        ),
        llm=MockLLMProvider(),
        embedder=SimpleEmbeddingProvider(dim=64),
        store=InMemoryStore(),
        judge=judge,
    )
    await engine.process(
        Interaction(role="user", content="Use uv for Python tooling"),
        scope="project",
        scope_id="repo-alpha",
    )
    await engine.process(
        Interaction(role="user", content="Deploys go out on Thursdays"),
        scope="project",
        scope_id="repo-alpha",
    )
    encoded_calls = len(judge.calls)
    assert encoded_calls >= 1

    results = await engine.retrieve(
        "python tooling",
        top_k=2,
        reconsolidate=False,
        current_scope="project",
        scope_id="repo-alpha",
        allowed_scopes=("project",),
    )

    assert len(judge.calls) == encoded_calls
    assert results
    assert results[0].metadata["retrieval_trace"]["state_source"] == "recent_state"


def test_a_hosted_judge_moves_recall_off_the_round_trip(tmp_path: Path) -> None:
    engine = build_configured_engine(
        env={
            "MNEMOS_CONFIG_PATH": str(tmp_path / "missing.toml"),
            "MNEMOS_SQLITE_PATH": str(tmp_path / "memory.db"),
            "MNEMOS_LLM_PROVIDER": "ollama",
            "MNEMOS_JUDGMENT_PROVIDER": "typesafe",
            "MNEMOS_TYPESAFE_API_KEY": "k",
        },
        cwd=tmp_path,
    )
    try:
        assert engine.config.retrieval.query_state_source == "recent_state"
    finally:
        asyncio.run(engine.aclose())

    local = build_configured_engine(
        env={
            "MNEMOS_CONFIG_PATH": str(tmp_path / "missing.toml"),
            "MNEMOS_SQLITE_PATH": str(tmp_path / "memory2.db"),
            "MNEMOS_LLM_PROVIDER": "ollama",
            "MNEMOS_JUDGMENT_PROVIDER": "ollama",
        },
        cwd=tmp_path,
    )
    try:
        assert local.config.retrieval.query_state_source == "classify"
    finally:
        asyncio.run(local.aclose())
