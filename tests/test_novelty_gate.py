"""Novelty mode: the judge's probability is the surprisal score; outages wait."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mnemos.config import SurprisalConfig
from mnemos.contracts import CaptureEvent
from mnemos.engine import MnemosEngine
from mnemos.judgment import MockJudge, Noul, UnavailableJudge
from mnemos.memory_service import LocalMemoryService
from mnemos.modules.surprisal import SurprisalGate
from mnemos.scopes import ScopeRegistry
from mnemos.types import Interaction
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import InMemoryStore, SQLiteStore


def _interaction(text: str) -> Interaction:
    return Interaction(
        role="user",
        content=text,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        metadata={"user_profile_id": "u", "scope": "project", "scope_id": "p", "session_id": "s"},
    )


def test_novelty_mode_asks_one_question_and_never_calls_the_llm() -> None:
    class ExplodingLLM(UnavailableLLMProvider):
        async def predict(self, prompt: str, **kwargs: object) -> str:
            raise AssertionError("prediction must not run in novelty mode")

    judge = MockJudge({"novel": 0.82})
    gate = SurprisalGate(
        llm=ExplodingLLM(),
        embedder=SimpleEmbeddingProvider(),
        store=InMemoryStore(),
        config=SurprisalConfig(mode="novelty"),
        judge=judge,
    )

    score, status, prediction, mode = asyncio.run(
        gate._compute_surprisal(
            _interaction("Deploys move to Thursdays."), ["Deploys are on Fridays."]
        )
    )

    assert (score, status, prediction, mode) == (0.82, "measured", None, "novelty_judgment")
    state, questions, purpose = judge.calls[0]
    assert purpose == "judge_novelty"
    assert isinstance(questions["novel"], Noul)
    assert state["existing_facts"] == [{"id": "f0", "text": "Deploys are on Fridays."}]
    assert state["newest_turn"]["text"] == "Deploys move to Thursdays."


def test_a_judge_outage_reports_provider_unavailable() -> None:
    gate = SurprisalGate(
        llm=UnavailableLLMProvider(),
        embedder=SimpleEmbeddingProvider(),
        store=InMemoryStore(),
        config=SurprisalConfig(mode="novelty"),
        judge=UnavailableJudge(),
    )
    score, status, _, mode = asyncio.run(gate._compute_surprisal(_interaction("x")))
    assert (score, status, mode) == (None, "provider_unavailable", "novelty_judgment")


def test_prediction_mode_is_untouched_by_the_judge() -> None:
    gate = SurprisalGate(
        llm=UnavailableLLMProvider(),
        embedder=SimpleEmbeddingProvider(),
        store=InMemoryStore(),
        judge=MockJudge({"novel": 0.99}),
    )
    score, status, _, mode = asyncio.run(gate._compute_surprisal(_interaction("x")))
    assert status == "provider_unavailable" and mode == "on_demand" and score is None


@pytest.mark.asyncio
async def test_the_service_gate_rejects_on_low_novelty_and_waits_on_outage(tmp_path: Path) -> None:
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="local", enabled=True)
    context = registry.resolve(tmp_path, user_profile_id="local", session_id="s", host="codex")
    store = SQLiteStore(str(tmp_path / "memory.sqlite"))
    judge = MockJudge({"novel": 0.1})
    engine = MnemosEngine(
        llm=UnavailableLLMProvider(), embedder=SimpleEmbeddingProvider(), store=store, judge=judge
    )
    engine.config.surprisal.mode = "novelty"
    engine.config.sleep.extraction_mode = "judged"
    memory = LocalMemoryService(engine, registry)
    try:
        memory.capture(
            CaptureEvent(
                event_id="e1",
                context=context,
                role="user",
                occurred_at=datetime.now(timezone.utc),
                content="ok thanks, sounds good to me",
                source_kind="user_prompt",
            )
        )
        job = store.claim_jobs(owner="w")[0]
        await memory.process_job(job)
        done = store.get_job(job.id)
        assert done is not None and done.result is not None
        assert done.result["decision"] == "surprisal_filtered"
        assert done.result["encoding_diagnostics"]["surprisal_score"] == 0.1
        assert done.result["encoding_diagnostics"]["prediction_mode"] == "novelty_judgment"
        # Only the novelty question ran; nothing was classified.
        assert [call[2] for call in judge.calls] == ["judge_novelty"]

        engine._judge = UnavailableJudge()
        engine.surprisal_gate._judge = engine._judge
        memory.capture(
            CaptureEvent(
                event_id="e2",
                context=context,
                role="user",
                occurred_at=datetime.now(timezone.utc),
                content="The marker is cedar.",
                source_kind="user_prompt",
            )
        )
        job = store.claim_jobs(owner="w")[0]
        with pytest.raises(ConnectionError):
            await memory.process_job(job)
    finally:
        store.close()
