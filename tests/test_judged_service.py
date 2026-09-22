"""extraction_mode=judged reaches the real service transaction through a judge."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from mnemos.contracts import CaptureEvent
from mnemos.engine import MnemosEngine
from mnemos.judgment import ChoiceAnswer, MockJudge, Noul, NoulAnswer, Score, ScoreAnswer
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeContext, ScopeRegistry
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore


def _judge(kinds: dict[str, str], *, relations: dict[str, str] | None = None) -> MockJudge:
    relations = relations or {}

    def script(key, question, state):
        if isinstance(question, Score):
            index = {"valence": 3, "arousal": 4, "complexity": 1}[key]
            return ScoreAnswer(
                value=float(index),
                confidence=0.9,
                probabilities={str(i): (1.0 if i == index else 0.0) for i in range(5)},
            )
        if isinstance(question, Noul):
            return NoulAnswer(0.05)
        if key.startswith("kind:"):
            choice = kinds[key]
        elif key.startswith("relation:"):
            choice = relations.get(key, "unrelated")
        else:
            choice = question.keys[0]
        return ChoiceAnswer(
            choice=choice,
            probabilities={k: (0.9 if k == choice else 0.01) for k in question.keys},
            confidence=0.9,
        )

    return MockJudge(script)


def service(
    tmp_path: Path, judge: MockJudge
) -> tuple[LocalMemoryService, ScopeContext, SQLiteStore]:
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="local", enabled=True)
    context = registry.resolve(tmp_path, user_profile_id="local", session_id="s", host="codex")
    store = SQLiteStore(str(tmp_path / "memory.sqlite"))
    engine = MnemosEngine(
        llm=UnavailableLLMProvider(),
        embedder=SimpleEmbeddingProvider(),
        store=store,
        judge=judge,
    )
    engine.config.sleep.extraction_mode = "judged"
    engine.config.modules.surprisal = False
    return LocalMemoryService(engine, registry), context, store


async def _capture_and_process(memory, context, store, content: str) -> dict:
    captured = memory.capture(
        CaptureEvent(
            event_id="event-" + str(abs(hash(content)))[:8],
            context=context,
            role="user",
            occurred_at=datetime.now(timezone.utc),
            content=content,
            source_kind="user_prompt",
        )
    )
    job = store.claim_jobs(owner="worker")[0]
    await memory.process_job(job)
    done = store.get_job(job.id)
    assert done is not None and done.status == "completed", done
    return {"captured": captured, "result": done.result}


@pytest.mark.asyncio
async def test_a_judged_capture_becomes_a_fact_with_judged_affect(tmp_path: Path) -> None:
    judge = _judge({"kind:r0": "continuing_requirement", "kind:r1": "temporary_instruction"})
    memory, context, store = service(tmp_path, judge)
    try:
        outcome = await _capture_and_process(
            memory, context, store, "The review window is Tuesday. For this reply, be brief."
        )
        result = outcome["result"]
        assert result["decision"] == "facts"
        assert result["encoding_diagnostics"]["extraction_mode"] == "judged"
        assert result["encoding_diagnostics"]["judgment_calls"] == 2
        assert result["encoding_diagnostics"]["judgment_input_tokens"] > 0
        (fact,) = store.get_all()
        assert fact.content.strip() == "The review window is Tuesday."
        assert fact.cognitive_state is not None
        assert fact.cognitive_state.valence == pytest.approx(0.5)
        assert fact.cognitive_state.arousal == 1.0
        assert (
            store.list_fact_evidence(fact.id)[0].source_event_id == outcome["captured"]["event_id"]
        )
        # The generative provider was never needed.
        assert all(call[2] in {"segment_source", "classify_source"} for call in judge.calls)
        assert memory.engine.affective_router.get_current_state(fact.metadata).arousal == 1.0
    finally:
        store.close()


@pytest.mark.asyncio
async def test_a_reworded_repeat_supports_instead_of_duplicating(tmp_path: Path) -> None:
    judge = _judge(
        {"kind:r0": "asserted_fact"},
        relations={"relation:c0:t0": "same_information"},
    )
    memory, context, store = service(tmp_path, judge)
    try:
        await _capture_and_process(memory, context, store, "The marker is cedar.")
        second = await _capture_and_process(memory, context, store, "Cedar is our marker.")
        facts = store.get_all()
        assert [f.content.strip() for f in facts] == ["The marker is cedar."]
        assert facts[0].version >= 1
        assert second["result"]["decision"] == "facts"
        purposes = [call[2] for call in judge.calls]
        assert "reconcile_candidates" in purposes
    finally:
        store.close()


@pytest.mark.asyncio
async def test_probe_cognition_runs_through_the_judge_in_judged_mode(tmp_path: Path) -> None:
    class Probe:
        """A judge that is not a MockJudge, as a configured backend would be."""

        provider = "probe"
        model = "probe"

        def __init__(self) -> None:
            self.inner = _judge({"kind:r0": "asserted_fact"})

        async def ask(self, state, questions, *, purpose, model=None):
            return await self.inner.ask(state, questions, purpose=purpose, model=model)

    memory, context, store = service(tmp_path, Probe())  # type: ignore[arg-type]
    try:
        probe = await memory.dispatch("probe_cognition", {"context": context.model_dump()})
        assert probe["status"] == "observed", probe
        assert probe["provider"] == "Probe"
        assert probe["extraction_mode"] == "judged"

        memory.engine._judge = MockJudge()
        refused = await memory.dispatch("probe_cognition", {"context": context.model_dump()})
        assert refused["status"] == "failed" and "cognition provider" in refused["reason"]
    finally:
        store.close()
