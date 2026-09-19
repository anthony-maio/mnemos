from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.local_memory.contracts import FactRecord, SessionScenario, SourceEvent
from benchmarks.local_memory.harness import DEFAULT_PROFILES, ExperimentProfile
from benchmarks.local_memory.ollama_adapter import OllamaEvaluatorAdapter
from mnemos.config import MnemosConfig, ModuleConfig, SleepConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import LLMProvider
from mnemos.utils.storage import SQLiteStore


class _DeterministicExtractionProvider(LLMProvider):
    supports_structured_output = True

    def __init__(self, *, decision: str, content: str) -> None:
        self.decision = decision
        self.content = content
        self.prompts: list[str] = []

    async def predict(self, prompt: str, **_: Any) -> str:
        self.prompts.append(prompt)
        if self.decision == "no_durable_facts":
            return json.dumps(
                {
                    "decision": "no_durable_facts",
                    "facts": [],
                    "reason": "The deterministic development response supplies no fact.",
                }
            )
        source_rows = json.loads(
            prompt.split("SOURCE EVENTS:\n", 1)[1].split("\nEXISTING FACTS", 1)[0]
        )
        return json.dumps(
            {
                "decision": "facts",
                "facts": [
                    {
                        "content": self.content,
                        "evidence_ids": [source_rows[0]["id"]],
                        "operation": "add",
                    }
                ],
            }
        )

    async def classify(self, _prompt: str, _labels: list[str]) -> dict[str, float]:
        return {}


def _scenario(project_id: str, *, content: str, replacement: str) -> SessionScenario:
    fact_id = "gold-target-fact"
    source_id = "direct-assertion"
    return SessionScenario(
        session_id="lifecycle-extraction-contract",
        split="development",
        host="codex",
        project_id=project_id,
        facts=(
            FactRecord(
                fact_id=fact_id,
                content=content,
                critical=False,
                scope="project",
                source_label="synthetic_development_fixture",
                gold_status="superseded",
                evidence_event_ids=(source_id,),
            ),
        ),
        queries=(),
        state_pairs=(),
        source_label="synthetic_development_fixture",
        human_acceptance="not_run",
        events=(
            SourceEvent(
                event_id=source_id,
                occurred_at="2026-01-01T00:00:00Z",
                role="user",
                content=content,
                source_kind="user_prompt",
                write_mode="automatic",
                lifecycle_operation="capture",
                evidence_fact_ids=(fact_id,),
            ),
            SourceEvent(
                event_id="direct-correction",
                occurred_at="2026-01-01T00:01:00Z",
                role="user",
                content="The user directly corrects the current project assertion.",
                source_kind="explicit_correction",
                write_mode="explicit",
                lifecycle_operation="correct",
                evidence_fact_ids=(),
                target_source_event_id=source_id,
                target_fact_id=fact_id,
                replacement_content=replacement,
            ),
        ),
        scenario_types=("correction",),
    )


def _service_factory(
    db_path: Path,
    project_root: Path,
    provider: _DeterministicExtractionProvider,
) -> tuple[Any, str]:
    registry = ScopeRegistry(profiles=["alice"])
    project = registry.register_project(project_root, user_profile_id="alice", enabled=True)

    def factory(_profile: ExperimentProfile, _run_key: str) -> LocalMemoryService:
        config = MnemosConfig(
            modules=ModuleConfig(
                surprisal=False,
                mutable_rag=False,
                affective=False,
                sleep=True,
                spreading=False,
            ),
            sleep=SleepConfig(
                consolidation_interval_seconds=0,
                min_episodes_before_consolidation=1,
            ),
        )
        store = SQLiteStore(str(db_path))
        engine = MnemosEngine(
            config=config,
            llm=provider,
            embedder=SimpleEmbeddingProvider(),
            store=store,
        )
        return LocalMemoryService(engine, registry)

    return factory, project.id


def _evaluate(
    tmp_path: Path, *, decision: str, content: str, replacement: str
) -> tuple[Any, _DeterministicExtractionProvider, Path]:
    provider = _DeterministicExtractionProvider(decision=decision, content=content)
    db_path = tmp_path / "memory.sqlite"
    factory, project_id = _service_factory(db_path, tmp_path, provider)
    session = _scenario(project_id, content=content, replacement=replacement)
    adapter = OllamaEvaluatorAdapter(factory, user_profile_id="alice")
    observation = adapter.evaluate_session(session, DEFAULT_PROFILES[0], 1)
    return observation, provider, db_path


def test_no_durable_facts_preserves_raw_job_and_partial_correction_gate(
    tmp_path: Path,
) -> None:
    content = "The earlier project note discusses Redis but supplies no current assertion."
    observation, provider, db_path = _evaluate(
        tmp_path,
        decision="no_durable_facts",
        content=content,
        replacement="The current project assertion after a valid correction.",
    )

    assert provider.prompts, "the extraction path must call the injected provider"
    assert observation.execution_status == "partial"
    assert observation.raw_output["lifecycle_summary"]["correct"] == {
        "attempted": 1,
        "committed": 0,
        "rejected": 0,
        "mapping_invalid": 0,
        "missing_target": 1,
    }
    assert observation.raw_output["extraction_judgements"] == []
    assert any(
        item.get("error") == "missing_accepted_target" for item in observation.raw_output["errors"]
    )

    reopened = SQLiteStore(str(db_path))
    try:
        jobs = reopened.list_jobs()
        assert len(jobs) == 1
        assert jobs[0].status == "completed"
        assert jobs[0].result is not None
        assert jobs[0].result["decision"] == "no_durable_facts"
        assert jobs[0].result["fact_ids"] == []
        assert jobs[0].result.get("fact_eligible", False) is False
        assert reopened.get_all() == []
        event = reopened.get_episodic_event(jobs[0].payload["event_id"])
        assert event is not None
        assert event.status == "consolidated"
        assert event.content == content
    finally:
        reopened.close()


def test_direct_assertion_creates_evidence_target_and_correction_succeeds(
    tmp_path: Path,
) -> None:
    content = "The project cache uses Redis for session tokens."
    replacement = "The project cache uses PostgreSQL for session tokens."
    observation, provider, db_path = _evaluate(
        tmp_path,
        decision="facts",
        content=content,
        replacement=replacement,
    )

    assert len(provider.prompts) == 1
    assert observation.execution_status == "complete"
    assert observation.raw_output["lifecycle_summary"]["correct"] == {
        "attempted": 1,
        "committed": 1,
        "rejected": 0,
        "mapping_invalid": 0,
        "missing_target": 0,
    }

    reopened = SQLiteStore(str(db_path))
    try:
        facts = reopened.get_all()
        assert len(facts) == 1
        fact = facts[0]
        assert fact.content == replacement
        assert fact.version == 2
        evidence = reopened.list_fact_evidence(fact.id)
        assert evidence
        captured_sources = []
        for item in evidence:
            if item.source_event_id is None:
                continue
            source = reopened.get_episodic_event(item.source_event_id)
            if source is not None and source.metadata.get("source_event_id", "").startswith(
                "direct-assertion"
            ):
                captured_sources.append(source)
        assert captured_sources
        assert captured_sources[0].content == content
        jobs = reopened.list_jobs()
        assert len(jobs) == 2
        assert all(job.status == "completed" for job in jobs)
    finally:
        reopened.close()
