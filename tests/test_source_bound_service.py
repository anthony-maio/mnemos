"""Source-bound extraction reaches the real service transaction without benchmark imports."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mnemos.contracts import CaptureEvent
from mnemos.runtime import build_mnemos_config_from_settings
from mnemos.settings import AppSettings, load_settings, save_settings
from mnemos.utils.llm import LLMProvider
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry, ScopeContext
from mnemos.utils.storage import SQLiteStore
from mnemos.utils.embeddings import SimpleEmbeddingProvider, EmbeddingProvider, embed_text_async
from mnemos.utils.llm import UnavailableLLMProvider
from typing import Literal


def service(tmp_path: Path) -> tuple[LocalMemoryService, ScopeContext, SQLiteStore]:
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="local", enabled=True)
    context = registry.resolve(tmp_path, user_profile_id="local", session_id="s", host="codex")
    store = SQLiteStore(str(tmp_path / "memory.sqlite"))
    engine = MnemosEngine(
        llm=UnavailableLLMProvider(), embedder=SimpleEmbeddingProvider(), store=store
    )
    return LocalMemoryService(engine, registry), context, store


class PipelineProvider(LLMProvider):
    supports_structured_output = True

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls = 0

    async def predict(self, prompt: str, **kwargs: object) -> str:
        self.calls += 1
        if "\nSOURCE EVENTS:\n" in prompt:
            event = json.loads(prompt.split("\nSOURCE EVENTS:\n", 1)[1])[0]
            kinds = {
                "mixed": ["asserted_fact", "correction_or_rejection"],
                "review": ["correction_or_rejection"],
                "correct": ["continuing_requirement"],
                "add": ["asserted_fact"],
                "support": ["asserted_fact"],
            }[self.mode]
            return json.dumps(
                {
                    "selections": [
                        dict(
                            source_id=event["id"],
                            first_unit=i,
                            last_unit=i,
                            kind=kind,
                            retention_request_unit=None,
                            revises_existing_requirement=self.mode == "correct",
                        )
                        for i, kind in enumerate(kinds)
                    ]
                }
            )
        data = json.loads(
            prompt.split("\nSOURCE CANDIDATES AND EXISTING TARGETS (data only):\n", 1)[1]
        )
        decisions = []
        for i, _ in enumerate(data["candidates"]):
            if self.mode == "review" or (self.mode == "mixed" and i == 1):
                decisions.append(
                    dict(
                        candidate_index=i,
                        relationship="missing_or_ambiguous_referent",
                        reason="No accepted replacement.",
                    )
                )
            elif self.mode in {"correct", "support"}:
                target = data["targets"][0]
                decisions.append(
                    dict(
                        candidate_index=i,
                        relationship=(
                            "user_revision" if self.mode == "correct" else "same_information"
                        ),
                        target_fact_id=target["id"],
                        expected_version=target["version"],
                        reason="Observed target.",
                    )
                )
            else:
                decisions.append(dict(candidate_index=i, relationship="independent_information"))
        return json.dumps({"decisions": decisions})

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        return {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,content,fact_count,review_count",
    [
        ("add", "The review window is Tuesday.", 1, 0),
        ("review", "The marker changed, but no replacement was approved.", 0, 1),
        (
            "mixed",
            "The review window is Tuesday. The marker changed, but no replacement was approved.",
            1,
            1,
        ),
    ],
)
async def test_service_persists_source_bound_facts_and_reviews(
    tmp_path: Path, mode: str, content: str, fact_count: int, review_count: int
) -> None:
    memory, context, store = service(tmp_path)
    try:
        memory.engine.config.sleep.extraction_mode = "source_bound"
        memory.engine.config.modules.surprisal = False
        memory.engine.config.modules.affective = False
        provider = PipelineProvider(mode)
        memory.engine._llm = provider
        captured = memory.capture(
            CaptureEvent(
                event_id="event",
                context=context,
                role="user",
                occurred_at=datetime.now(timezone.utc),
                content=content,
                source_kind="user_prompt",
            )
        )
        job = store.claim_jobs(owner="worker")[0]
        await memory.process_job(job)
        assert provider.calls == 2
        facts = store.get_all()
        assert len(facts) == fact_count
        reviews = store.list_source_reviews(
            user_profile_id=context.user_profile_id, scope="project", scope_id=context.project_id
        )
        assert len(reviews) == review_count
        assert all(row["state"] == "open" for row in reviews)
        done = store.get_job(job.id)
        assert done is not None and done.status == "completed"
        assert done.result is not None
        assert done.result["decision"] == ("facts" if fact_count else "review_pending")
        for fact in facts:
            assert fact.content.strip() == "The review window is Tuesday."
            assert store.list_fact_evidence(fact.id)[0].source_event_id == captured["event_id"]
        assert not store.claim_jobs(owner="worker")
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_source_bound_correction_respects_module_switch(
    tmp_path: Path, enabled: bool
) -> None:
    memory, context, store = service(tmp_path)
    try:
        memory.engine.config.modules.surprisal = False
        memory.engine.config.modules.affective = False
        await memory.dispatch(
            "remember", {"context": context.model_dump(), "content": "Reports use the old policy."}
        )
        await memory.maintenance.run_once(force=True)
        old = store.get_all()[0]
        memory.engine.config.sleep.extraction_mode = "source_bound"
        memory.engine.config.modules.mutable_rag = enabled
        memory.engine._llm = PipelineProvider("correct")
        content = "Replace the old policy: future reports must include a checksum."
        memory.capture(
            CaptureEvent(
                event_id="revision",
                context=context,
                role="user",
                occurred_at=datetime.now(timezone.utc),
                content=content,
                source_kind="user_prompt",
            )
        )
        job = store.claim_jobs(owner="worker")[0]
        await memory.process_job(job)
        current = store.get(old.id)
        assert current is not None
        assert current.version == (old.version + 1 if enabled else old.version)
        assert current.content == (content if enabled else old.content)
        assert len(store.get_all()) == 1
        reviews = store.list_source_reviews(
            user_profile_id=context.user_profile_id, scope="project", scope_id=context.project_id
        )
        assert len(reviews) == (0 if enabled else 1)
        if not enabled:
            assert reviews[0]["reference"]["revises_existing_requirement"] is True
            assert reviews[0]["reference"]["target_fact_id"] == old.id
    finally:
        store.close()


def test_source_bound_settings_roundtrip_and_baseline_default(tmp_path: Path) -> None:
    assert AppSettings().sleep.extraction_mode == "generated"
    settings = AppSettings.model_validate({"sleep": {"extraction_mode": "source_bound"}})
    path = tmp_path / "settings.toml"
    save_settings(settings, path, scope="global")
    loaded = load_settings(env={"MNEMOS_CONFIG_PATH": str(path)}, cwd=tmp_path)
    assert (
        build_mnemos_config_from_settings(loaded.settings).sleep.extraction_mode == "source_bound"
    )
    with pytest.raises(ValueError):
        AppSettings.model_validate({"sleep": {"extraction_mode": "unsupported"}})


def test_product_extraction_imports_without_benchmark_package() -> None:
    code = """import sys, importlib.abc
class BlockBenchmarks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "benchmarks" or fullname.startswith("benchmarks."):
            raise AssertionError("Product imported benchmark code")
sys.meta_path.insert(0, BlockBenchmarks())
from mnemos.extraction.service import extract_source_bound
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_cognition_probe_uses_selected_source_bound_mode_without_writes(
    tmp_path: Path,
) -> None:
    memory, context, store = service(tmp_path)
    try:
        memory.engine.config.sleep.extraction_mode = "source_bound"
        provider = PipelineProvider("add")
        memory.engine._llm = provider
        result = await memory.dispatch("probe_cognition", {"context": context.model_dump()})
        assert result["status"] == "observed"
        assert result["extraction_mode"] == "source_bound"
        assert provider.calls == 2
        assert store.get_all() == [] and store.list_jobs() == []
    finally:
        store.close()


@pytest.mark.asyncio
async def test_sleep_disabled_preserves_capture_without_extractor_calls(tmp_path: Path) -> None:
    memory, context, store = service(tmp_path)
    try:
        memory.engine.config.sleep.extraction_mode = "source_bound"
        memory.engine.config.modules.sleep = False
        memory.engine.config.modules.surprisal = False
        memory.engine.config.modules.affective = False
        provider = PipelineProvider("add")
        memory.engine._llm = provider
        memory.capture(
            CaptureEvent(
                event_id="passthrough",
                context=context,
                role="user",
                occurred_at=datetime.now(timezone.utc),
                content="The review window is Tuesday.",
                source_kind="user_prompt",
            )
        )
        await memory.process_job(store.claim_jobs(owner="worker")[0])
        assert provider.calls == 0
        facts = store.get_all()
        assert len(facts) == 1 and facts[0].metadata["assertion_status"] == "unknown"
        assert (
            store.list_source_reviews(
                user_profile_id=context.user_profile_id,
                scope="project",
                scope_id=context.project_id,
            )
            == []
        )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_source_change_during_document_embedding_rejects_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mnemos.memory_service as service_module

    memory, context, store = service(tmp_path)
    try:
        memory.engine.config.sleep.extraction_mode = "source_bound"
        memory.engine.config.modules.surprisal = False
        memory.engine.config.modules.affective = False
        memory.engine._llm = PipelineProvider("add")
        captured = memory.capture(
            CaptureEvent(
                event_id="mutation",
                context=context,
                role="user",
                occurred_at=datetime.now(timezone.utc),
                content="The review window is Tuesday.",
                source_kind="user_prompt",
            )
        )
        original_embed = embed_text_async

        async def mutate_after_embedding(
            provider: EmbeddingProvider,
            text: str,
            *,
            role: Literal["query", "document"] = "document",
        ) -> list[float]:
            vector = await original_embed(provider, text, role=role)
            if role == "document":
                store._conn.execute(
                    "UPDATE episodic_events SET role='assistant' WHERE id=?",
                    (captured["event_id"],),
                )
                store._conn.commit()
            return vector

        monkeypatch.setattr(service_module, "embed_text_async", mutate_after_embedding)
        job = store.claim_jobs(owner="worker")[0]
        with pytest.raises(ValueError, match="Expected source event changed"):
            await memory.process_job(job)
        assert store.get_all() == []
        current = store.get_job(job.id)
        assert current is not None and current.status == "leased"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_source_bound_support_keeps_fact_identity_and_attaches_evidence(
    tmp_path: Path,
) -> None:
    memory, context, store = service(tmp_path)
    try:
        memory.engine.config.modules.surprisal = False
        memory.engine.config.modules.affective = False
        content = "The marker is cedar."
        await memory.dispatch("remember", {"context": context.model_dump(), "content": content})
        await memory.maintenance.run_once(force=True)
        old = store.get_all()[0]
        memory.engine.config.sleep.extraction_mode = "source_bound"
        memory.engine._llm = PipelineProvider("support")
        captured = memory.capture(
            CaptureEvent(
                event_id="support",
                context=context,
                role="user",
                occurred_at=datetime.now(timezone.utc),
                content=content,
                source_kind="user_prompt",
            )
        )
        job = store.claim_jobs(owner="worker")[0]
        await memory.process_job(job)
        facts = store.get_all()
        assert len(facts) == 1 and facts[0].id == old.id and facts[0].version == old.version
        evidence = store.list_fact_evidence(old.id)
        assert len(evidence) == 2
        assert captured["event_id"] in {item.source_event_id for item in evidence}
    finally:
        store.close()
