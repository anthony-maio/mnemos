from datetime import datetime, timedelta, timezone

import pytest

from mnemos.config import MnemosConfig
from mnemos.contracts import CaptureEvent
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore


def make_service(tmp_path, service_type=LocalMemoryService, registry=None):
    if registry is None:
        registry = ScopeRegistry()
        registry.register_project(tmp_path, user_profile_id="local", enabled=True)
    context = registry.resolve(tmp_path, user_profile_id="local", session_id="s", host="codex")
    config = MnemosConfig()
    for flag in type(config.modules).model_fields:
        setattr(config.modules, flag, False)
    engine = MnemosEngine(
        config=config,
        llm=UnavailableLLMProvider(),
        embedder=SimpleEmbeddingProvider(),
        store=SQLiteStore(str(tmp_path / "memory.sqlite")),
    )
    return service_type(engine, registry), context


def capture(memory, context, content, *, event_id="source", explicit=False):
    return memory.capture(
        CaptureEvent(
            event_id=event_id,
            context=context,
            role="user",
            occurred_at=datetime.now(timezone.utc),
            content=content,
            source_kind="user_prompt",
        ),
        explicit=explicit,
    )


@pytest.mark.asyncio
async def test_encoding_policy_runs_after_capture_and_commits_its_decision(tmp_path):
    class RejectAutomatic(LocalMemoryService):
        async def _automatic_encoding_gate(self, interaction):
            return "length_noise_filtered", {"policy": "reject-fixture", "kept": False}

    memory, context = make_service(tmp_path, RejectAutomatic)
    result = capture(memory, context, "This captured source must survive its encoding rejection.")
    try:
        assert memory.engine.store.get_episodic_event(result["event_id"]) is not None
        assert await memory.maintenance.run_once(force=True) == 1
        assert memory.engine.store.get_all() == []
        job = memory.engine.store.get_job(result["job_id"])
        assert job.status == "completed"
        assert job.result["decision"] == "length_noise_filtered"
        assert job.result["encoding_diagnostics"] == {"policy": "reject-fixture", "kept": False}
    finally:
        memory.engine.store.close()
    reopened = SQLiteStore(str(tmp_path / "memory.sqlite"))
    try:
        assert reopened.get_episodic_event(result["event_id"]).status == "consolidated"
        assert reopened.get_job(result["job_id"]).result["decision"] == "length_noise_filtered"
        assert reopened.get_all() == []
    finally:
        reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("size,kept", [(49, False), (50, True), (51, True)])
async def test_applied_length_policy_preserves_capture_and_explicit_writes(tmp_path, size, kept):
    from benchmarks.local_memory.encoding_comparator import LengthNoiseService

    memory, context = make_service(tmp_path, LengthNoiseService)
    try:
        result = capture(memory, context, "  " + "a" * size + "  ")
        explicit = capture(memory, context, "Use Rust", event_id="explicit", explicit=True)
        assert await memory.maintenance.run_once(force=True) == 2
        job = memory.engine.store.get_job(result["job_id"])
        assert job.result["encoding_diagnostics"]["kept"] is kept
        assert job.result["encoding_diagnostics"]["content_chars"] == size
        assert job.result["decision"] == ("facts" if kept else "length_noise_filtered")
        assert memory.engine.store.get_episodic_event(result["event_id"]) is not None
        assert memory.engine.store.get_job(explicit["job_id"]).result["decision"] == "facts"
        assert len(memory.engine.store.get_all()) == 1 + int(kept)
        fact_id = memory.engine.store.get_job(explicit["job_id"]).result["fact_ids"][0]
        correction = await memory.dispatch(
            "correct",
            {
                "event_id": "correct",
                "context": context.model_dump(),
                "fact_id": fact_id,
                "expected_version": 1,
                "content": "Use Go",
            },
        )
        assert await memory.maintenance.run_once(force=True) == 1
        assert memory.engine.store.get(fact_id).content == "Use Go"
        assert memory.engine.store.get_job(correction["job_id"]).result["decision"] == "facts"
    finally:
        memory.engine.store.close()


@pytest.mark.parametrize("filtered_target", [False, True])
def test_comparator_reports_actual_filter_and_retains_source_counts(tmp_path, filtered_target):
    from benchmarks.local_memory.contracts import SourceEvent, SessionScenario, FactRecord
    from benchmarks.local_memory.encoding_comparator import (
        EncodingComparatorAdapter,
        LengthNoiseService,
        LENGTH_PROFILE,
    )

    memory, context = make_service(tmp_path, LengthNoiseService)
    registry = memory.registry
    memory.engine.store.close()

    def factory(profile, run_key=None):
        return make_service(tmp_path, LengthNoiseService, registry=registry)[0]

    stamp = datetime.now(timezone.utc).isoformat()
    events = (
        SourceEvent(
            "explicit",
            stamp,
            "user",
            "Use Rust",
            "user_prompt",
            "automatic" if filtered_target else "explicit",
            "capture" if filtered_target else "remember",
            ("f1",),
        ),
        SourceEvent("noise", stamp, "user", "ok", "user_prompt", "automatic", "noise", ()),
    )
    if filtered_target:
        events += (
            SourceEvent(
                "correction",
                (datetime.fromisoformat(stamp) + timedelta(seconds=1)).isoformat(),
                "user",
                "Actually use Go",
                "user_prompt",
                "explicit",
                "correct",
                (),
                target_source_event_id="explicit",
                target_fact_id="f1",
                replacement_content="Use Go",
            ),
        )
    session = SessionScenario(
        "comparator-dev",
        "development",
        "codex",
        context.project_id,
        (
            FactRecord(
                "f1", "Use Rust", True, "project", "synthetic_seed", "supported", ("explicit",)
            ),
        ),
        (),
        (),
        "synthetic_seed",
        "not_run",
        events,
        ("noise",),
    )
    observed = EncodingComparatorAdapter(factory, user_profile_id="local").evaluate_session(
        session, LENGTH_PROFILE, 1
    )
    assert observed.execution_status == ("partial" if filtered_target else "complete")
    assert observed.length_filter_kept == ((False, False) if filtered_target else (False,))
    assert observed.surprisal_filter_kept == ((None, None) if filtered_target else (None,))
    assert observed.critical_retention == (not filtered_target,)
    assert observed.raw_output["length_filter_policy"]["applied"] == "durable_extraction_job"
    assert observed.raw_output["encoding_policy_decisions"][0]["kept"] is False
    assert observed.raw_output["final_row_counts"] == {
        "facts": 0 if filtered_target else 1,
        "episodes": 2,
        "jobs": 2,
    }
    if filtered_target:
        assert any(
            row["error"] == "missing_accepted_target" for row in observed.raw_output["errors"]
        ), observed.raw_output["errors"]
    else:
        assert observed.raw_output["errors"] == []
    assert observed.verified_model_provenance is False


def test_comparator_factory_wires_the_policy_only_for_length_arm(tmp_path):
    from benchmarks.local_memory.encoding_comparator import (
        COMPARATOR_PROFILES,
        LengthNoiseService,
        make_comparator_service_factory,
    )

    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="local", enabled=True)
    factory = make_comparator_service_factory(tmp_path / "factory.sqlite", registry)
    for profile in COMPARATOR_PROFILES:
        memory = factory(profile, profile.name)
        try:
            assert isinstance(memory, LengthNoiseService) == (profile.name == "length-noise")
            assert memory.engine.config.modules.surprisal == (profile.name == "full")
            assert memory.engine.config.modules.sleep is True
        finally:
            memory.engine.store.close()


@pytest.mark.asyncio
async def test_length_policy_reads_persisted_source_and_survives_pending_restart(tmp_path):
    from benchmarks.local_memory.encoding_comparator import LengthNoiseService

    memory, context = make_service(tmp_path, LengthNoiseService)
    result = capture(memory, context, "Okay", event_id="pending")
    registry = memory.registry
    memory.engine.store.close()
    memory, context = make_service(tmp_path, LengthNoiseService, registry=registry)
    try:
        assert await memory.maintenance.run_once(force=True) == 1
        job = memory.engine.store.get_job(result["job_id"])
        assert job.result["encoding_diagnostics"]["reason"] == "exact_boilerplate"
        assert job.result["decision"] == "length_noise_filtered"
        assert memory.engine.store.get_all() == []
        assert await memory.maintenance.run_once(force=True) == 0
    finally:
        memory.engine.store.close()


def test_length_policy_rejects_accidentally_enabled_surprisal(tmp_path):
    from benchmarks.local_memory.encoding_comparator import LengthNoiseService

    memory, context = make_service(tmp_path)
    try:
        memory.engine.config.modules.surprisal = True
        with pytest.raises(ValueError, match="surprisal"):
            LengthNoiseService(memory.engine, memory.registry)
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_pending_job_is_not_reported_as_an_applied_decision(tmp_path):
    from types import SimpleNamespace
    from benchmarks.local_memory.contracts import SourceEvent
    from benchmarks.local_memory.encoding_comparator import (
        EncodingComparatorAdapter,
        LengthNoiseService,
    )

    memory, context = make_service(tmp_path, LengthNoiseService)
    result = capture(memory, context, "Okay", event_id="source-length-noise-r1")
    event = SourceEvent(
        "source",
        datetime.now(timezone.utc).isoformat(),
        "user",
        "Okay",
        "user_prompt",
        "automatic",
        "capture",
        (),
    )
    session = SimpleNamespace(events=(event,))
    operations = [{"operation": "capture", "event_id": "source", "job_id": result["job_id"]}]
    try:
        values, details = EncodingComparatorAdapter._surprisal_filter_decisions(
            memory, session, operations
        )
        assert values == [None]
        assert details[0]["job_status"] == "pending"
        assert details[0]["kept"] is None
        assert await memory.maintenance.run_once(force=True) == 1
        values, details = EncodingComparatorAdapter._surprisal_filter_decisions(
            memory, session, operations
        )
        assert values == [None]  # Length decisions are not surprisal decisions.
        assert details[0]["kept"] is False
        assert details[0]["durable_event_id"] == result["event_id"]
    finally:
        memory.engine.store.close()
