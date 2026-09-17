"""Offline contract tests for the local-memory evaluation harness."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from benchmarks.local_memory import (
    DEFAULT_PROFILES,
    ExperimentConfig,
    ExperimentHarness,
    RunObservation,
    build_completed_artifact,
    load_dataset,
    paired_bootstrap_interval,
    save_checkpoint,
    wilson_interval,
)
from benchmarks.local_memory.artifacts import ArtifactRecorder
from benchmarks.local_memory import OllamaEvaluatorAdapter, make_ollama_service_factory
from benchmarks.local_memory.ollama_adapter import SHADOW_LENGTH_MIN_CHARS
from benchmarks.local_memory.contracts import MINIMUMS
from benchmarks.local_memory.dev_smoke import _code_provenance
from benchmarks.local_memory.harness import compare_observation_metrics, main
from benchmarks.local_memory.heldout_run import _live_model_metadata, _registry_for_dataset
from benchmarks.local_memory.module_challenge import load_module_challenge_dataset
from benchmarks.local_memory.module_challenge_run import (
    CHALLENGE_PROFILES,
    _challenge_registry,
    _validate_challenge_contexts,
    _select_profiles,
    main as module_challenge_main,
)
from benchmarks.local_memory.performance import _StoreCallTrace
from benchmarks.local_memory.resumption import Checkpoint, load_checkpoint
from mnemos.contracts import ExtractionResponse
from mnemos.modules.sleep import SleepDaemon
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.llm import LLMProvider, OllamaProvider
from mnemos.utils.storage import InMemoryStore, SQLiteStore

DATASET = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "local_memory"
    / "datasets"
    / "local-memory-heldout-v2.jsonl"
)
MODULE_CHALLENGE_DATASET = DATASET.with_name("local-memory-module-challenge-v1.jsonl")


def test_heldout_dataset_meets_frozen_contract_and_is_labeled() -> None:
    dataset = load_dataset(DATASET)

    assert len(dataset.heldout_sessions) >= MINIMUMS["heldout_sessions"]
    assert (
        sum(fact.critical for fact in dataset.facts if fact.fact_id.startswith("fact-"))
        >= MINIMUMS["critical_facts"]
    )
    assert (
        sum(query.kind == "answerable" for query in dataset.queries)
        >= MINIMUMS["answerable_queries"]
    )
    assert (
        sum(query.kind == "no_answer" for query in dataset.queries) >= MINIMUMS["no_answer_queries"]
    )
    assert len(dataset.state_pairs) >= MINIMUMS["state_pairs"]
    assert all(session.source_label == "synthetic_seed" for session in dataset.sessions)
    assert all(session.human_acceptance == "not_run" for session in dataset.sessions)
    assert set(dataset.development_session_ids).isdisjoint(
        {session.session_id for session in dataset.heldout_sessions}
    )
    assert all(session.events for session in dataset.sessions)
    assert all(
        event.occurred_at <= later.occurred_at
        for session in dataset.sessions
        for event, later in zip(session.events, session.events[1:])
    )
    assert any(
        event.lifecycle_operation == "correct"
        for session in dataset.heldout_sessions
        for event in session.events
    )
    assert any(
        event.lifecycle_operation == "forget"
        for session in dataset.heldout_sessions
        for event in session.events
    )


def test_profile_matrix_has_baseline_full_and_one_module_ablations() -> None:
    names = {profile.name for profile in DEFAULT_PROFILES}
    assert names == {
        "baseline",
        "full",
        "ablate-surprisal_gating",
        "ablate-mutable_rag",
        "ablate-affective_routing",
        "ablate-sleep_consolidation",
        "ablate-spreading_activation",
    }
    assert len(DEFAULT_PROFILES[1].enabled_modules) == 5
    assert all(len(profile.enabled_modules) == 4 for profile in DEFAULT_PROFILES[2:])


def test_module_challenge_fixture_has_competing_candidates_and_synthetic_label() -> None:
    dataset = load_module_challenge_dataset(MODULE_CHALLENGE_DATASET)
    assert len(dataset.heldout_sessions) == 50
    assert all(
        len(session.state_pairs[0].candidate_fact_ids) >= 25 for session in dataset.heldout_sessions
    )
    assert all(
        set(session.state_pairs[0].graph_target_fact_ids)
        <= set(session.state_pairs[0].candidate_fact_ids)
        for session in dataset.heldout_sessions
    )
    assert all(
        not set(session.state_pairs[0].graph_target_fact_ids)
        & set(session.state_pairs[0].direct_vector_seed_fact_ids)
        for session in dataset.heldout_sessions
    )
    assert dataset.manifest["advertised_as_real"] is False
    assert dataset.manifest["human_acceptance"] == "not_run"


def test_module_challenge_launcher_selects_only_supported_profiles() -> None:
    assert {profile.name for profile in _select_profiles(None)} == set(CHALLENGE_PROFILES)
    with pytest.raises(ValueError, match="unknown challenge profiles"):
        _select_profiles({"baseline", "unknown"})


def test_module_challenge_targets_validate_before_execution(tmp_path: Path) -> None:
    dataset = load_module_challenge_dataset(MODULE_CHALLENGE_DATASET)
    registry = _challenge_registry(dataset, tmp_path)
    _validate_challenge_contexts(dataset, registry)


def test_module_challenge_launcher_help_is_available() -> None:
    with pytest.raises(SystemExit) as result:
        module_challenge_main(["--help"])
    assert result.value.code == 0


def test_code_provenance_excludes_repo_children_not_venv_ancestors(tmp_path: Path) -> None:
    repo = tmp_path / ".venv" / "eval-frozen" / "repo"
    repo.mkdir(parents=True)
    (repo / "main.py").write_text("print('fixture')\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    ignored = repo / ".venv" / "Scripts"
    ignored.mkdir(parents=True)
    (ignored / "python.py").write_text("ignored = True\n", encoding="utf-8")

    digest, files = _code_provenance(repo)

    assert len(digest) == 64
    assert files == ["main.py", "pyproject.toml"]


def test_code_provenance_rejects_empty_runnable_surface(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No runnable Python/config files"):
        _code_provenance(tmp_path)


def test_uncertainty_helpers_are_reproducible_and_include_counts() -> None:
    interval = wilson_interval(95, 100)
    assert interval.sample_count == 100
    assert 0.85 < interval.lower < interval.estimate < interval.upper < 1.0
    first = paired_bootstrap_interval([0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], repetitions=500)
    second = paired_bootstrap_interval([0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], repetitions=500)
    assert first.as_dict() == second.as_dict()
    assert first.estimate == pytest.approx(0.5)
    assert first.lower >= 0.0


class _Adapter:
    def evaluate_session(self, session, profile, repetition):
        critical = tuple(True for fact in session.facts if fact.critical)
        supported = tuple(True for fact in session.facts if fact.gold_status == "supported")
        answerable = tuple(True for query in session.queries if query.kind == "answerable")
        no_answer = tuple(True for query in session.queries if query.kind == "no_answer")
        state = tuple(True for _ in session.state_pairs)
        return RunObservation(
            session_id=session.session_id,
            repetition=repetition,
            critical_retention=critical,
            critical_retrieval=critical,
            supported_precision=supported,
            answerable_hits=answerable,
            no_answer_correct=no_answer,
            state_pair_semantic_hits=state,
            state_pair_state_hits=state,
            latency_ms=4.0,
            bytes_before=100,
            bytes_after=125,
            provider="test-adapter",
            model_revision="fixture-model-rev",
            embedding_revision="fixture-embedding-rev",
            execution_status="complete",
            adapter_kind="test",
            raw_output={"profile": profile.name},
        )


def test_harness_runs_adapter_and_reports_measured_counts(tmp_path: Path) -> None:
    dataset = load_dataset(DATASET)
    config = ExperimentConfig(profile=DEFAULT_PROFILES[0], provider="test-adapter")
    checkpoint = tmp_path / "run.checkpoint.json"
    artifact = ExperimentHarness(dataset, config).run(
        _Adapter(), checkpoint_path=checkpoint, run_id="fixture-run"
    )

    assert artifact["status"] == "completed"
    assert artifact["sample_counts"]["sessions"] == 200
    assert artifact["sample_counts"]["observations"] == 600
    assert artifact["measurements"]["answerable_recall_at_k"]["sample_count"] == 300
    assert artifact["measurements"]["supported_precision"]["total_count"] == 360
    assert artifact["measurements"]["supported_precision"]["missing_count"] == 0
    assert artifact["measurements"]["latency_ms"]["p95"] == 4.0
    assert artifact["measurements"]["row_or_byte_growth"]["total_growth_bytes"] == 15_000
    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_payload["status"] == "complete"


def test_cli_writes_truthful_not_run_artifact_without_adapter(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    assert main(["--dataset", str(DATASET), "--output", str(output), "--provider", "ollama"]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["status"] == "not_run"
    assert artifact["gates"]["real_model"] == "NOT RUN"
    assert artifact["measurements"] == {}
    assert artifact["dataset"]["advertised_as_real"] is False


def test_artifact_recorder_keeps_raw_streams_distinct_in_shared_directory(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    full_path = tmp_path / "full.json"
    ArtifactRecorder(baseline_path).write_run(
        {"status": "complete", "profile": "baseline"},
        [{"profile": "baseline", "value": 1}],
    )
    ArtifactRecorder(full_path).write_run(
        {"status": "complete", "profile": "full"},
        [{"profile": "full", "value": 2}],
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    full = json.loads(full_path.read_text(encoding="utf-8"))
    baseline_raw = Path(baseline["raw_outputs_file"]["path"])
    full_raw = Path(full["raw_outputs_file"]["path"])
    assert baseline_raw != full_raw
    assert baseline_raw.name == "baseline.raw-outputs.jsonl"
    assert full_raw.name == "full.raw-outputs.jsonl"
    assert json.loads(baseline_raw.read_text(encoding="utf-8")) == {
        "profile": "baseline",
        "value": 1,
    }
    assert json.loads(full_raw.read_text(encoding="utf-8")) == {
        "profile": "full",
        "value": 2,
    }


def test_checkpoint_rejects_dataset_or_config_drift(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    checkpoint = Checkpoint("run", "dataset-a", "config-a", "full", 3)
    save_checkpoint(path, checkpoint)
    with pytest.raises(ValueError, match="dataset_sha256"):
        load_checkpoint(
            path,
            dataset_sha256="dataset-b",
            config_sha256="config-a",
            profile="full",
            repetitions=3,
            model_sha256="model-a",
            embedding_sha256="embedding-a",
            options_sha256="options-a",
            code_sha256="code-a",
        )


def test_paired_comparison_uses_same_observation_keys() -> None:
    base = {
        "raw_outputs": [
            {
                "session_id": "s1",
                "repetition": 1,
                "execution_status": "complete",
                "answerable_hits": [False],
            }
        ]
    }
    treatment = {
        "raw_outputs": [
            {
                "session_id": "s1",
                "repetition": 1,
                "execution_status": "complete",
                "answerable_hits": [True],
            }
        ]
    }
    result = compare_observation_metrics(base, treatment, config=ExperimentConfig())
    assert result["paired_observations"] == 1
    assert result["metrics"]["answerable_hits"]["estimate"] == 1.0


def test_adapter_mapping_rejects_string_booleans() -> None:
    class BadAdapter:
        def evaluate_session(self, session, profile, repetition):
            return {"critical_retention": ["false"]}

    with pytest.raises(TypeError, match="non-boolean"):
        ExperimentHarness(load_dataset(DATASET), ExperimentConfig()).run(BadAdapter())


def test_observation_cardinality_is_required_before_artifact_completion() -> None:
    class PartialAdapter:
        def evaluate_session(self, session, profile, repetition):
            return RunObservation(
                session_id=session.session_id,
                repetition=repetition,
                execution_status="complete",
                adapter_kind="test",
            )

    with pytest.raises(ValueError, match="critical_retention"):
        ExperimentHarness(load_dataset(DATASET), ExperimentConfig()).run(PartialAdapter())


def test_complete_row_count_with_partial_status_stays_incomplete() -> None:
    dataset = load_dataset(DATASET)
    observations = []
    for session in dataset.heldout_sessions:
        critical = tuple(False for fact in session.facts if fact.critical)
        supported = tuple(False for fact in session.facts if fact.gold_status == "supported")
        answers = tuple(False for query in session.queries if query.kind == "answerable")
        abstentions = tuple(False for query in session.queries if query.kind == "no_answer")
        state = tuple(False for _ in session.state_pairs)
        for repetition in range(1, 4):
            observations.append(
                RunObservation(
                    session_id=session.session_id,
                    repetition=repetition,
                    critical_retention=critical,
                    critical_retrieval=critical,
                    supported_precision=supported,
                    answerable_hits=answers,
                    no_answer_correct=abstentions,
                    state_pair_semantic_hits=state,
                    state_pair_state_hits=state,
                    execution_status="partial",
                    adapter_kind="test",
                )
            )
    artifact = build_completed_artifact(dataset, ExperimentConfig(), observations, "partial-run")
    assert artifact["status"] == "incomplete"
    assert artifact["gates"]["real_model"] == "NOT VERIFIED"


def test_test_adapter_cannot_close_real_model_gate() -> None:
    dataset = load_dataset(DATASET)
    config = ExperimentConfig(profile=DEFAULT_PROFILES[0], provider="test-adapter")
    artifact = ExperimentHarness(dataset, config).run(_Adapter())
    assert artifact["status"] == "completed"
    assert artifact["gates"]["real_model"] == "NOT VERIFIED"


def test_checkpoint_binds_model_options_code_and_run_identity(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    checkpoint = Checkpoint(
        "run-a",
        "dataset-a",
        "config-a",
        "full",
        3,
        model_sha256="model-a",
        embedding_sha256="embedding-a",
        options_sha256="options-a",
        code_sha256="code-a",
    )
    save_checkpoint(path, checkpoint)
    with pytest.raises(ValueError, match="run_id"):
        load_checkpoint(
            path,
            dataset_sha256="dataset-a",
            config_sha256="config-a",
            profile="full",
            repetitions=3,
            model_sha256="model-a",
            embedding_sha256="embedding-a",
            options_sha256="options-a",
            code_sha256="code-a",
            run_id="run-b",
        )
    with pytest.raises(ValueError, match="model_sha256"):
        load_checkpoint(
            path,
            dataset_sha256="dataset-a",
            config_sha256="config-a",
            profile="full",
            repetitions=3,
            model_sha256="model-b",
            embedding_sha256="embedding-a",
            options_sha256="options-a",
            code_sha256="code-a",
            run_id="run-a",
        )


def test_clustered_bootstrap_records_session_clusters() -> None:
    interval = paired_bootstrap_interval(
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        repetitions=300,
        cluster_ids=["s1", "s1", "s2", "s2"],
    )
    assert interval.cluster_count == 2
    assert interval.sample_count == 4


def test_performance_trace_records_actual_engine_store_calls_and_restores_methods() -> None:
    store = SQLiteStore(":memory:")
    store.store(
        MemoryChunk(
            id="trace-fact",
            content="Alpha trace fact",
            embedding=[1.0, 0.0],
            metadata={"scope": "project", "scope_id": "p", "user_profile_id": "u"},
        )
    )
    original_retrieve = store.retrieve

    with _StoreCallTrace(store) as trace:
        assert store.retrieve([1.0, 0.0], top_k=1)[0].id == "trace-fact"
        assert store.retrieve_lexical("alpha", top_k=1)[0].id == "trace-fact"
        assert store.mark_retrieved_many(["trace-fact"]) == 1
        assert store.get("trace-fact") is not None

    artifact = trace.artifact()
    assert [event["method"] for event in artifact["events"]] == [
        "retrieve",
        "retrieve_lexical",
        "mark_retrieved_many",
        "get",
    ]
    assert artifact["retrieve"] == {
        "call_count": 1,
        "completed_count": 1,
        "error_count": 0,
        "elapsed_ms": artifact["retrieve"]["elapsed_ms"],
        "error_types": [],
        "requested_top_ks": [1],
        "result_counts": [1],
    }
    assert artifact["retrieve_lexical"]["result_counts"] == [1]
    assert artifact["mark_retrieved_many"]["chunk_id_counts"] == [1]
    assert artifact["get"]["result_present"] == [True]
    assert store.retrieve.__func__ is original_retrieve.__func__
    assert store.retrieve.__self__ is original_retrieve.__self__
    store.close()


def test_performance_trace_reraises_store_errors_and_records_type() -> None:
    store = SQLiteStore(":memory:")

    def fail(*args: object, **kwargs: object) -> None:
        _ = args, kwargs
        raise RuntimeError("trace failure")

    store.get = fail  # type: ignore[method-assign]
    trace = _StoreCallTrace(store)
    with pytest.raises(RuntimeError, match="trace failure"):
        with trace:
            store.get("missing")
    assert trace.artifact()["get"]["error_types"] == ["RuntimeError"]
    assert store.get is fail
    store.close()


class _FakeChunk:
    def __init__(self, content: str, identifier: str = "runtime-fact") -> None:
        self.id = identifier
        self.version = 1
        self.content = content
        self.metadata = {
            "scope": "project",
            "scope_id": "project-01",
            "user_profile_id": "t18-synthetic-profile",
        }


class _FakeStore:
    def __init__(self) -> None:
        self.chunks: list[_FakeChunk] = []
        self.db_path = ":memory:"

    def get_all(self) -> list[_FakeChunk]:
        return list(self.chunks)


class _FakeService:
    def __init__(self, store: _FakeStore, calls: list[str]) -> None:
        self.engine = type("Engine", (), {"store": store})()
        self.calls = calls

    async def dispatch(self, operation: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(operation)
        if operation in {"capture", "remember"}:
            content = str(payload["content"])
            if operation == "remember":
                content = content.split("explicitly: ", 1)[-1]
                self.engine.store.chunks.append(_FakeChunk(content))
            return {"status": "committed"}
        if operation == "maintenance":
            return {"completed": 1}
        if operation == "retrieve":
            query = str(payload["query"])
            facts = [
                {"id": chunk.id, "version": chunk.version, "content": chunk.content}
                for chunk in self.engine.store.chunks
                if "deployment target" in query.lower()
            ]
            return {"facts": facts}
        raise ValueError(f"unsupported operation: {operation}")


def test_ollama_adapter_drives_ordered_service_boundary_and_reports_gaps() -> None:
    dataset = load_dataset(DATASET)
    session = dataset.heldout_sessions[0]
    store = _FakeStore()
    calls: list[str] = []

    def factory(profile):
        assert profile.name == "baseline"
        return _FakeService(store, calls)

    adapter = OllamaEvaluatorAdapter(factory, real_model=False)
    observation = adapter.evaluate_session(session, DEFAULT_PROFILES[0], 1)

    assert observation.adapter_kind == "test"
    assert observation.execution_status == "complete"
    assert observation.verified_model_provenance is False
    assert calls[:3] == ["remember", "capture", "maintenance"]
    assert calls.count("retrieve") == 3
    retrieve_operations = [
        item for item in observation.raw_output["operations"] if item["operation"] == "retrieve"
    ]
    assert retrieve_operations[0]["facts"]
    assert observation.answerable_hits == (True,)
    assert observation.critical_retention == (True,)
    assert observation.raw_output["unsupported_operations"] == []
    assert observation.raw_output["module_switches"]["source"] == "service_factory"


def test_ollama_adapter_preserves_missing_retrievals_and_never_fabricates_forget() -> None:
    dataset = load_dataset(DATASET)
    session = next(
        item for item in dataset.heldout_sessions if "forgotten_replay" in item.scenario_types
    )
    calls: list[str] = []

    class MissingService(_FakeService):
        async def dispatch(self, operation: str, payload: dict[str, object]) -> dict[str, object]:
            self.calls.append(operation)
            if operation == "retrieve":
                raise ConnectionError("test provider unavailable")
            if operation == "maintenance":
                return {"completed": 1}
            raise ValueError(f"unsupported operation: {operation}")

    store = _FakeStore()
    adapter = OllamaEvaluatorAdapter(lambda profile: MissingService(store, calls))
    observation = adapter.evaluate_session(session, DEFAULT_PROFILES[0], 1)

    assert observation.execution_status == "partial"
    assert observation.no_answer_correct == (None,)
    assert observation.raw_output["unsupported_operations"] == ["forget"]
    assert observation.supported_precision == ()
    assert any(item["operation"] == "retrieve" for item in observation.raw_output["errors"])


def test_critical_retention_reads_accepted_store_separately_from_retrieval() -> None:
    dataset = load_dataset(DATASET)
    session = dataset.heldout_sessions[0]
    fact = session.facts[0]
    store = _FakeStore()
    store.chunks.append(_FakeChunk(fact.content))
    service = _FakeService(store, [])
    adapter = OllamaEvaluatorAdapter(lambda profile: service)

    retained, details = adapter._critical_retention(service, session)
    query = session.queries[0]
    retrieval_hit = adapter._fact_retrieved(fact, session, {query.query_id: {"facts": []}})

    assert retained == [True]
    assert details[0]["judgement"] == "exact_content_and_scope_in_accepted_store"
    assert retrieval_hit is False


def test_extraction_judgement_rejects_empty_and_generic_substrings() -> None:
    dataset = load_dataset(DATASET)
    fact = dataset.heldout_sessions[0].facts[0]
    assert OllamaEvaluatorAdapter._fact_in_result(fact, {"facts": [{"content": ""}]}) is False
    assert (
        OllamaEvaluatorAdapter._fact_in_result(
            fact, {"facts": [{"content": fact.content.split()[-1]}]}
        )
        is False
    )


def test_shadow_length_policy_is_frozen_and_explicitly_counterfactual() -> None:
    dataset = load_dataset(DATASET)
    adapter = OllamaEvaluatorAdapter(lambda profile: None)
    session = next(
        item
        for item in dataset.heldout_sessions
        if any(e.write_mode == "automatic" for e in item.events)
    )
    short = next(event for event in session.events if event.write_mode == "automatic")
    long_event = short.__class__(
        event_id="long",
        occurred_at=short.occurred_at,
        role=short.role,
        content="x" * SHADOW_LENGTH_MIN_CHARS,
        source_kind=short.source_kind,
        write_mode="automatic",
        lifecycle_operation=short.lifecycle_operation,
        evidence_fact_ids=short.evidence_fact_ids,
    )
    assert adapter._shadow_length_keep(long_event) is True
    assert adapter._shadow_length_keep(short) is (
        len(short.content.strip()) >= SHADOW_LENGTH_MIN_CHARS
    )


def test_surprisal_metric_reads_durable_job_decisions_and_preserves_missing() -> None:
    dataset = load_dataset(DATASET)
    adapter = OllamaEvaluatorAdapter(lambda profile: None)
    session = next(
        item
        for item in dataset.heldout_sessions
        if any(e.write_mode == "automatic" for e in item.events)
    )
    events = [event for event in session.events if event.write_mode == "automatic"][:2]

    class Job:
        def __init__(self, status: str, decision: str | None) -> None:
            self.status = status
            self.result = {"decision": decision} if decision is not None else None

    jobs = {"job-1": Job("completed", "facts"), "job-2": Job("failed", None)}
    store = type("Store", (), {"get_job": lambda self, job_id: jobs.get(job_id)})()
    service = type("Service", (), {"engine": type("Engine", (), {"store": store})()})()
    operations = [
        {"operation": "capture", "event_id": events[0].event_id, "job_id": "job-1"},
        {"operation": "capture", "event_id": events[1].event_id, "job_id": "job-2"},
    ]
    values, details = adapter._surprisal_filter_decisions(service, session, operations)
    assert values[:2] == [True, None]
    assert len(values) == sum(event.write_mode == "automatic" for event in session.events)
    assert details[0]["source"] == "durable_job_result"
    assert details[1]["job_status"] == "failed"


def test_heldout_launcher_requires_exact_live_model_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "a" * 64

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            _ = exc_type, exc, tb

        def read(self) -> bytes:
            return json.dumps({"models": [{"name": "model-a", "digest": digest}]}).encode("utf-8")

    monkeypatch.setattr(
        "benchmarks.local_memory.heldout_run.urlopen", lambda request, timeout: Response()
    )
    records = _live_model_metadata("http://example.invalid", ("model-a",))
    assert records["model-a"]["digest"] == digest

    class BadResponse(Response):
        def read(self) -> bytes:
            return json.dumps({"models": [{"name": "model-a", "digest": "short"}]}).encode("utf-8")

    monkeypatch.setattr(
        "benchmarks.local_memory.heldout_run.urlopen", lambda request, timeout: BadResponse()
    )
    with pytest.raises(RuntimeError, match="invalid live digest"):
        _live_model_metadata("http://example.invalid", ("model-a",))


def test_provider_trace_is_shared_across_restart_service_instances(tmp_path: Path) -> None:
    dataset = load_dataset(DATASET)
    registry = _registry_for_dataset(dataset, tmp_path)
    factory = make_ollama_service_factory(tmp_path / "trace.db", registry)
    first = factory(DEFAULT_PROFILES[0], "trace-session")
    second = factory(DEFAULT_PROFILES[0], "trace-session")
    separate_profile = factory(DEFAULT_PROFILES[1], "trace-session")
    first_trace = first._benchmark_provider_trace
    second_trace = second._benchmark_provider_trace
    separate_trace = separate_profile._benchmark_provider_trace
    first_trace["predict_calls"].append({"restart": "before"})
    second_trace["predict_calls"].append({"restart": "after"})
    assert first_trace is second_trace
    assert [item["restart"] for item in first_trace["predict_calls"]] == ["before", "after"]
    assert first_trace is not separate_trace
    assert first_trace["service_instances"] == 2
    for service in (first, second, separate_profile):
        service.engine.store.close()


def test_extraction_schema_requires_operation_and_validates_no_fact_reason() -> None:
    with pytest.raises(ValueError):
        ExtractionResponse.model_validate(
            {
                "decision": "facts",
                "facts": [{"content": "A fact", "evidence_ids": ["event-1"]}],
            }
        )
    with pytest.raises(ValueError):
        ExtractionResponse.model_validate(
            {"decision": "no_durable_facts", "facts": [], "reason": "   "}
        )


@pytest.mark.asyncio
async def test_sleep_passes_schema_to_structured_provider() -> None:
    class StructuredProvider(LLMProvider):
        supports_structured_output = True

        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        async def predict(self, prompt: str, **kwargs: object) -> str:
            _ = prompt
            self.kwargs = kwargs
            return json.dumps(
                {
                    "decision": "facts",
                    "facts": [
                        {
                            "content": "The user uses SQLite.",
                            "evidence_ids": ["event-1"],
                            "operation": "add",
                        }
                    ],
                }
            )

        async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
            _ = prompt, labels
            return {}

    event = EpisodicEvent(
        id="event-1",
        event_fingerprint="fingerprint-1",
        role="user",
        content="I use SQLite.",
        normalized_content="i use sqlite",
        scope="project",
        scope_id="project-1",
        metadata={"user_profile_id": "profile-1"},
        occurred_at=datetime.now(timezone.utc),
    )
    provider = StructuredProvider()
    proposals = await SleepDaemon(InMemoryStore()).propose_facts([event], provider)

    assert len(proposals) == 1
    assert proposals[0].operation == "add"
    schema = provider.kwargs["format"]
    assert isinstance(schema, dict)
    branches = schema["$defs"]["FactProposal"]["oneOf"]
    assert len(branches) == 4
    assert all("operation" in branch["required"] for branch in branches)

    class MalformedProvider(StructuredProvider):
        async def predict(self, prompt: str, **kwargs: object) -> str:
            _ = prompt, kwargs
            return json.dumps(
                {
                    "decision": "facts",
                    "facts": [{"content": "The user uses SQLite.", "evidence_ids": ["event-1"]}],
                }
            )

    with pytest.raises(ValueError):
        await SleepDaemon(InMemoryStore()).propose_facts([event], MalformedProvider())


@pytest.mark.asyncio
async def test_ollama_maps_extraction_schema_to_actual_format_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, object]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"response": '{"decision":"no_durable_facts","facts":[],"reason":"none"}'}

    class Client:
        def __init__(self, *, timeout: float) -> None:
            _ = timeout

        async def __aenter__(self) -> "Client":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = exc_type, exc, tb

        async def post(self, url: str, *, json: dict[str, object]) -> Response:
            _ = url
            requests.append(json)
            return Response()

    monkeypatch.setattr("mnemos.utils.llm.httpx.AsyncClient", Client)
    schema = ExtractionResponse.model_json_schema()
    result = await OllamaProvider().predict("extract", format=schema)

    assert result.startswith("{")
    assert requests[0]["format"] == schema
