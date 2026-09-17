from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from benchmarks.local_memory.contracts import (
    FactRecord,
    HeldoutDataset,
    SessionScenario,
    SourceEvent,
)
from benchmarks.local_memory.harness import (
    DEFAULT_PROFILES,
    ExperimentConfig,
    RunObservation,
    build_completed_artifact,
    compare_observation_metrics,
)
from benchmarks.local_memory.ollama_adapter import OllamaEvaluatorAdapter


def _dataset() -> HeldoutDataset:
    session = SessionScenario(
        session_id="s1",
        split="heldout",
        host="codex",
        project_id="p1",
        facts=(
            FactRecord(
                fact_id="f1",
                content="A supported fact",
                critical=True,
                scope="project",
                source_label="synthetic_seed",
                gold_status="supported",
                evidence_event_ids=("e1",),
            ),
        ),
        queries=(),
        state_pairs=(),
        source_label="synthetic_seed",
        human_acceptance="not_run",
        events=(),
        scenario_types=(),
    )
    return HeldoutDataset(
        version="fixture",
        sessions=(session,),
        development_session_ids=(),
        digest="fixture",
        manifest={},
        source_path=None,
    )


def _observation(status: str) -> RunObservation:
    return RunObservation(
        session_id="s1",
        repetition=1 if status == "complete" else 2,
        critical_retention=(True,),
        critical_retrieval=(True,),
        execution_status=status,
        adapter_kind="test",
    )


def test_incomplete_aggregate_is_labeled_and_complete_subset_is_separate() -> None:
    artifact = build_completed_artifact(
        _dataset(),
        ExperimentConfig(profile=DEFAULT_PROFILES[0], repetitions=2),
        [_observation("complete"), _observation("partial")],
        "fixture",
    )

    assert artifact["status"] == "incomplete"
    assert artifact["measurements"]["critical_retention"]["diagnostic_when_incomplete"] is True
    complete = artifact["complete_observation_diagnostics"]
    assert complete["observation_count"] == 1
    assert complete["excluded_incomplete_observation_count"] == 1
    assert complete["status"] == "partial_available"
    assert complete["measurements"]["critical_retention"]["total_count"] == 1


def test_paired_comparison_reports_null_exclusions() -> None:
    result = compare_observation_metrics(
        {
            "raw_outputs": [
                {
                    "session_id": "s1",
                    "repetition": 1,
                    "execution_status": "complete",
                    "answerable_hits": [True, None],
                }
            ]
        },
        {
            "raw_outputs": [
                {
                    "session_id": "s1",
                    "repetition": 1,
                    "execution_status": "complete",
                    "answerable_hits": [False, True],
                }
            ]
        },
        config=ExperimentConfig(),
    )

    assert result["paired_observations"] == 1
    assert result["pair_counts"]["answerable_hits"] == {
        "total_count": 1,
        "eligible_count": 0,
        "excluded_count": 1,
        "excluded_reasons": {
            "incomplete_execution": 0,
            "missing_components": 1,
            "no_applicable_observations": 0,
        },
    }
    assert result["metrics"]["answerable_hits"]["status"] == "NOT MEASURED"


def test_rejected_lifecycle_response_marks_observation_partial() -> None:
    class Store:
        db_path = ":memory:"

        def get_all(self) -> list[Any]:
            return [
                type(
                    "Chunk",
                    (),
                    {
                        "id": "runtime-fact",
                        "version": 3,
                        "content": "A supported fact",
                        "metadata": {"scope": "project", "scope_id": "p1"},
                    },
                )()
            ]

    class Service:
        def __init__(self) -> None:
            self.engine = type("Engine", (), {"store": Store()})()

        def dispatch(self, operation: str, payload: dict[str, object]) -> dict[str, object]:
            if operation == "maintenance":
                return {"completed": 1}
            if operation == "correct":
                return {
                    "status": "rejected",
                    "fact_id": "runtime-fact",
                    "expected_version": 3,
                }
            if operation == "retrieve":
                return {"facts": []}
            raise AssertionError(operation)

    event = SourceEvent(
        event_id="e1",
        occurred_at="2026-01-01T00:01:00Z",
        role="user",
        content="A supported fact",
        source_kind="explicit_correction",
        write_mode="explicit",
        lifecycle_operation="correct",
        evidence_fact_ids=("f1",),
    )
    session = SessionScenario(**{**_dataset().sessions[0].__dict__, "events": (event,)})
    observation = OllamaEvaluatorAdapter(lambda _: Service()).evaluate_session(
        session, DEFAULT_PROFILES[0], 1
    )

    assert observation.execution_status == "partial"
    operation = observation.raw_output["operations"][1]
    assert operation["status"] == "rejected"
    assert operation["fact_id"] == "runtime-fact"
    assert operation["expected_version"] == 3
    assert observation.raw_output["lifecycle_summary"]["correct"] == {
        "attempted": 1,
        "committed": 0,
        "rejected": 1,
        "mapping_invalid": 0,
        "missing_target": 0,
    }


def test_real_retrieval_filters_foreign_store_rows_and_records_observed_candidates() -> None:
    class Store:
        def __init__(self) -> None:
            self.rows = {
                "local": SimpleNamespace(
                    metadata={
                        "user_profile_id": "profile-a",
                        "scope": "project",
                        "scope_id": "p1",
                    }
                ),
                "foreign": SimpleNamespace(
                    metadata={
                        "user_profile_id": "profile-b",
                        "scope": "project",
                        "scope_id": "p1",
                    }
                ),
            }

        def get(self, chunk_id: str) -> Any:
            return self.rows.get(chunk_id)

    class Service:
        def __init__(self) -> None:
            self.engine = SimpleNamespace(store=Store())

        def dispatch(self, operation: str, payload: dict[str, object]) -> dict[str, object]:
            assert operation == "retrieve"
            return {
                "facts": [
                    {
                        "id": "foreign",
                        "content": "foreign fact",
                        "scope": "project",
                        "scope_id": "p1",
                    },
                    {"id": "local", "content": "local fact", "scope": "project", "scope_id": "p1"},
                ]
            }

    operations: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    adapter = OllamaEvaluatorAdapter(lambda _: Service(), real_model=True)
    result = adapter._run_retrieval(
        Service(),
        "query",
        {
            "user_profile_id": "profile-a",
            "project_id": "p1",
            "workspace_id": None,
            "session_id": "s1",
            "host": "codex",
        },
        DEFAULT_PROFILES[0],
        operations,
        errors,
        "q1",
    )

    assert result is not None
    assert [item["id"] for item in result["facts"]] == ["local"]
    assert [item["id"] for item in operations[0]["observed_facts"]] == ["foreign", "local"]
    assert operations[0]["scope_validation"]["foreign_count"] == 1
    assert operations[0]["scope_validation"]["verified_count"] == 1
    assert errors[0]["error"] == "scope_validation_failed"


def test_artifact_aggregates_lifecycle_operations() -> None:
    observations = []
    for repetition in range(1, 4):
        observation = _observation("complete")
        summary = {
            "correct": {
                "attempted": 1,
                "committed": 1,
                "rejected": 0,
                "mapping_invalid": 0,
                "missing_target": 0,
            },
            "forget": {
                "attempted": 1,
                "committed": 0,
                "rejected": 0,
                "mapping_invalid": 0,
                "missing_target": 1,
            },
        }
        observations.append(
            RunObservation(
                **{
                    **observation.__dict__,
                    "repetition": repetition,
                    "raw_output": {"lifecycle_summary": summary},
                }
            )
        )
    artifact = build_completed_artifact(
        _dataset(), ExperimentConfig(profile=DEFAULT_PROFILES[0]), observations, "fixture"
    )
    assert artifact["lifecycle_operations"]["correct"]["committed"] == 3
    assert artifact["lifecycle_operations"]["forget"]["missing_target"] == 3
