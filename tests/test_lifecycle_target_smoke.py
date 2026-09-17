from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.local_memory.lifecycle_target_smoke import (
    DEFAULT_DATASET,
    _diagnose_observation,
    load_target_scenarios,
    run_lifecycle_target_smoke,
)
from benchmarks.local_memory.harness import RunObservation


def _observation(session_id: str, *, complete: bool = True) -> RunObservation:
    if session_id == "development-correct-restart":
        operations = [
            {"operation": "correct", "status": "committed", "fact_id": "runtime-id"},
            {
                "operation": "maintenance",
                "before": "correct",
                "status": "completed",
                "jobs": [{"kind": "correct", "status": "completed"}],
            },
            {"operation": "restart", "status": "completed"},
            {
                "operation": "retrieve",
                "query_id": "correct-retrieval-query",
                "facts": [
                    {
                        "id": "runtime-id",
                        "version": 2,
                        "content": "The cache layer uses SQLite for session tokens.",
                    }
                ],
            },
        ]
    else:
        operations = [
            {"operation": "remember", "status": "committed"},
            {
                "operation": "maintenance",
                "before": "forget",
                "status": "completed",
                "jobs": [{"kind": "remember", "status": "completed"}],
            },
            {"operation": "forget", "status": "committed", "forgotten": True},
            {"operation": "remember", "status": "committed", "processing_status": "suppressed"},
            {"operation": "restart", "status": "completed"},
            {"operation": "retrieve", "query_id": "forget-abstention-query", "facts": []},
        ]
    if not complete:
        operations = [row for row in operations if row.get("operation") != "maintenance"]
    return RunObservation(
        session_id=session_id,
        repetition=1,
        execution_status="complete" if complete else "partial",
        raw_output={"operations": operations, "errors": []},
    )


class _FakeAdapter:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def evaluate_session(self, session: object, profile: object, repetition: int) -> RunObservation:
        del profile, repetition
        return _observation(getattr(session, "session_id"))


def _fake_factory(*_args: object, **_kwargs: object) -> object:
    return object()


def _fake_model_metadata(endpoint: str, names: tuple[str, ...]) -> dict[str, dict[str, object]]:
    del endpoint
    return {name: {"name": name, "digest": "a" * 64} for name in names}


def test_authored_fixture_has_two_disjoint_lifecycle_scenarios() -> None:
    scenarios = load_target_scenarios(DEFAULT_DATASET)
    assert {scenario.split for scenario in scenarios} == {"development"}
    assert len({fact.fact_id for scenario in scenarios for fact in scenario.facts}) == 3
    replay = [event for event in scenarios[1].events if event.lifecycle_operation == "remember"]
    assert len(replay) == 2
    assert replay[0].event_id == replay[1].event_id


def test_diagnostic_uses_durable_operation_gates_without_quality_match() -> None:
    scenarios = load_target_scenarios(DEFAULT_DATASET)
    correct = _diagnose_observation(scenarios[0], _observation(scenarios[0].session_id))
    assert correct["status"] == "complete"
    assert correct["correct_job_processed"] is True
    assert correct["current_version_observed"] is True

    forget = _diagnose_observation(scenarios[1], _observation(scenarios[1].session_id))
    assert forget["status"] == "complete"
    assert forget["replay_suppressed"] is True
    assert forget["abstain_observed"] is True

    malformed = _observation(forget["session_id"])
    malformed_raw = dict(malformed.raw_output)
    malformed_operations = [dict(row) for row in malformed_raw["operations"]]
    malformed_operations[-1]["facts"] = None
    malformed_raw["operations"] = malformed_operations
    malformed = RunObservation(
        session_id="development-forget-replay",
        repetition=1,
        execution_status="complete",
        raw_output=malformed_raw,
    )
    assert _diagnose_observation(scenarios[1], malformed)["status"] == "blocked"


def test_runner_writes_blocked_report_for_unprocessed_correction(tmp_path: Path) -> None:
    class BlockedAdapter(_FakeAdapter):
        def evaluate_session(
            self, session: object, profile: object, repetition: int
        ) -> RunObservation:
            del profile, repetition
            return _observation(getattr(session, "session_id"), complete=False)

    report = run_lifecycle_target_smoke(
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        endpoint="http://example.invalid",
        model="model-for-test",
        embedding_model="embedding-for-test",
        service_factory_builder=_fake_factory,
        adapter_type=BlockedAdapter,
        model_metadata_loader=_fake_model_metadata,
    )
    assert report["status"] == "blocked"
    assert report["blocked_runs"] == 4
    artifact = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert artifact["quality"] == "NOT MEASURED"
    assert artifact["human_acceptance"] == "NOT RUN"


def test_runner_fails_closed_without_live_model_digests(tmp_path: Path) -> None:
    calls: list[str] = []

    def unavailable(_endpoint: str, _names: tuple[str, ...]) -> dict[str, dict[str, object]]:
        calls.append("metadata")
        raise RuntimeError("model tags unavailable")

    def must_not_start(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("must not start service")

    report = run_lifecycle_target_smoke(
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        endpoint="http://example.invalid",
        model="model-for-test",
        embedding_model="embedding-for-test",
        service_factory_builder=must_not_start,
        adapter_type=_FakeAdapter,
        model_metadata_loader=unavailable,
    )
    assert report["status"] == "blocked"
    assert calls == ["metadata"]


def test_loader_rejects_missing_or_unknown_fixture(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": "wrong", "scenarios": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_target_scenarios(path)
