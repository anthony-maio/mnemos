import json
from pathlib import Path
from typing import Any, NoReturn

import pytest

from benchmarks.local_memory import retrieval_comparison_run as runner
from benchmarks.local_memory.contracts import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks" / "local_memory" / "datasets" / "local-memory-heldout-v3.jsonl"
PROTOCOL = ROOT / "docs" / "plans" / "2026-09-18-retrieval-comparison-protocol-v1.md"


def test_prepare_only_is_offline_and_binds_all_strategies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_live(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("prepare-only must not contact Ollama")

    monkeypatch.setattr(runner, "run_heldout", fail_live)
    result = runner.run_comparator(
        dataset_path=DATASET,
        protocol_path=PROTOCOL,
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        endpoint="http://127.0.0.1:1",
        model="fixture-model",
        prepare_only=True,
    )
    assert result["manifest"]["status"] == "not_run"
    assert result["manifest"]["strategies"] == list(runner.STRATEGIES)
    report = result["report"]
    assert report["gates"] == {
        "real_model": "NOT RUN",
        "retrieval_comparison": "NOT RUN",
        "human_acceptance": "NOT RUN",
    }
    assert (tmp_path / "output" / "retrieval-comparison.json").is_file()


def test_resume_rejects_binding_or_path_drift_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "output"
    work = tmp_path / "work"
    runner.run_comparator(
        dataset_path=DATASET,
        protocol_path=PROTOCOL,
        output_dir=output,
        work_dir=work,
        endpoint="http://127.0.0.1:1",
        model="fixture-model",
        prepare_only=True,
    )
    before = (work / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="binding drift"):
        runner.run_comparator(
            dataset_path=DATASET,
            protocol_path=PROTOCOL,
            output_dir=tmp_path / "other-output",
            work_dir=work,
            endpoint="http://127.0.0.1:1",
            model="fixture-model",
            resume=True,
        )
    assert (work / "manifest.json").read_bytes() == before


def test_execution_injects_full_profile_adapter_and_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> dict[str, dict[str, Any]]:
        captured.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        (output_dir / "full.raw-outputs.jsonl").write_text("", encoding="utf-8")
        (output_dir / "full.json").write_text(
            json.dumps(
                {
                    "run_id": f"{runner.RUN_NAMESPACE}-local-memory-heldout-v3-full",
                    "status": "completed",
                    "raw_outputs": [],
                }
            ),
            encoding="utf-8",
        )
        return {"full": {"status": "completed", "raw_outputs": []}}

    monkeypatch.setattr(runner, "run_heldout", fake_run)
    result = runner.run_comparator(
        dataset_path=DATASET,
        protocol_path=PROTOCOL,
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        endpoint="http://127.0.0.1:1",
        model="fixture-model",
    )
    assert captured["profile_names"] == {"full"}
    assert captured["run_namespace"] == runner.RUN_NAMESPACE
    assert captured["adapter_type"] is runner.RetrievalComparatorAdapter
    assert captured["experiment_binding"]["strategy_config_sha256"]
    assert result["manifest"]["status"] == "incomplete"
    summary_path = tmp_path / "output" / "retrieval-comparison-summary.json"
    with pytest.raises(ValueError, match="incomplete or failed"):
        runner.run_comparator(
            dataset_path=DATASET,
            protocol_path=PROTOCOL,
            output_dir=tmp_path / "output",
            work_dir=tmp_path / "work",
            endpoint="http://127.0.0.1:1",
            model="fixture-model",
            resume=True,
        )
    summary_before = summary_path.read_bytes()
    (tmp_path / "output" / "full.raw-outputs.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete or failed"):
        runner.run_comparator(
            dataset_path=DATASET,
            protocol_path=PROTOCOL,
            output_dir=tmp_path / "output",
            work_dir=tmp_path / "work",
            endpoint="http://127.0.0.1:1",
            model="fixture-model",
            resume=True,
        )
    assert summary_path.read_bytes() == summary_before


def test_summary_excludes_missing_and_partial_rows_from_hit_metrics() -> None:
    dataset = load_dataset(DATASET)
    answerable = next(
        (query, session)
        for session in dataset.heldout_sessions
        for query in session.queries
        if query.kind == "answerable"
    )
    no_answer = next(
        (query, session)
        for session in dataset.heldout_sessions
        for query in session.queries
        if query.kind == "no_answer"
    )

    def strategy(status: str, hit: bool | None) -> dict[str, Any]:
        return {"status": status, "quality_hit": hit, "latency_ms": 4.0, "errors": []}

    report = {
        "raw_outputs": [
            {
                "session_id": answerable[1].session_id,
                "repetition": 1,
                "execution_status": "complete",
                "verified_model_provenance": True,
                "verified_embedding_provenance": True,
                "raw_output": {
                    "retrieval_comparisons": [
                        {
                            "query_id": answerable[0].query_id,
                            "quality_kind": answerable[0].kind,
                            "strategies": {
                                "full": strategy("complete", True),
                                "vector": strategy("complete", False),
                                "lexical": strategy("partial", None),
                                "hybrid": strategy("complete", True),
                            },
                        },
                        {
                            "query_id": no_answer[0].query_id,
                            "quality_kind": no_answer[0].kind,
                            "strategies": {
                                "full": strategy("complete", False),
                                "vector": strategy("complete", False),
                                "lexical": strategy("complete", True),
                                "hybrid": strategy("complete", False),
                            },
                        },
                    ]
                },
            }
        ]
    }
    # The two rows intentionally belong to different sessions.
    first_output = report["raw_outputs"][0]
    second_comparison = first_output["raw_output"]["retrieval_comparisons"].pop()
    report["raw_outputs"].append(
        {
            "session_id": no_answer[1].session_id,
            "repetition": 1,
            "execution_status": "complete",
            "verified_model_provenance": True,
            "verified_embedding_provenance": True,
            "raw_output": {"retrieval_comparisons": [second_comparison]},
        }
    )
    summary = runner.summarize_strategy_reports(report, dataset=dataset, repetitions=1)
    answerable_summary = summary["quality_kinds"]["answerable"]
    assert answerable_summary["strategies"]["lexical"]["measured_count"] == 0
    assert (
        answerable_summary["strategies"]["lexical"]["missing_count"]
        == answerable_summary["expected_count"]
    )
    assert answerable_summary["strategies"]["lexical"]["status"] == "NOT MEASURED"
    assert answerable_summary["paired_to_full"]["vector"]["estimate"] == -1.0
    no_answer_summary = summary["quality_kinds"]["no_answer"]
    assert no_answer_summary["paired_to_full"]["lexical"]["estimate"] == 1.0


def test_summary_does_not_promote_all_missing_to_a_result() -> None:
    dataset = load_dataset(DATASET)
    query = dataset.heldout_sessions[0].queries[0]
    session_id = dataset.heldout_sessions[0].session_id
    report = {
        "raw_outputs": [
            {
                "session_id": session_id,
                "repetition": 1,
                "raw_output": {
                    "retrieval_comparisons": [
                        {
                            "query_id": query.query_id,
                            "quality_kind": query.kind,
                            "strategies": {
                                name: {"status": "partial", "quality_hit": None}
                                for name in runner.STRATEGIES
                            },
                        }
                    ]
                },
            }
        ]
    }
    summary = runner.summarize_strategy_reports(report, dataset=dataset, repetitions=1)
    assert summary["quality_kinds"][query.kind]["strategies"]["full"]["status"] == "NOT MEASURED"
    assert (
        summary["quality_kinds"][query.kind]["paired_to_full"]["vector"]["status"] == "NOT MEASURED"
    )


def test_summary_rejects_duplicate_and_unknown_query_identities() -> None:
    dataset = load_dataset(DATASET)
    session = dataset.heldout_sessions[0]
    query = session.queries[0]
    comparison = {
        "query_id": query.query_id,
        "quality_kind": query.kind,
        "strategies": {name: {"status": "partial"} for name in runner.STRATEGIES},
    }
    duplicate = {
        "raw_outputs": [
            {
                "session_id": session.session_id,
                "repetition": 1,
                "raw_output": {"retrieval_comparisons": [comparison, comparison.copy()]},
            }
        ]
    }
    with pytest.raises(ValueError, match="duplicate"):
        runner.summarize_strategy_reports(duplicate, dataset=dataset, repetitions=1)
    unknown = {
        "raw_outputs": [
            {
                "session_id": session.session_id,
                "repetition": 1,
                "raw_output": {
                    "retrieval_comparisons": [{**comparison, "query_id": "unknown-query"}]
                },
            }
        ]
    }
    with pytest.raises(ValueError, match="unknown"):
        runner.summarize_strategy_reports(unknown, dataset=dataset, repetitions=1)


def test_summary_accepts_adapter_state_pair_identity_suffixes() -> None:
    dataset = load_dataset(DATASET)
    session = next(session for session in dataset.heldout_sessions if session.state_pairs)
    pair = session.state_pairs[0]
    report = {
        "raw_outputs": [
            {
                "session_id": session.session_id,
                "repetition": 1,
                "execution_status": "complete",
                "verified_model_provenance": True,
                "verified_embedding_provenance": True,
                "raw_output": {
                    "retrieval_comparisons": [
                        {
                            "query_id": pair.pair_id + ":semantic",
                            "quality_kind": "state_pair_semantic",
                            "strategies": {
                                name: {"status": "complete", "quality_hit": True}
                                for name in runner.STRATEGIES
                            },
                        }
                    ]
                },
            }
        ]
    }
    summary = runner.summarize_strategy_reports(report, dataset=dataset, repetitions=1)
    assert summary["quality_kinds"]["state_pair_semantic"]["observed_rows"] == 1
    assert (
        summary["quality_kinds"]["state_pair_semantic"]["strategies"]["full"]["measured_count"] == 1
    )


@pytest.mark.parametrize(
    "changed", ["full.json", "full.raw-outputs.jsonl", "retrieval-comparison-summary.json"]
)
def test_completed_artifact_validation_rejects_changed_evidence(tmp_path, changed):
    from benchmarks.local_memory.artifacts import ArtifactRecorder, file_digest

    binding = {"dataset": {"version": "fixture"}}
    full_path = tmp_path / "full.json"
    raw_path = tmp_path / "full.raw-outputs.jsonl"
    summary_path = tmp_path / "retrieval-comparison-summary.json"
    record = {"fixture": True}
    ArtifactRecorder(full_path).write_run(
        {
            "status": "completed",
            "run_id": f"{runner.RUN_NAMESPACE}-fixture-full",
            "raw_outputs": [record],
        },
        [record],
    )
    ArtifactRecorder(summary_path).write(
        {
            "binding": binding,
            "summary": {"coverage_status": "complete"},
            "artifact_binding": {
                "full_report": file_digest(full_path),
                "full_raw_outputs": file_digest(raw_path),
            },
        }
    )
    manifest = {
        "status": "completed",
        "summary_sha256": file_digest(summary_path)["sha256"],
        "full_report_sha256": file_digest(full_path)["sha256"],
        "full_raw_outputs_sha256": file_digest(raw_path)["sha256"],
    }
    assert runner._validate_completed_artifacts(manifest, binding, tmp_path)["binding"] == binding
    with (tmp_path / changed).open("a", encoding="utf-8") as handle:
        handle.write(" ")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="drift"):
        runner._validate_completed_artifacts(manifest, binding, tmp_path)
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


def test_unverified_provenance_prevents_completed_coverage():
    report = {
        "raw_outputs": [
            {
                "session_id": "fixture",
                "repetition": 1,
                "execution_status": "complete",
                "verified_model_provenance": False,
                "verified_embedding_provenance": False,
                "raw_output": {
                    "retrieval_comparisons": [
                        {
                            "query_id": "query",
                            "quality_kind": "answerable",
                            "strategies": {
                                name: {"status": "complete", "quality_hit": True}
                                for name in runner.STRATEGIES
                            },
                        }
                    ]
                },
            }
        ]
    }
    summary = runner.summarize_strategy_reports(report)
    assert summary["coverage_status"] == "incomplete"
    assert summary["provenance"]["unverified_observations"] == 1
    assert summary["quality_kinds"]["answerable"]["strategies"]["full"]["measured_count"] == 0
