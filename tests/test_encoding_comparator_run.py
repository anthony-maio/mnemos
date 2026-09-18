import json
from pathlib import Path
from typing import Any, NoReturn

import pytest

from benchmarks.local_memory.encoding_comparator_run import (
    PROTOCOL_VERSION,
    _attach_comparisons,
    build_binding,
    compare_filter_decisions,
    run_comparator,
    validate_existing_binding,
)
from benchmarks.local_memory.encoding_comparator import COMPARATOR_PROFILES
from benchmarks.local_memory.harness import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks" / "local_memory" / "datasets" / "local-memory-heldout-v3.jsonl"
PROTOCOL = ROOT / "docs" / "plans" / "2026-09-18-encoding-comparison-protocol-v1.md"


def test_prepare_only_writes_three_arms_without_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_live_metadata(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("prepare-only must not contact Ollama")

    monkeypatch.setattr(
        "benchmarks.local_memory.encoding_comparator_run._live_model_metadata",
        fail_live_metadata,
    )
    result = run_comparator(
        dataset_path=DATASET,
        protocol_path=PROTOCOL,
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        endpoint="http://127.0.0.1:1",
        model="qwen3.5:9b",
        repetitions=3,
        prepare_only=True,
    )
    assert result["manifest"]["status"] == "not_run"
    assert result["manifest"]["binding"]["protocol"]["version"] == PROTOCOL_VERSION
    assert set(result["reports"]) == {profile.name for profile in COMPARATOR_PROFILES}
    for name, report in result["reports"].items():
        assert report["status"] == "not_run"
        assert report["profile"]["name"] == name
        assert report["gates"]["real_model"] == "NOT RUN"
        assert (tmp_path / "output" / f"{name}.json").is_file()


def test_prepare_only_refuses_existing_output_or_work_directory(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_comparator(
            dataset_path=DATASET,
            protocol_path=PROTOCOL,
            output_dir=output,
            work_dir=tmp_path / "work",
            endpoint="http://127.0.0.1:1",
            model="qwen3.5:9b",
            prepare_only=True,
        )


def test_resume_binding_rejects_protocol_or_option_drift(tmp_path: Path) -> None:
    binding = {
        "namespace": "encoding-comparator-v1",
        "options_sha256": "one",
        "protocol": {"sha256": "protocol-one"},
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"binding": binding}), encoding="utf-8")
    assert validate_existing_binding(manifest, binding)["binding"] == binding
    with pytest.raises(ValueError, match="binding drift"):
        validate_existing_binding(manifest, {**binding, "options_sha256": "two"})


def test_three_arm_comparison_reports_treatment_minus_full_sign() -> None:
    config = ExperimentConfig(profile=COMPARATOR_PROFILES[0], repetitions=3)
    rows = {
        "session_id": "s1",
        "repetition": 1,
        "execution_status": "complete",
        "answerable_hits": [True],
        "no_answer_correct": [],
        "critical_retention": [],
        "critical_retrieval": [],
    }
    full: dict[str, Any] = {"raw_outputs": [rows.copy()]}
    length: dict[str, Any] = {"raw_outputs": [{**rows, "answerable_hits": [False]}]}
    control: dict[str, Any] = {"raw_outputs": [rows.copy()]}
    reports: dict[str, Any] = {
        "full": full,
        "length-noise": length,
        "ablate-surprisal_gating": control,
    }
    _attach_comparisons(reports, config=config)
    comparator = full["comparator"]
    assert comparator["ungated_control"] == "ablate-surprisal_gating"
    assert set(comparator["paired_comparisons"]) == {
        "length-noise",
        "ablate-surprisal_gating",
    }
    assert (
        comparator["paired_comparisons"]["length-noise"]["metrics"]["answerable_hits"]["estimate"]
        == -1.0
    )


def test_binding_includes_protocol_policy_and_options(tmp_path: Path) -> None:
    from benchmarks.local_memory.contracts import load_dataset

    dataset = load_dataset(DATASET)
    binding = build_binding(
        dataset=dataset,
        dataset_path=DATASET,
        protocol_path=PROTOCOL,
        code_sha256="a" * 64,
        options={"model": "qwen3.5:9b", "think": False},
    )
    assert binding["protocol"]["sha256"]
    assert binding["policy"]["version"]
    assert binding["options_sha256"]


@pytest.mark.parametrize("interrupt", [False, True])
def test_execution_preserves_first_arm_on_failure_and_refuses_foreign_report(
    tmp_path, monkeypatch, interrupt
):
    from benchmarks.local_memory import encoding_comparator_run as runner

    monkeypatch.setattr(
        runner,
        "_live_model_metadata",
        lambda endpoint, names: {
            name: {"name": name, "digest": str(index + 1) * 64} for index, name in enumerate(names)
        },
    )
    calls = []

    class HarnessFixture:
        def __init__(self, dataset, config):
            self.dataset, self.config = dataset, config

        def run(self, adapter, checkpoint_path, run_id):
            calls.append(self.config.profile.name)
            service = adapter._new_service(self.config.profile, "runner-fixture")
            try:
                assert service.engine.config.modules.sleep is True
                assert service.engine.config.modules.surprisal == (
                    self.config.profile.name == "full"
                )
            finally:
                service.engine.store.close()
            if interrupt and self.config.profile.name == "ablate-surprisal_gating":
                raise RuntimeError("fixture interrupted after first arm")
            return {
                "run_id": run_id,
                "status": "completed",
                "raw_outputs": [],
                "dataset": {"sha256": self.dataset.digest},
                "config": self.config.as_dict(),
                "profile": self.config.profile.as_dict(),
            }

    monkeypatch.setattr(runner, "ExperimentHarness", HarnessFixture)
    args = dict(
        dataset_path=DATASET,
        protocol_path=PROTOCOL,
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        endpoint="http://127.0.0.1:1",
        model="fixture-model",
    )
    if interrupt:
        with pytest.raises(RuntimeError, match="fixture interrupted"):
            runner.run_comparator(**args)
    else:
        result = runner.run_comparator(**args)
        length = result["reports"]["length-noise"]
        assert length["gates"]["length_filter_decision"] == "NOT VERIFIED"
        assert length["applied_length_coverage"]["missing"] > 0
    expected_calls = ["full", "ablate-surprisal_gating"] + ([] if interrupt else ["length-noise"])
    assert calls == expected_calls
    manifest_path = tmp_path / "work/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == ("failed" if interrupt else "completed")
    assert manifest["attempts"][-1]["status"] == ("failed" if interrupt else "completed")
    first_report = tmp_path / "output/full.json"
    report = json.loads(first_report.read_text())
    report["run_manifest"]["binding"]["code_sha256"] = "foreign-source"
    first_report.write_text(json.dumps(report))
    preserved = first_report.read_bytes()
    with pytest.raises(ValueError, match="report.*binding"):
        runner.run_comparator(**args, resume=True)
    assert first_report.read_bytes() == preserved
    assert calls == expected_calls


def test_filter_comparison_keeps_unobserved_decisions_out_of_counts() -> None:
    base = {
        "raw_outputs": [
            {
                "session_id": "s1",
                "repetition": 1,
                "raw_output": {
                    "encoding_policy_decisions": [
                        {"event_id": "e1", "policy": "surprisal", "kept": True},
                        {"event_id": "e2", "policy": "surprisal", "kept": None},
                    ]
                },
            }
        ]
    }
    treatment = {
        "raw_outputs": [
            {
                "session_id": "s1",
                "repetition": 1,
                "raw_output": {
                    "encoding_policy_decisions": [
                        {"event_id": "e1", "policy": "length_noise", "kept": False},
                        {"event_id": "e2", "policy": "length_noise", "kept": True},
                    ]
                },
            }
        ]
    }
    comparison = compare_filter_decisions(base, treatment)
    assert comparison["paired_event_count"] == 2
    assert comparison["comparable_event_count"] == 1
    assert comparison["unobserved_event_count"] == 1
    assert comparison["decision_pairs"]["surprisal_kept_true__length_noise_kept_false"] == 1
