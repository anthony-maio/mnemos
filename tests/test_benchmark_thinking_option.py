from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from benchmarks.local_memory import ollama_adapter
from benchmarks.local_memory.harness import DEFAULT_PROFILES
from benchmarks.local_memory import heldout_run, lifecycle_target_smoke, module_mechanism_run
from mnemos.scopes import ScopeRegistry


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("FALSE", False),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("max", "max"),
    ],
)
def test_parse_ollama_think_uses_documented_values(raw: str, expected: object) -> None:
    assert ollama_adapter.parse_ollama_think(raw) == expected


def test_parse_ollama_think_rejects_undocumented_value() -> None:
    with pytest.raises(ValueError, match="think must be"):
        ollama_adapter.parse_ollama_think("auto")


def test_default_options_omit_think_and_explicit_mode_changes_fingerprint() -> None:
    default: dict[str, object] = {"temperature": 0.0, "max_tokens": 512}
    explicit = dict(default)
    ollama_adapter.add_ollama_think(explicit, "low")

    assert "think" not in default
    assert explicit["think"] == "low"
    assert ollama_adapter.ollama_options_sha256(default) != ollama_adapter.ollama_options_sha256(
        explicit
    )


@pytest.mark.parametrize("think", [False, True, "low", "medium", "high", "max"])
def test_service_factory_wires_think_and_records_it(tmp_path: Path, think: object) -> None:
    factory = ollama_adapter.make_ollama_service_factory(
        tmp_path / "memory.db",
        ScopeRegistry(),
        base_url="http://unused.invalid",
        think=think,  # type: ignore[arg-type]
    )
    service = factory(DEFAULT_PROFILES[0], "thinking-test")
    try:
        provider = service.engine.llm._provider
        trace = service._benchmark_provider_trace
        assert provider.think == think
        assert trace["config"]["think"] == think
    finally:
        service.engine.store.close()


def test_service_factory_default_trace_omits_think(tmp_path: Path) -> None:
    factory = ollama_adapter.make_ollama_service_factory(
        tmp_path / "memory.db", ScopeRegistry(), base_url="http://unused.invalid"
    )
    service = factory(DEFAULT_PROFILES[0], "default-thinking-test")
    try:
        assert "think" not in service._benchmark_provider_trace["config"]
        assert service.engine.llm._provider.think is None
    finally:
        service.engine.store.close()


def test_runner_clis_forward_explicit_think_without_starting_services(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def heldout_stub(**kwargs: object) -> dict[str, dict[str, str]]:
        seen["heldout"] = kwargs["think"]
        return {"baseline": {"status": "complete"}}

    def mechanism_stub(**kwargs: object) -> dict[str, str]:
        seen["mechanism"] = kwargs["think"]
        return {"status": "prepared", "binding": "fixture"}

    def lifecycle_stub(**kwargs: object) -> dict[str, object]:
        seen["lifecycle"] = kwargs["think"]
        return {"status": "complete", "blocked_runs": 0}

    monkeypatch.setattr(heldout_run, "run_heldout", heldout_stub)
    monkeypatch.setattr(module_mechanism_run, "run_mechanisms", mechanism_stub)
    monkeypatch.setattr(lifecycle_target_smoke, "run_lifecycle_target_smoke", lifecycle_stub)

    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "heldout_run",
            "--output-dir",
            str(output_dir),
            "--work-dir",
            str(work_dir),
            "--think",
            "false",
        ],
    )
    assert heldout_run.main() == 0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "module_mechanism_run",
            "--output-dir",
            str(output_dir),
            "--work-dir",
            str(work_dir),
            "--think",
            "low",
        ],
    )
    assert module_mechanism_run.main() == 0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lifecycle_target_smoke",
            "--endpoint",
            "http://unused.invalid",
            "--model",
            "fixture",
            "--embedding-model",
            "fixture-embedding",
            "--output-dir",
            str(output_dir),
            "--work-dir",
            str(work_dir),
            "--think",
            "high",
        ],
    )
    assert lifecycle_target_smoke.main() == 0
    assert seen == {"heldout": False, "mechanism": "low", "lifecycle": "high"}


def test_mechanism_manifest_binding_rejects_changed_think_without_model_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from benchmarks.local_memory import module_mechanism

    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "version": "fixture",
                "graph_cases": [{"id": "graph-1", "split": "development"}],
                "affective_pairs": [{"id": "affective-1", "split": "development"}],
            }
        ),
        encoding="utf-8",
    )
    protocol_path = tmp_path / "protocol.md"
    protocol_path.write_text("fixture protocol\n", encoding="utf-8")
    metadata = lambda endpoint, names: {name: {"name": name, "digest": "a" * 64} for name in names}
    monkeypatch.setattr(module_mechanism_run, "_live_model_metadata", metadata)
    monkeypatch.setattr(module_mechanism, "load_mechanism_dataset", lambda path: None)

    def prepare(selected: Path, temporary: Path, cache: object) -> None:
        temporary.write_text(json.dumps({"manifest": {"binding": cache.binding}}), encoding="utf-8")

    monkeypatch.setattr(module_mechanism, "prepare_frozen_embeddings", prepare)
    common = {
        "dataset_path": dataset_path,
        "protocol_path": protocol_path,
        "output_dir": tmp_path / "output",
        "work_dir": tmp_path / "work",
        "endpoint": "http://unused.invalid",
        "model": "fixture",
        "embedding_model": "fixture-embedding",
        "split": "development",
        "repetitions": 3,
        "prepare_only": True,
    }
    first = module_mechanism_run.run_mechanisms(**common)
    assert "think" not in first["options"]
    with pytest.raises(ValueError, match="changed provenance"):
        module_mechanism_run.run_mechanisms(**common, think=False)


def test_lifecycle_blocked_report_keeps_explicit_think_configuration(tmp_path: Path) -> None:
    def unavailable_metadata(endpoint: str, names: tuple[str, ...]) -> dict[str, dict[str, object]]:
        raise RuntimeError("offline fixture")

    report = lifecycle_target_smoke.run_lifecycle_target_smoke(
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        endpoint="http://unused.invalid",
        model="fixture",
        embedding_model="fixture-embedding",
        profile_names=("baseline",),
        model_metadata_loader=unavailable_metadata,
        think="medium",
    )
    assert report["status"] == "blocked"
    assert report["model_pins"]["options"]["think"] == "medium"
    assert report["model_pins"]["options_sha256"]
