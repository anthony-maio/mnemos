"""The held-out harness can run a judged service and label its provenance honestly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.local_memory.contracts import load_dataset
from benchmarks.local_memory.harness import DEFAULT_PROFILES
from benchmarks.local_memory.heldout_run import (
    _live_typesafe_model,
    _registry_for_dataset,
    run_heldout,
)
from benchmarks.local_memory.ollama_adapter import (
    OllamaEvaluatorAdapter,
    make_ollama_service_factory,
)
from mnemos.utils.typesafe import TypeSafeJudge

DATASET = (
    Path(__file__).parents[1] / "benchmarks/local_memory/datasets/local-memory-heldout-v3.jsonl"
)


def test_the_factory_builds_a_traced_judge_and_judged_config(tmp_path: Path) -> None:
    dataset = load_dataset(DATASET)
    registry = _registry_for_dataset(dataset, tmp_path)
    factory = make_ollama_service_factory(
        tmp_path / "judged.db",
        registry,
        extraction_mode="judged",
        surprisal_mode="novelty",
        judge_provider="typesafe",
        typesafe_api_key="k",
    )
    service = factory(DEFAULT_PROFILES[1], "run")
    try:
        engine = service.engine
        assert engine.config.sleep.extraction_mode == "judged"
        assert engine.config.surprisal.mode == "novelty"
        assert engine.config.retrieval.query_state_source == "recent_state"
        assert engine.judge.provider == "typesafe"
        assert isinstance(engine.judge._judge, TypeSafeJudge)  # type: ignore[attr-defined]
        trace = service._benchmark_provider_trace
        assert trace["judge_calls"] == []
        assert trace["config"]["judge_provider"] == "typesafe"
        assert trace["config"]["judge_model"] == "jev-latest"
    finally:
        service.engine.store.close()

    with pytest.raises(ValueError, match="need a judge_provider"):
        make_ollama_service_factory(tmp_path / "x.db", registry, extraction_mode="judged")
    with pytest.raises(ValueError, match="typesafe_api_key"):
        make_ollama_service_factory(tmp_path / "y.db", registry, judge_provider="typesafe")


def test_judge_calls_count_as_cognition_and_carry_cost() -> None:
    adapter = OllamaEvaluatorAdapter(lambda profile, run_key=None: None, real_model=True)

    class Service:
        _benchmark_provider_trace: dict[str, Any] = {
            "predict_calls": [],
            "classify_calls": [],
            "embed_calls": [{"x": 1}],
            "judge_calls": [
                {"provider": "typesafe", "input_tokens": 1000, "cost_usd": 0.000042},
                {"provider": "typesafe", "input_tokens": 500, "cost_usd": 0.000021},
            ],
        }

    assert adapter._trace_has_cognition_calls(Service())
    assert adapter._judge_cost(Service()) == pytest.approx(0.000063)

    class Silent:
        _benchmark_provider_trace = {"predict_calls": [], "classify_calls": [], "judge_calls": []}

    assert not adapter._trace_has_cognition_calls(Silent())
    assert adapter._judge_cost(Silent()) is None


def test_hosted_model_identity_is_a_labeled_version_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"model": "jev-1.13.0", "answers": {"ready": {"type": "noul", "noul": 0.9}}}

    def post(url: str, **kwargs: Any) -> Response:
        assert url == "https://api.typesafe.ai/v1/systemone"
        assert kwargs["headers"]["Authorization"] == "Bearer k"
        assert kwargs["json"]["model"] == "jev-latest"
        assert kwargs["timeout"] == 30.0
        return Response()

    monkeypatch.setattr("benchmarks.local_memory.heldout_run.httpx.post", post)
    record = _live_typesafe_model("https://api.typesafe.ai/v1", "k", "jev-latest")
    assert record["name"] == "jev-1.13.0" and record["requested"] == "jev-latest"
    assert record["identity_kind"] == "vendor_version_string"
    assert len(record["digest"]) == 64
    assert OllamaEvaluatorAdapter._provenance_complete(record, "jev-1.13.0")


def test_run_heldout_binds_the_judge_into_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "benchmarks.local_memory.heldout_run._live_model_metadata",
        lambda endpoint, names: {name: {"name": name, "digest": "a" * 64} for name in names},
    )
    monkeypatch.setattr(
        "benchmarks.local_memory.heldout_run._live_typesafe_model",
        lambda base_url, key, model: {
            "name": "jev-1.13.0",
            "model": "jev-1.13.0",
            "digest": "b" * 64,
            "identity_kind": "vendor_version_string",
        },
    )
    monkeypatch.setattr(
        "benchmarks.local_memory.heldout_run._code_provenance", lambda root: ("c" * 64, [])
    )
    captured: dict[str, Any] = {}

    def fake_factory(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return None

    monkeypatch.setattr(
        "benchmarks.local_memory.heldout_run.make_ollama_service_factory", fake_factory
    )

    class Harness:
        def __init__(self, dataset: Any, config: Any) -> None:
            captured["model_sha256"] = config.model_sha256

        def run(self, adapter: Any, **kwargs: Any) -> dict[str, Any]:
            captured["adapter_model"] = adapter.model
            return {"status": "completed", "raw_outputs": []}

    monkeypatch.setattr("benchmarks.local_memory.heldout_run.ExperimentHarness", Harness)
    run_heldout(
        dataset_path=DATASET,
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        endpoint="http://unused.invalid",
        model="qwen3.5:9b",
        embedding_model="nomic-embed-text:latest",
        root=tmp_path,
        profile_names={"baseline"},
        extraction_mode="judged",
        judge_provider="typesafe",
        typesafe_api_key="k",
        surprisal_mode="novelty",
    )
    assert captured["judge_provider"] == "typesafe" and captured["surprisal_mode"] == "novelty"
    assert captured["model_sha256"] == "b" * 64
    assert captured["adapter_model"] == "jev-1.13.0"
    manifest = json.loads((tmp_path / "work/baseline.manifest.json").read_text())
    assert manifest["provider"]["judge"]["identity_kind"] == "vendor_version_string"
    assert manifest["provider"]["options"]["judge_provider"] == "typesafe"
    assert manifest["provider"]["options"]["typesafe_price_usd_per_million_input_tokens"] == 0.042

    with pytest.raises(ValueError, match="need a judge_provider"):
        run_heldout(
            dataset_path=DATASET,
            output_dir=tmp_path / "out2",
            work_dir=tmp_path / "work2",
            endpoint="http://unused.invalid",
            model="m",
            embedding_model="e",
            root=tmp_path,
            profile_names={"baseline"},
            extraction_mode="judged",
        )
