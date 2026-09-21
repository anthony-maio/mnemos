"""Generation settings must reach providers and bind resumable evaluations."""

import json
from pathlib import Path

import pytest

from benchmarks.local_memory import heldout_run as runner

DATASET = (
    Path(__file__).parents[1] / "benchmarks/local_memory/datasets/local-memory-heldout-v3.jsonl"
)


def invoke(tmp_path, **kwargs):
    return runner.run_heldout(
        dataset_path=DATASET,
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        endpoint="http://unused.invalid",
        model="model",
        embedding_model="embedding",
        root=tmp_path,
        profile_names={"baseline"},
        **kwargs,
    )


@pytest.mark.parametrize(
    "setting,value",
    [
        ("temperature", -1.0),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("max_tokens", 0),
        ("max_tokens", True),
        ("max_tokens", 1.5),
        ("timeout_seconds", 0.0),
        ("timeout_seconds", float("inf")),
    ],
)
def test_invalid_settings_fail_before_provider_or_output(monkeypatch, tmp_path, setting, value):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid settings reached provider discovery")

    monkeypatch.setattr(runner, "_live_model_metadata", forbidden)
    with pytest.raises(ValueError):
        invoke(tmp_path, **{setting: value})
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "work").exists()


def test_settings_reach_factory_and_change_checkpoint_binding(monkeypatch, tmp_path):
    factories, configs = [], []
    monkeypatch.setattr(
        runner,
        "_live_model_metadata",
        lambda *args: {"model": {"digest": "a" * 64}, "embedding": {"digest": "b" * 64}},
    )
    monkeypatch.setattr(runner, "_code_provenance", lambda root: ("c" * 64, {}))
    monkeypatch.setattr(
        runner, "make_ollama_service_factory", lambda *args, **kwargs: factories.append(kwargs)
    )

    class Harness:
        def __init__(self, dataset, config):
            configs.append(config)

        def run(self, *args, **kwargs):
            return {"status": "complete", "raw_outputs": []}

    monkeypatch.setattr(runner, "ExperimentHarness", Harness)
    adapter = lambda *args, **kwargs: None
    variations = [{}, {"temperature": 1.0}, {"max_tokens": 8192}, {"timeout_seconds": 180.0}]
    for index, changes in enumerate(variations):
        root = tmp_path / str(index)
        invoke(root, adapter_type=adapter, **changes)
        manifest = json.loads((root / "work/baseline.manifest.json").read_text())
        options = manifest["provider"]["options"]
        assert options["temperature"] == changes.get("temperature", 0.0)
        assert options["max_tokens"] == changes.get("max_tokens", 512)
        assert options["timeout_seconds"] == changes.get("timeout_seconds", 120.0)
        assert factories[-1]["temperature"] == options["temperature"]
        assert factories[-1]["max_tokens"] == options["max_tokens"]
        assert factories[-1]["timeout"] == options["timeout_seconds"]
        assert configs[-1].options_sha256 == manifest["options_sha256"]
    assert len({c.options_sha256 for c in configs}) == 4


def test_cli_passes_explicit_generation_settings(monkeypatch, tmp_path):
    seen = {}

    def run(**kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(runner, "run_heldout", run)
    assert (
        runner.main(
            [
                "--output-dir",
                str(tmp_path / "out"),
                "--work-dir",
                str(tmp_path / "work"),
                "--temperature",
                "1.0",
                "--max-tokens",
                "8192",
                "--timeout-seconds",
                "180",
            ]
        )
        == 0
    )
    assert (seen["temperature"], seen["max_tokens"], seen["timeout_seconds"]) == (1.0, 8192, 180.0)
