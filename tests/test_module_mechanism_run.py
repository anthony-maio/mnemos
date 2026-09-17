from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.local_memory.module_mechanism_run import (
    CheckpointedEmbeddingProvider,
    RecordedOllamaEmbeddings,
    RecordedOllamaProvider,
)
from mnemos.utils.embeddings import EmbeddingProvider, OllamaEmbeddingProvider
from mnemos.utils.llm import OllamaProvider


class CountingEmbedder(EmbeddingProvider):
    def __init__(self, *, fail_after=None, dimension=2):
        self.calls = []
        self.fail_after = fail_after
        self.dimension = dimension

    @property
    def dim(self):
        return self.dimension

    def embed(self, text):
        return self.embed_batch([text])[0]

    def embed_batch(self, texts):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise ConnectionError("simulated outage")
        self.calls.append(texts)
        return [[float(len(text)), *([1.0] * (self.dimension - 1))] for text in texts]


def test_embedding_preparation_resumes_without_reembedding_committed_batches(tmp_path):
    path = tmp_path / "cache.json"
    original = CountingEmbedder(fail_after=1)
    cache = CheckpointedEmbeddingProvider(original, path, "pinned", batch_size=2)
    with pytest.raises(ConnectionError):
        cache.embed_batch(["first", "second", "third", "first"])
    assert set(json.loads(path.read_text())["vectors"]) == {"first", "second"}

    resumed_provider = CountingEmbedder()
    resumed = CheckpointedEmbeddingProvider(resumed_provider, path, "pinned", batch_size=2)
    results = resumed.embed_batch(["first", "second", "third", "first"])
    assert resumed_provider.calls == [["third"]]
    assert results[0] == results[3] == [5.0, 1.0]
    assert resumed.dim == 2
    with pytest.raises(ValueError, match="provenance"):
        CheckpointedEmbeddingProvider(resumed_provider, path, "different-model")


def test_dimension_drift_does_not_replace_the_committed_embedding_cache(tmp_path):
    path = tmp_path / "cache.json"
    cache = CheckpointedEmbeddingProvider(CountingEmbedder(), path, "pinned")
    cache.embed("first")
    before = path.read_bytes()
    drift = CheckpointedEmbeddingProvider(CountingEmbedder(dimension=3), path, "pinned")
    with pytest.raises(ValueError, match="dimensions changed"):
        drift.embed("second")
    assert path.read_bytes() == before


def test_embedding_raw_trace_is_written_before_batch_is_acknowledged(tmp_path, monkeypatch):
    def fake_batch(self, texts):
        self.response_observer({"embeddings": [[1.0, 0.0]], "model": "fixture"})
        return [[1.0, 0.0]]

    monkeypatch.setattr(OllamaEmbeddingProvider, "embed_batch", fake_batch)
    trace = tmp_path / "trace.jsonl"
    provider = RecordedOllamaEmbeddings(trace_path=trace, binding="pinned", model="fixture")
    cache = CheckpointedEmbeddingProvider(provider, tmp_path / "cache.json", "pinned")
    assert cache.embed("source fact") == [1.0, 0.0]
    record = json.loads(trace.read_text())
    assert record["texts"] == ["source fact"]
    assert record["raw_responses"][0]["embeddings"] == [[1.0, 0.0]]
    assert record["binding"] == "pinned"


@pytest.mark.asyncio
async def test_classifier_trace_preserves_failed_raw_response(tmp_path, monkeypatch):
    async def failing_predict(self, prompt, **kwargs):
        self.response_observer({"response": "malformed state"})
        raise ValueError("parse failure")

    monkeypatch.setattr(OllamaProvider, "predict", failing_predict)
    trace = tmp_path / "classifier.jsonl"
    provider = RecordedOllamaProvider(trace_path=trace, binding="pinned", model="fixture")
    with pytest.raises(ValueError):
        await provider.predict("synthetic classifier prompt")
    record = json.loads(trace.read_text())
    assert record["status"] == "failed"
    assert record["raw_responses"] == [{"response": "malformed state"}]
    assert record["error_type"] == "ValueError"


def test_launcher_freezes_vectors_and_rejects_changed_provenance(tmp_path, monkeypatch):
    from benchmarks.local_memory import module_mechanism as core
    from benchmarks.local_memory import module_mechanism_run as runner

    authored = json.loads(
        Path(runner.__file__).with_name("datasets").joinpath("module-mechanism-v1.json").read_text()
    )
    for key in ("graph_cases", "affective_pairs"):
        authored[key] = [next(item for item in authored[key] if item["split"] == "development")]
    dataset = tmp_path / "authored.json"
    dataset.write_text(json.dumps(authored))
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Frozen fixture protocol")
    monkeypatch.setattr(
        runner,
        "_live_model_metadata",
        lambda endpoint, models: {
            model: {"digest": "fixture-digest", "name": model} for model in models
        },
    )
    monkeypatch.setattr(runner, "_code_provenance", lambda root: ("fixture-source", ["fixture.py"]))
    requests = []

    def fake_batch(self, texts):
        requests.append(list(texts))
        vectors = [[1.0, float(len(text) % 10) / 10] for text in texts]
        self.response_observer({"embeddings": vectors})
        return vectors

    monkeypatch.setattr(OllamaEmbeddingProvider, "embed_batch", fake_batch)

    def fake_evaluation(path, *, output_dir, **kwargs):
        assert core.load_mechanism_dataset(path).graph_cases
        report = {
            "status": "complete",
            "execution": {"complete": True},
            "coverage": {"failed": True},
        }
        (output_dir / "module-mechanism.json").write_text(json.dumps(report))
        return report

    monkeypatch.setattr(core, "evaluate_dataset", fake_evaluation)
    options = dict(
        dataset_path=dataset,
        protocol_path=protocol,
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        endpoint="http://unused.invalid",
        model="fixture",
        embedding_model="fixture-embed",
        split="development",
    )
    first = runner.run_mechanisms(**options)
    assert first["status"] == "complete"
    assert first["coverage"]["failed"] is True
    assert first["embedding_raw_outputs_sha256"]
    assert first["report_sha256"]
    request_count = len(requests)
    second = runner.run_mechanisms(**options)
    assert first["binding"] == second["binding"]
    assert len(requests) == request_count
    vector_path = tmp_path / "work" / "frozen-vectors.json"
    original = vector_path.read_bytes()
    vector_path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="changed or disappeared"):
        runner.run_mechanisms(**options)
    vector_path.write_bytes(original)
    protocol.write_text("Changed protocol")
    with pytest.raises(ValueError, match="changed provenance"):
        runner.run_mechanisms(**options)
    assert len(requests) == request_count
