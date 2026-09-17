"""Prepare pinned real embeddings and run the separate mechanism experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

from mnemos.modules.spreading import SpreadingActivation
from mnemos.utils.embeddings import EmbeddingProvider, OllamaEmbeddingProvider
from mnemos.utils.llm import OllamaProvider
from mnemos.utils.storage import SQLiteStore

from .artifacts import environment_metadata
from .dev_smoke import _code_provenance
from .heldout_run import _live_model_metadata


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(name, path)


class CheckpointedEmbeddingProvider(EmbeddingProvider):
    """Bounded batches with a persistent, provenance-bound exact-text cache."""

    def __init__(
        self, provider: EmbeddingProvider, path: Path, binding: str, *, batch_size: int = 16
    ) -> None:
        self.provider = provider
        self.path = path
        self.binding = binding
        self.batch_size = batch_size
        self.vectors: dict[str, list[float]] = {}
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if path.exists():
            prior = json.loads(path.read_text(encoding="utf-8"))
            if prior.get("binding") != binding:
                raise ValueError("Embedding checkpoint provenance mismatch")
            self.vectors = prior["vectors"]
            self._validate(list(self.vectors.values()))

    @staticmethod
    def _validate(vectors: list[list[float]]) -> None:
        if any(
            not vector or not all(math.isfinite(float(x)) for x in vector) for vector in vectors
        ):
            raise ValueError("Embedding provider returned an empty or nonfinite vector")
        if len({len(vector) for vector in vectors}) > 1:
            raise ValueError("Embedding dimensions changed during preparation")

    @property
    def dim(self) -> int:
        return len(next(iter(self.vectors.values()))) if self.vectors else self.provider.dim

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(text for text in texts if text not in self.vectors))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = self.provider.embed_batch(batch)
            if len(vectors) != len(batch):
                raise ValueError("Embedding provider returned the wrong number of vectors")
            self._validate(vectors)
            if self.vectors and vectors and len(vectors[0]) != self.dim:
                raise ValueError("Embedding dimensions changed between batches")
            self.vectors.update(zip(batch, vectors))
            _write_json(self.path, {"binding": self.binding, "vectors": self.vectors})
        return [list(self.vectors[text]) for text in texts]


class RecordedOllamaProvider(OllamaProvider):
    """Retain exact synthetic classifier responses, including failed attempts."""

    def __init__(self, *, trace_path: Path, binding: str, **kwargs: Any) -> None:
        self.trace_path = trace_path
        self.binding = binding
        self.responses: list[dict[str, Any]] = []
        super().__init__(response_observer=self.responses.append, **kwargs)

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        self.responses.clear()
        record: dict[str, Any] = {"binding": self.binding, "prompt": prompt, "options": kwargs}
        try:
            result = await super().predict(prompt, **kwargs)
            record.update(status="complete", response=result)
            return result
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            record["raw_responses"] = list(self.responses)
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


class RecordedOllamaEmbeddings(OllamaEmbeddingProvider):
    def __init__(self, *, trace_path: Path, binding: str, **kwargs: Any) -> None:
        self.trace_path = trace_path
        self.binding = binding
        self.responses: list[dict[str, Any]] = []
        super().__init__(response_observer=self.responses.append, **kwargs)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.responses.clear()
        record: dict[str, Any] = {"binding": self.binding, "texts": texts}
        try:
            vectors = super().embed_batch(texts)
            record["status"] = "complete"
            return vectors
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            record["raw_responses"] = list(self.responses)
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def run_mechanisms(
    *,
    dataset_path: Path,
    protocol_path: Path,
    output_dir: Path,
    work_dir: Path,
    endpoint: str,
    model: str,
    embedding_model: str,
    split: str = "heldout",
    repetitions: int = 3,
    prepare_only: bool = False,
) -> dict[str, Any]:
    from .module_mechanism import (
        evaluate_dataset,
        load_mechanism_dataset,
        prepare_frozen_embeddings,
    )

    if split not in {"development", "heldout"} or repetitions < 3:
        raise ValueError("A declared split and at least three repetitions are required")
    # Validate all authored scope/label contracts before requesting embeddings.
    load_mechanism_dataset(dataset_path)
    source = json.loads(dataset_path.read_text(encoding="utf-8"))
    source["graph_cases"] = [case for case in source["graph_cases"] if case["split"] == split]
    source["affective_pairs"] = [
        pair for pair in source["affective_pairs"] if pair["split"] == split
    ]
    if not source["graph_cases"] or not source["affective_pairs"]:
        raise ValueError("Selected split has no graph cases or affective pairs")
    if split == "heldout" and min(len(source["graph_cases"]), len(source["affective_pairs"])) < 50:
        raise ValueError("Held-out mechanism protocol requires at least 50 cases of each kind")
    metadata = _live_model_metadata(endpoint, (model, embedding_model))
    code_sha256, code_files = _code_provenance(Path(__file__).resolve().parents[2])
    pins = {
        "schema_version": "module-mechanism-run-v1",
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "selected_cases_sha256": _digest(source),
        "split": split,
        "repetitions": repetitions,
        "execution_seed": 20260916,
        "code_sha256": code_sha256,
        "code_files": code_files,
        "python": platform.python_version(),
        "runtime": environment_metadata(),
        "storage_schema_version": SQLiteStore._SCHEMA_VERSION,
        "provider": {
            "endpoint": endpoint,
            "model": metadata[model],
            "embedding_model": metadata[embedding_model],
        },
        "graph_contract": SpreadingActivation.persistence_contract(),
        "options": {
            "temperature": 0.0,
            "max_tokens": 512,
            "timeout_seconds": 120,
            "top_k": 5,
            "embedding_batch_size": 16,
            "normalization": "provider vectors unchanged; production cosine normalizes during scoring",
        },
    }
    binding = _digest(pins)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("binding") != binding:
            raise ValueError("Refusing to resume mechanism run with changed provenance")
        prior_vectors = work_dir / "frozen-vectors.json"
        if prior.get("frozen_vectors_sha256") and (
            not prior_vectors.exists()
            or hashlib.sha256(prior_vectors.read_bytes()).hexdigest()
            != prior["frozen_vectors_sha256"]
        ):
            raise ValueError("Previously frozen vectors changed or disappeared")
    manifest: dict[str, Any] = {**pins, "binding": binding, "status": "preparing_embeddings"}
    _write_json(manifest_path, manifest)
    work_dir.mkdir(parents=True, exist_ok=True)
    source["manifest"] = {**pins, "binding": binding, "repetitions": repetitions}
    selected = work_dir / "authored-selected.json"
    if selected.exists() and json.loads(selected.read_text(encoding="utf-8")) != source:
        raise ValueError("Selected authored case file changed")
    if not selected.exists():
        _write_json(selected, source)
    prepared = work_dir / "frozen-vectors.json"
    cache = CheckpointedEmbeddingProvider(
        RecordedOllamaEmbeddings(
            trace_path=output_dir / "embedding.raw-outputs.jsonl",
            binding=binding,
            model=embedding_model,
            base_url=endpoint,
            timeout=120,
        ),
        work_dir / "embeddings.checkpoint.json",
        binding,
    )
    if not prepared.exists():
        temporary = work_dir / "vectors-in-progress.json"
        # An incomplete fixture is never accepted as a frozen input. Completed
        # embedding batches survive separately in the bound cache.
        if temporary.exists():
            raise ValueError(
                "An unfinished vector file exists; preserve it and use a fresh work directory"
            )
        prepare_frozen_embeddings(selected, temporary, cache)
        load_mechanism_dataset(temporary)
        os.replace(temporary, prepared)
    frozen = json.loads(prepared.read_text(encoding="utf-8"))
    if frozen.get("manifest", {}).get("binding") != binding:
        raise ValueError("Frozen vectors provenance mismatch")
    manifest.update(
        frozen_vectors_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest(),
        embedding_raw_outputs_sha256=(
            hashlib.sha256((output_dir / "embedding.raw-outputs.jsonl").read_bytes()).hexdigest()
            if (output_dir / "embedding.raw-outputs.jsonl").exists()
            else None
        ),
        status="prepared",
    )
    _write_json(manifest_path, manifest)
    if prepare_only:
        return manifest
    trace = output_dir / "classifier.raw-outputs.jsonl"
    provider = RecordedOllamaProvider(
        trace_path=trace,
        binding=binding,
        model=model,
        base_url=endpoint,
        timeout=120,
        temperature=0.0,
        max_tokens=512,
    )
    try:
        report = evaluate_dataset(
            prepared,
            output_dir=output_dir,
            work_dir=work_dir,
            repetitions=repetitions,
            llm=provider,
        )
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        _write_json(manifest_path, manifest)
        raise
    manifest.update(
        status=report["status"],
        execution=report.get("execution"),
        coverage=report.get("coverage"),
        report_sha256=hashlib.sha256(
            (output_dir / "module-mechanism.json").read_bytes()
        ).hexdigest(),
        raw_outputs_sha256=(
            hashlib.sha256(trace.read_bytes()).hexdigest() if trace.exists() else None
        ),
    )
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("datasets") / "module-mechanism-v1.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root / "docs/plans/2026-09-16-module-evaluation-protocol.md",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://192.168.86.35:11434")
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--split", choices=("development", "heldout"), default="heldout")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    result = run_mechanisms(
        dataset_path=args.dataset,
        protocol_path=args.protocol,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        endpoint=args.endpoint,
        model=args.model,
        embedding_model=args.embedding_model,
        split=args.split,
        repetitions=args.repetitions,
        prepare_only=args.prepare_only,
    )
    print(json.dumps({"status": result["status"], "binding": result["binding"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
