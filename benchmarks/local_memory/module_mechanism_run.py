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
from mnemos.runtime import embedding_task_contract
from mnemos.utils.embeddings import (
    EmbeddingProvider,
    EmbeddingRole,
    OllamaEmbeddingProvider,
    PrefixEmbeddingProvider,
    embed_batch_for_role,
)
from mnemos.utils.llm import OllamaProvider
from mnemos.utils.storage import SQLiteStore

from .artifacts import environment_metadata
from .dev_smoke import _code_provenance
from .heldout_run import _live_model_metadata
from .ollama_adapter import OllamaThink, add_ollama_think, parse_ollama_think


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
        self,
        provider: EmbeddingProvider,
        path: Path,
        binding: str,
        *,
        batch_size: int = 16,
        task_contract: dict[str, str] | None = None,
    ) -> None:
        self.provider = provider
        self.path = path
        self.binding = binding
        self.batch_size = batch_size
        self.task_contract = task_contract
        self._role_vectors: dict[str, dict[str, list[float]]] = {
            "query": {},
            "document": {},
        }
        self.vectors: dict[str, list[float]] = {}
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if path.exists():
            prior = json.loads(path.read_text(encoding="utf-8"))
            if prior.get("binding") != binding:
                raise ValueError("Embedding checkpoint provenance mismatch")
            if prior.get("task_contract") != task_contract:
                raise ValueError("Embedding checkpoint task contract mismatch")
            self.vectors = prior["vectors"]
            self._validate(list(self.vectors.values()))
            role_vectors = prior.get("role_vectors", {})
            if role_vectors and not isinstance(role_vectors, dict):
                raise ValueError("Embedding checkpoint role_vectors must be an object")
            for role in self._role_vectors:
                values = role_vectors.get(role, {}) if isinstance(role_vectors, dict) else {}
                if not isinstance(values, dict):
                    raise ValueError("Embedding checkpoint role vectors must be objects")
                self._validate(list(values.values()))
                self._role_vectors[role] = {
                    str(text): list(map(float, vector)) for text, vector in values.items()
                }
            self._validate(self._all_vectors())

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
        values = self._all_vectors()
        return len(values[0]) if values else self.provider.dim

    def _all_vectors(self) -> list[list[float]]:
        return list(self.vectors.values()) + [
            vector for values in self._role_vectors.values() for vector in values.values()
        ]

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch_role(texts, "symmetric")

    def _embed_batch_role(self, texts: list[str], role: EmbeddingRole) -> list[list[float]]:
        cache = self.vectors if role == "symmetric" else self._role_vectors[role]
        missing = list(dict.fromkeys(text for text in texts if text not in cache))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = embed_batch_for_role(self.provider, batch, role=role)
            if len(vectors) != len(batch):
                raise ValueError("Embedding provider returned the wrong number of vectors")
            self._validate(vectors)
            if self._all_vectors() and vectors and len(vectors[0]) != self.dim:
                raise ValueError("Embedding dimensions changed between batches")
            cache.update(zip(batch, vectors))
            payload: dict[str, Any] = {
                "binding": self.binding,
                "vectors": self.vectors,
            }
            if self.task_contract is not None:
                payload["task_contract"] = self.task_contract
            if any(self._role_vectors.values()):
                payload["role_vectors"] = self._role_vectors
            _write_json(self.path, payload)
        return [list(cache[text]) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch_role([text], "query")[0]

    def embed_document(self, text: str) -> list[float]:
        return self._embed_batch_role([text], "document")[0]

    def embed_symmetric(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_batch_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch_role(texts, "query")

    def embed_batch_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch_role(texts, "document")

    def embed_batch_symmetric(self, texts: list[str]) -> list[list[float]]:
        return self.embed_batch(texts)


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
    think: OllamaThink = None,
    query_prefix: str = "",
    document_prefix: str = "",
    symmetric_prefix: str = "",
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
    options: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": 512,
        "timeout_seconds": 120,
        "top_k": 5,
        "embedding_batch_size": 16,
        "normalization": "provider vectors unchanged; production cosine normalizes during scoring",
    }
    task_contract = embedding_task_contract(
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        symmetric_prefix=symmetric_prefix,
    )
    if task_contract is not None:
        options["embedding_task_contract"] = task_contract
    add_ollama_think(options, think)
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
        "options": options,
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
        task_contract=task_contract,
    )
    if task_contract is not None:
        cache.provider = PrefixEmbeddingProvider(
            cache.provider,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            symmetric_prefix=symmetric_prefix,
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
        think=think,
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
        default=root / "docs/plans/2026-09-17-module-evaluation-protocol-v2.md",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://192.168.86.35:11434")
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--split", choices=("development", "heldout"), default="heldout")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--think", type=parse_ollama_think, default=None)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--document-prefix", default="")
    parser.add_argument("--symmetric-prefix", default="")
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
        think=args.think,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
        symmetric_prefix=args.symmetric_prefix,
    )
    print(json.dumps({"status": result["status"], "binding": result["binding"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
