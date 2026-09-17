"""Measure the bounded local-memory latency gates on isolated temporary state.

The ten-thousand-row load uses vectors derived from one observed Ollama
embedding.  Those vectors exercise SQLite storage and ranking at scale; they
are explicitly a synthetic load fixture and do not measure semantic quality.
Query embeddings are still requested from the configured Ollama endpoint for
the end-to-end retrieval timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from mnemos.config import MnemosConfig, ModuleConfig, SurprisalConfig
from mnemos.engine import MnemosEngine
from mnemos.scopes import ScopeContext, ScopeRegistry
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import OllamaEmbeddingProvider, SimpleEmbeddingProvider
from mnemos.utils.storage import SQLiteStore
from mnemos.scopes import scope_allows

DEFAULT_ENDPOINT = "http://192.168.86.35:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
DEFAULT_COGNITION_MODEL = "gemma3:4b"
RECALL_GATE_MS = 500.0
CAPTURE_GATE_MS = 100.0
QUERY_COUNT = 32
CAPTURE_COUNT = 50
SOURCE_FILES = (
    "mnemos/engine.py",
    "mnemos/utils/storage.py",
    "mnemos/client.py",
    "mnemos/service.py",
    "mnemos/memory_service.py",
    "mnemos/utils/embeddings.py",
    "benchmarks/local_memory/performance.py",
)

QUERY_TEXTS = (
    "Observed seed vector for synthetic load construction.",
    "Which local database migration convention is active?",
    "Where does the repository keep its deployment label?",
    "What timeout should the local service use for retrieval?",
    "Which project scope contains the current memory?",
    "How should an acknowledged event be replayed safely?",
    "What is the local policy for correcting a stored fact?",
    "Which embedding model is configured for this run?",
    "How does the service separate user profiles?",
    "What happens when the local model is unavailable?",
    "Which files are retained by the benchmark fixture?",
    "What is the expected retrieval context budget?",
    "How does SQLite rank scoped vector candidates?",
    "Which host event records an assistant response?",
    "What is the project identity for this workspace?",
    "How does a durable capture differ from a queued outbox event?",
    "Which graph edges are allowed across scopes?",
    "What does the maintenance scheduler process after capture?",
    "How is an old fact marked forgotten?",
    "Which provider endpoint supplies the query vector?",
    "What prevents a foreign project from entering retrieval?",
    "Where is the service runtime descriptor stored?",
    "What is the configured top k retrieval limit?",
    "How are duplicate source events identified?",
    "Which metadata identifies a project fact?",
    "What does a restart preserve in SQLite?",
    "Which operation creates a correction job?",
    "How does the benchmark report cold startup?",
    "What evidence links a fact to its source event?",
    "Which scope is available to a project query?",
    "How does the baseline profile rank facts?",
    "What is the local capture failure behavior?",
    "Which rows are included in the ten-thousand-fact load?",
)


class TimedEmbeddingProvider:
    """Record provider time while retaining the last vector for SQL timing."""

    def __init__(self, inner: OllamaEmbeddingProvider) -> None:
        self.inner = inner
        self.last_ms = 0.0
        self.last_vector: list[float] = []

    @property
    def dim(self) -> int:
        return self.inner.dim

    @property
    def should_offload_in_async(self) -> bool:
        return True

    def embed(self, text: str) -> list[float]:
        started = time.perf_counter()
        vector = self.inner.embed(text)
        self.last_ms = (time.perf_counter() - started) * 1000
        self.last_vector = vector
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        vectors = self.inner.embed_batch(texts)
        self.last_ms = (time.perf_counter() - started) * 1000
        self.last_vector = vectors[-1] if vectors else []
        return vectors


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min_ms": round(min(values), 3) if values else 0.0,
        "mean_ms": round(statistics.mean(values), 3) if values else 0.0,
        "p50_ms": round(_percentile(values, 0.50), 3),
        "p95_ms": round(_percentile(values, 0.95), 3),
        "max_ms": round(max(values), 3) if values else 0.0,
    }


def _file_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm"))
        if candidate.exists()
    )


def _source_provenance(root: Path) -> dict[str, str | None]:
    """Hash the implementation files that materially affect this run."""
    result: dict[str, str | None] = {}
    for relative in SOURCE_FILES:
        path = root / relative
        try:
            result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            result[relative] = None
    return result


def _sqlite_provenance(store: SQLiteStore) -> dict[str, Any]:
    """Capture SQLite settings that influence a latency result."""
    with store._lock:
        synchronous = store._conn.execute("PRAGMA synchronous").fetchone()[0]
        journal_mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    return {
        "synchronous": str(synchronous),
        "synchronous_label": {0: "OFF", 1: "NORMAL", 2: "FULL", 3: "EXTRA"}.get(
            int(synchronous), "unknown"
        ),
        "journal_mode": str(journal_mode),
    }


def _vector_from_observed(base: list[float], index: int) -> list[float]:
    """Derive a distinct synthetic load vector from the observed seed vector."""
    digest = hashlib.sha256(f"mnemos-performance-load:{index}".encode()).digest()
    values = []
    for offset, value in enumerate(base):
        byte = digest[offset % len(digest)]
        perturbation = (byte - 127.5) / 127.5
        values.append(value + 0.015 * perturbation)
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else list(base)


def _ollama_provenance(endpoint: str, embedding_model: str, cognition_model: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "endpoint": endpoint,
        "embedding_model": embedding_model,
        "cognition_model": cognition_model,
        "embedding_digest": None,
        "cognition_digest": None,
        "tags_observed": False,
    }
    try:
        response = httpx.get(f"{endpoint.rstrip('/')}/api/tags", timeout=10.0)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", []) if isinstance(payload, dict) else []
        if isinstance(models, list):
            result["tags_observed"] = True
            for item in models:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("model")
                digest = item.get("digest")
                if not isinstance(name, str) or not isinstance(digest, str):
                    continue
                if name == embedding_model:
                    result["embedding_digest"] = digest
                if name == cognition_model:
                    result["cognition_digest"] = digest
    except (OSError, ValueError, httpx.HTTPError):
        result["tags_error"] = "unavailable"
    return result


def _make_engine(
    db_path: Path, registry: ScopeRegistry, embedder: Any
) -> tuple[MnemosEngine, SQLiteStore]:
    store = SQLiteStore(str(db_path))
    config = MnemosConfig(
        modules=ModuleConfig(
            surprisal=False,
            mutable_rag=False,
            affective=False,
            sleep=False,
            spreading=False,
        ),
        surprisal=SurprisalConfig(embedding_dim=768),
    )
    engine = MnemosEngine(
        config=config,
        embedder=embedder,
        store=store,
        scope_registry=registry,
        require_scope_context=True,
    )
    return engine, store


def _load_facts(store: SQLiteStore, base_vector: list[float], project_id: str) -> None:
    now = datetime.now(timezone.utc)
    insert_sql = """
        INSERT INTO memory_chunks
        (id, content, embedding, metadata, salience, cognitive_state,
         created_at, updated_at, access_count, version, canonical_key,
         user_profile_id, surprisal_score, surprisal_status, cognitive_state_status,
         last_observed_at, observation_count, last_retrieved_at, reconsolidation_due_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    rows: list[tuple[Any, ...]] = []
    for index in range(10_000):
        scope = "project" if index < 8_000 else "workspace" if index < 9_000 else "global"
        scope_id = (
            project_id if scope == "project" else "workspace-a" if scope == "workspace" else None
        )
        chunk = MemoryChunk(
            id=f"performance-fact-{index:05d}",
            content=f"Synthetic benchmark fact {index:05d} in {scope} scope.",
            embedding=_vector_from_observed(base_vector, index),
            metadata={
                "scope": scope,
                "scope_id": scope_id,
                "user_profile_id": "performance-profile",
                "benchmark_fixture": "synthetic_vector_load",
                "embedding_provenance": "derived_from_one_observed_embedding",
            },
            created_at=now,
            updated_at=now,
        )
        rows.append(store._chunk_to_row(chunk))
        if len(rows) == 500:
            store._conn.executemany(insert_sql, rows)
            store._conn.commit()
            rows.clear()
    if rows:
        store._conn.executemany(insert_sql, rows)
    store._conn.commit()
    # Build sqlite-vec once after the committed fixture load.  Per-row index
    # maintenance would measure fixture construction overhead rather than the
    # retrieval path under test.
    store.rebuild_indexes(rebuild_fts=False, rebuild_vec=True)


def measure_retrieval(
    root: Path, *, endpoint: str, embedding_model: str, cognition_model: str
) -> dict[str, Any]:
    repo = root / "project"
    repo.mkdir()
    registry = ScopeRegistry()
    project = registry.register_project(repo, user_profile_id="performance-profile", enabled=True)
    workspace = registry.register_workspace("workspace-a", user_profile_id="performance-profile")
    registry.add_membership(project.id, workspace.id)
    context = ScopeContext(
        user_profile_id="performance-profile",
        project_id=project.id,
        workspace_id=workspace.id,
        session_id="performance-session",
        host="manual",
    )
    embedder = OllamaEmbeddingProvider(model=embedding_model, base_url=endpoint, timeout=60.0)
    observed_started = time.perf_counter()
    observed = embedder.embed("Observed seed vector for synthetic load construction.")
    observed_ms = (time.perf_counter() - observed_started) * 1000
    if len(observed) != 768:
        raise ValueError(f"expected 768-dimensional embedding, received {len(observed)}")

    db_path = root / "retrieval.sqlite"
    timed_embedder = TimedEmbeddingProvider(embedder)
    engine, store = _make_engine(db_path, registry, timed_embedder)
    insert_started = time.perf_counter()
    _load_facts(store, observed, project.id)
    insert_ms = (time.perf_counter() - insert_started) * 1000
    bytes_after_load = _file_bytes(db_path)

    def scoped_filter(chunk: MemoryChunk) -> bool:
        return scope_allows(chunk.metadata, context, ("project", "workspace", "global"))

    setattr(
        scoped_filter,
        "_mnemos_scope_filter",
        {
            "user_profile_id": context.user_profile_id,
            "project_id": context.project_id,
            "workspace_id": context.workspace_id,
            "allowed_scopes": ["project", "workspace", "global"],
        },
    )

    for query in QUERY_TEXTS[:3]:
        asyncio_run(
            engine.retrieve(
                query,
                top_k=5,
                reconsolidate=False,
                context=context,
                allowed_scopes=["project", "workspace", "global"],
                profile="baseline",
            )
        )
    latencies: list[float] = []
    result_counts: list[int] = []
    raw: list[dict[str, Any]] = []
    lexical_latencies: list[float] = []
    mark_and_refresh_latencies: list[float] = []
    for index, query in enumerate(QUERY_TEXTS):
        started = time.perf_counter()
        results = asyncio_run(
            engine.retrieve(
                query,
                top_k=5,
                reconsolidate=False,
                context=context,
                allowed_scopes=["project", "workspace", "global"],
                profile="baseline",
            )
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        sql_started = time.perf_counter()
        sql_results = store.retrieve(
            timed_embedder.last_vector,
            top_k=120,
            filter_fn=scoped_filter,
        )
        sql_ms = (time.perf_counter() - sql_started) * 1000
        lexical_started = time.perf_counter()
        lexical_results = store.retrieve_lexical(
            query,
            top_k=120,
            filter_fn=scoped_filter,
        )
        lexical_ms = (time.perf_counter() - lexical_started) * 1000
        mark_started = time.perf_counter()
        selected_ids = [chunk.id for chunk in results[:5]]
        store.mark_retrieved_many(selected_ids, retrieved_at=datetime.now(timezone.utc))
        for chunk_id in selected_ids:
            store.get(chunk_id)
        mark_ms = (time.perf_counter() - mark_started) * 1000
        lexical_latencies.append(lexical_ms)
        mark_and_refresh_latencies.append(mark_ms)
        latencies.append(elapsed_ms)
        result_counts.append(len(results))
        raw.append(
            {
                "index": index,
                "latency_ms": round(elapsed_ms, 3),
                "embedding_ms": round(timed_embedder.last_ms, 3),
                "scoped_sql_ranking_ms": round(sql_ms, 3),
                "sql_candidate_count": len(sql_results),
                "lexical_candidate_ms": round(lexical_ms, 3),
                "lexical_candidate_count": len(lexical_results),
                "mark_and_refresh_ms": round(mark_ms, 3),
                "result_count": len(results),
            }
        )
    stats = store.get_stats()
    sqlite_settings = _sqlite_provenance(store)
    store.close()
    return {
        "status": "completed",
        "facts": 10_000,
        "rows_indexed": stats.get("total_chunks"),
        "embedding_dimension": len(observed),
        "load_fixture": "synthetic vectors derived from one observed Ollama embedding; semantic quality NOT RUN",
        "observed_seed_embedding_ms": round(observed_ms, 3),
        "load_insert_ms": round(insert_ms, 3),
        "bytes_after_load": bytes_after_load,
        "query_count": len(latencies),
        "warm_recall": _summary(latencies),
        "recall_gate_ms": RECALL_GATE_MS,
        "recall_gate": "PASS" if _percentile(latencies, 0.95) <= RECALL_GATE_MS else "FAIL",
        "result_count_mean": round(statistics.mean(result_counts), 3),
        "embedding_latency": _summary([item["embedding_ms"] for item in raw]),
        "scoped_sql_ranking_latency": _summary([item["scoped_sql_ranking_ms"] for item in raw]),
        "lexical_candidate_latency": _summary(lexical_latencies),
        "mark_and_refresh_latency": _summary(mark_and_refresh_latencies),
        "sqlite": sqlite_settings,
        "raw_queries": raw,
    }


def measure_capture(root: Path) -> dict[str, Any]:
    """Measure durable HTTP event acceptance and one full maintenance drain."""
    from mnemos.client import MnemosClient
    from mnemos.memory_service import LocalMemoryService
    from mnemos.service import MnemosService

    repo = root / "capture-project"
    repo.mkdir()
    registry = ScopeRegistry()
    project = registry.register_project(repo, user_profile_id="performance-profile", enabled=True)
    workspace = registry.register_workspace("workspace-a", user_profile_id="performance-profile")
    registry.add_membership(project.id, workspace.id)
    context = ScopeContext(
        user_profile_id="performance-profile",
        project_id=project.id,
        workspace_id=workspace.id,
        session_id="capture-session",
        host="manual",
    )
    db_path = root / "capture.sqlite"
    engine, store = _make_engine(db_path, registry, SimpleEmbeddingProvider(dim=768))
    dispatcher = LocalMemoryService(engine, registry)
    descriptor = root / "runtime.json"
    registry_path = root / "scopes.json"
    registry.save(registry_path)
    service = MnemosService(
        dispatcher=dispatcher,
        descriptor_path=descriptor,
        lock_path=root / "runtime.lock",
        startup_timeout=10.0,
        dispatch_timeout=30.0,
    )
    service.start()
    client = MnemosClient(descriptor_path=descriptor, auto_start=False, request_timeout=30.0)
    import asyncio

    capture_loop = asyncio.new_event_loop()
    latencies: list[float] = []
    statuses: list[str] = []
    try:
        warmup_payload = {
            "event_id": "performance-capture-warmup",
            "context": context.model_dump(mode="json"),
            "scope": "project",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "role": "user",
            "content": "Warmup durable benchmark event.",
            "source_kind": "performance_benchmark",
            "source_ref": "capture-warmup",
        }
        cold_client_started = time.perf_counter()
        capture_loop.run_until_complete(client.capture(warmup_payload))
        cold_client_ms = (time.perf_counter() - cold_client_started) * 1000
        for index in range(CAPTURE_COUNT):
            payload = {
                "event_id": f"performance-capture-{index:03d}",
                "context": context.model_dump(mode="json"),
                "scope": "project",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "role": "user",
                "content": f"Durable benchmark event {index:03d}.",
                "source_kind": "performance_benchmark",
                "source_ref": f"capture-{index:03d}",
            }
            started = time.perf_counter()
            result = capture_loop.run_until_complete(client.capture(payload))
            elapsed_ms = (time.perf_counter() - started) * 1000
            latencies.append(elapsed_ms)
            statuses.append(str(result.get("status", "unknown")))
        sync_started = time.perf_counter()
        sync_result = capture_loop.run_until_complete(
            client.request("maintenance", {}, queue_on_failure=False)
        )
        full_sync_ms = (time.perf_counter() - sync_started) * 1000
        hook_payload = {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(repo),
            "session_id": "cold-hook-session",
            "turn_id": "cold-hook-turn",
            "model": "benchmark",
            "permission_mode": "default",
            "transcript_path": None,
            "prompt": "Cold synthetic Codex hook measurement.",
        }
        hook_environment = os.environ.copy()
        hook_environment.update(
            {
                "MNEMOS_RUNTIME_DESCRIPTOR": str(descriptor),
                "MNEMOS_SCOPE_REGISTRY": str(registry_path),
                "MNEMOS_USER_PROFILE_ID": "performance-profile",
                "PYTHONPATH": str(Path(__file__).resolve().parents[2])
                + os.pathsep
                + hook_environment.get("PYTHONPATH", ""),
            }
        )
        cold_started = time.perf_counter()
        hook_process = subprocess.run(
            [sys.executable, "-m", "mnemos.adapters.codex"],
            cwd=str(repo),
            input=json.dumps(hook_payload),
            text=True,
            capture_output=True,
            timeout=30,
            env=hook_environment,
            check=False,
        )
        cold_hook_ms = (time.perf_counter() - cold_started) * 1000
    finally:
        capture_loop.run_until_complete(client.close())
        capture_loop.close()
        service.stop()
        store.close()
    return {
        "status": "completed",
        "event_count": CAPTURE_COUNT,
        "ack_statuses": statuses,
        "warm_http_capture": _summary(latencies),
        "http_client_warmup_ms": round(cold_client_ms, 3),
        "capture_gate_ms": CAPTURE_GATE_MS,
        "capture_gate": "PASS" if _percentile(latencies, 0.95) <= CAPTURE_GATE_MS else "FAIL",
        "full_sync_ms": round(full_sync_ms, 3),
        "full_sync_response": {
            "status": sync_result.get("status"),
            "completed": sync_result.get("completed"),
        },
        "bytes_after_sync": _file_bytes(db_path),
        "cold_cli_hook": {
            "status": "completed" if hook_process.returncode == 0 else "failed",
            "latency_ms": round(cold_hook_ms, 3),
            "returncode": hook_process.returncode,
            "host": "synthetic Codex adapter subprocess; real Codex session NOT RUN",
        },
    }


def asyncio_run(awaitable: Any) -> Any:
    """Run one client coroutine from this synchronous benchmark process."""
    import asyncio

    return asyncio.run(awaitable)


def run(
    *, output: Path, endpoint: str, embedding_model: str, cognition_model: str
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    source_root = Path(__file__).resolve().parents[2]
    source_before = _source_provenance(source_root)
    provenance = _ollama_provenance(endpoint, embedding_model, cognition_model)
    with tempfile.TemporaryDirectory(prefix="mnemos-performance-") as temporary:
        root = Path(temporary)
        retrieval = measure_retrieval(
            root,
            endpoint=endpoint,
            embedding_model=embedding_model,
            cognition_model=cognition_model,
        )
        capture = measure_capture(root)
        cold_hook = capture.pop("cold_cli_hook")
    source_after = _source_provenance(source_root)
    artifact = {
        "schema_version": "local-memory-performance-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "code_provenance": {
            "before": source_before,
            "after": source_after,
            "consistent": source_before == source_after,
        },
        "function_trace": [
            "OllamaEmbeddingProvider.embed",
            "MnemosEngine.retrieve",
            "SQLiteStore.retrieve",
            "SQLiteStore.retrieve_lexical",
            "SQLiteStore.mark_retrieved_many",
            "SQLiteStore.get",
            "MnemosClient.capture",
            "MnemosClient.request(maintenance)",
            "mnemos.adapters.codex subprocess",
        ],
        "configuration": {
            "retrieval_profile": "baseline",
            "modules": {
                "surprisal": False,
                "mutable_rag": False,
                "affective": False,
                "sleep": False,
                "spreading": False,
            },
            "retrieval_top_k": 5,
            "scoped_sql_candidate_top_k": 120,
            "retrieval_warmup_queries": 3,
            "capture_events": CAPTURE_COUNT,
            "capture_warmup_events": 1,
            "capture_provider": "simple(dim=768)",
            "cognition_calls": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "provider": provenance,
        "measurement_scope": {
            "database": "isolated temporary SQLite database",
            "semantic_quality": "NOT RUN for synthetic 10,000-row vector load",
            "cognition": "not used; no extraction model calls",
        },
        "retrieval": retrieval,
        "capture": capture,
        "cold_cli_hook": cold_hook,
        "total_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure isolated local-memory performance gates")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--cognition-model", default=DEFAULT_COGNITION_MODEL)
    args = parser.parse_args(argv)
    try:
        artifact = run(
            output=args.output,
            endpoint=args.endpoint,
            embedding_model=args.embedding_model,
            cognition_model=args.cognition_model,
        )
    except Exception as exc:
        artifact = {
            "schema_version": "local-memory-performance-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "not_run",
            "reason": f"Measurement could not complete: {type(exc).__name__}",
            "details": "No input content or credentials were written to the failure artifact.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(artifact, sort_keys=True))
        return 2
    print(json.dumps({"output": str(args.output), "status": artifact["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
