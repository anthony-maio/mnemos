"""
mnemos/benchmark.py — Retrieval benchmark harness.

Measures retrieval quality and latency with metrics:
- Recall@k
- MRR (Mean Reciprocal Rank)
- p95 latency (milliseconds)
"""

from __future__ import annotations

import asyncio
import argparse
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import MnemosConfig, MutableRAGConfig, SurprisalConfig
from .engine import MnemosEngine
from .runtime import build_embedder_from_env
from .types import Interaction, MemoryChunk, RetrievalFeedbackEvent
from .utils.llm import MockLLMProvider
from .utils.storage import InMemoryStore, MemoryStore, SQLiteStore

BENCHMARK_DATASET_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "datasets"
DATASET_PACKS: dict[str, tuple[str, ...]] = {
    "claim-driving": (
        "contradiction-update-v1.jsonl",
        "preference-drift-v1.jsonl",
        "cross-project-scope-v1.jsonl",
    )
}
DEFAULT_MAX_P95_LATENCY_RATIO_BY_STORE: dict[str, float] = {
    "memory": 2.0,
    "sqlite": 4.0,
}
DEFAULT_LATENCY_RATIO_FLOOR_MS_BY_STORE: dict[str, float] = {
    "memory": 1.0,
    "sqlite": 2.0,
}


@dataclass(frozen=True)
class BenchmarkDocument:
    """A memory document with one or more benchmark query variants."""

    id: str
    content: str
    queries: tuple[str, ...]
    scope: str = "project"
    scope_id: str | None = "default"


@dataclass(frozen=True)
class BenchmarkQuery:
    """A query with a set of relevant memory IDs."""

    text: str
    relevant_ids: set[str]
    current_scope: str = "project"
    scope_id: str | None = "default"
    allowed_scopes: tuple[str, ...] = ("project", "workspace", "global")


def _allowed_scopes_for_feedback_scope(scope: str) -> tuple[str, ...]:
    normalized = _normalize_scope(scope)
    if normalized == "project":
        return ("project", "workspace", "global")
    if normalized == "workspace":
        return ("workspace", "global")
    return ("global",)


def feedback_events_to_eval_rows(
    events: list[RetrievalFeedbackEvent],
) -> list[dict[str, Any]]:
    """
    Convert retrieval feedback events into stable JSONL-friendly eval rows.

    This intentionally captures only explicit human judgment for now. Helpful
    recalls preserve known-good chunk IDs as relevant targets. Negative events
    keep the retrieved chunk IDs for later audit, but leave relevant targets
    empty until we have a stronger labeling path.
    """
    rows: list[dict[str, Any]] = []
    for event in events:
        relevant_chunk_ids = list(event.chunk_ids) if event.event_type == "helpful" else []
        rows.append(
            {
                "id": event.id,
                "feedback_event_type": event.event_type,
                "query": event.query,
                "current_scope": event.scope,
                "scope_id": _normalize_scope_id(event.scope, event.scope_id),
                "allowed_scopes": list(_allowed_scopes_for_feedback_scope(event.scope)),
                "relevant_chunk_ids": relevant_chunk_ids,
                "retrieved_chunk_ids": list(event.chunk_ids),
                "notes": event.notes,
                "created_at": event.created_at.isoformat(),
            }
        )
    return rows


def write_feedback_eval_dataset(
    events: list[RetrievalFeedbackEvent],
    output_path: Path,
) -> int:
    """Write eval rows derived from retrieval feedback events as JSONL."""
    rows = feedback_events_to_eval_rows(events)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row) for row in rows)
    if text:
        text += "\n"
    output_path.write_text(text, encoding="utf-8")
    return len(rows)


def _normalize_scope(scope: str | None, *, default: str = "project") -> str:
    normalized = (scope or default).strip().lower()
    if normalized not in {"project", "workspace", "global"}:
        raise ValueError(f"Invalid scope {scope!r}; expected one of project, workspace, global.")
    return normalized


def _normalize_scope_id(scope: str, scope_id: str | None) -> str | None:
    if scope == "global":
        return None
    if scope_id is None:
        return "default"
    trimmed = scope_id.strip()
    return trimmed if trimmed else "default"


def _normalize_allowed_scopes(
    allowed_scopes: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if not allowed_scopes:
        return ("project", "workspace", "global")
    normalized: list[str] = []
    for scope in allowed_scopes:
        norm = _normalize_scope(scope)
        if norm not in normalized:
            normalized.append(norm)
    return tuple(normalized)


def _scope_filter_for_query(query: BenchmarkQuery) -> Callable[[MemoryChunk], bool]:
    def _matches(chunk: MemoryChunk) -> bool:
        chunk_scope = _normalize_scope(str(chunk.metadata.get("scope", "global")), default="global")
        if chunk_scope not in query.allowed_scopes:
            return False
        if chunk_scope == "global":
            return True
        chunk_scope_id_raw = chunk.metadata.get("scope_id")
        chunk_scope_id = (
            str(chunk_scope_id_raw).strip()
            if chunk_scope_id_raw is not None and str(chunk_scope_id_raw).strip()
            else "default"
        )
        if query.scope_id is None:
            return False
        return chunk_scope_id == query.scope_id

    return _matches


def default_benchmark_documents() -> list[BenchmarkDocument]:
    """Built-in dataset used when no dataset path is provided."""
    return [
        BenchmarkDocument(
            id="aws-ecs",
            content="We deploy microservices on AWS ECS with blue-green releases.",
            queries=("blue green ecs deployment", "microservices platform on aws ecs"),
        ),
        BenchmarkDocument(
            id="k8s-observability",
            content="Kubernetes clusters use Prometheus and Grafana for observability.",
            queries=("kubernetes monitoring stack", "prometheus grafana cluster metrics"),
        ),
        BenchmarkDocument(
            id="postgres-backup",
            content="PostgreSQL backups run nightly with point-in-time recovery enabled.",
            queries=("postgres backup schedule", "point in time recovery database"),
        ),
        BenchmarkDocument(
            id="redis-cache",
            content="Redis caches session tokens with a 15 minute TTL policy.",
            queries=("session token cache ttl", "redis policy for session cache"),
        ),
        BenchmarkDocument(
            id="pytest-ci",
            content="CI pipelines run pytest with coverage gates above ninety percent.",
            queries=("pytest coverage gate in ci", "test coverage threshold pipeline"),
        ),
        BenchmarkDocument(
            id="fastapi-auth",
            content="FastAPI services use OAuth2 bearer tokens for API authentication.",
            queries=("fastapi api auth method", "oauth2 bearer token service"),
        ),
        BenchmarkDocument(
            id="react-design",
            content="The frontend stack uses React with a shared design token system.",
            queries=("frontend framework and design tokens", "react shared design system"),
        ),
        BenchmarkDocument(
            id="svelte-migration",
            content="The web team is migrating from React to Svelte this quarter.",
            queries=("which framework migration is planned", "move from react to svelte"),
        ),
        BenchmarkDocument(
            id="incident-pagerduty",
            content="Critical incidents page the on-call engineer through PagerDuty.",
            queries=("how are critical incidents escalated", "on call paging tool"),
        ),
        BenchmarkDocument(
            id="runbook-nginx",
            content="The outage runbook restarts nginx then validates upstream health checks.",
            queries=("nginx outage runbook steps", "restart nginx and validate upstream"),
        ),
        BenchmarkDocument(
            id="terraform-modules",
            content="Infrastructure uses Terraform modules with environment-specific workspaces.",
            queries=("terraform workspace strategy", "infra modules per environment"),
        ),
        BenchmarkDocument(
            id="llm-routing",
            content="The assistant routes urgent bug triage to a high reasoning model.",
            queries=("model routing for bug triage", "urgent issue high reasoning model"),
        ),
    ]


def build_queries(documents: list[BenchmarkDocument]) -> list[BenchmarkQuery]:
    """Expand document query variants into query objects."""
    queries: list[BenchmarkQuery] = []
    for doc in documents:
        for query in doc.queries:
            queries.append(
                BenchmarkQuery(
                    text=query,
                    relevant_ids={doc.id},
                    current_scope=doc.scope,
                    scope_id=doc.scope_id,
                )
            )
    return queries


def _benchmark_store_id(doc_id: str) -> str:
    """
    Map arbitrary dataset IDs to stable UUIDs for backend compatibility.

    Keeps benchmark point IDs deterministic across runs and stores.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mnemos-benchmark:{doc_id}"))


def compute_retrieval_metrics(
    *,
    retrieved_ids: list[list[str]],
    relevant_ids: list[set[str]],
    latencies_ms: list[float],
    top_k: int,
) -> dict[str, float | int]:
    """Compute Recall@k, MRR, and p95 latency."""
    if len(retrieved_ids) != len(relevant_ids):
        raise ValueError("retrieved_ids and relevant_ids must have the same length.")

    query_count = len(retrieved_ids)
    if query_count == 0:
        return {
            "query_count": 0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "latency_p95_ms": 0.0,
            "latency_mean_ms": 0.0,
        }

    hit_count = 0
    reciprocal_rank_sum = 0.0

    for candidates, relevant in zip(retrieved_ids, relevant_ids):
        first_rank: int | None = None
        for index, candidate_id in enumerate(candidates[:top_k], start=1):
            if candidate_id in relevant:
                first_rank = index
                break
        if first_rank is not None:
            hit_count += 1
            reciprocal_rank_sum += 1.0 / first_rank

    p95_ms = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    mean_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    return {
        "query_count": query_count,
        "recall_at_k": hit_count / query_count,
        "mrr": reciprocal_rank_sum / query_count,
        "latency_p95_ms": p95_ms,
        "latency_mean_ms": mean_ms,
    }


def _coerce_document(item: dict[str, Any], index: int) -> BenchmarkDocument:
    content = item.get("content", item.get("text", ""))
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Dataset item at index {index} is missing non-empty 'content'.")

    raw_queries = item.get("queries", [])
    if raw_queries is None:
        raw_queries = []
    if not isinstance(raw_queries, list):
        raise ValueError(f"Dataset item at index {index} must include 'queries' as a list.")
    parsed_query_texts: list[str] = []
    parsed_scope: str | None = None
    parsed_scope_id: str | None = None
    for query_item in raw_queries:
        if isinstance(query_item, dict):
            text = str(query_item.get("text", "")).strip()
            if not text:
                continue
            parsed_query_texts.append(text)
            parsed_scope = str(query_item.get("current_scope", "project"))
            parsed_scope_id_raw = query_item.get("scope_id", "default")
            parsed_scope_id = str(parsed_scope_id_raw) if parsed_scope_id_raw is not None else None
        else:
            text = str(query_item).strip()
            if text:
                parsed_query_texts.append(text)
    queries = tuple(parsed_query_texts)

    scope = _normalize_scope(str(item.get("scope", parsed_scope or "project")))
    scope_id = _normalize_scope_id(
        scope, str(item.get("scope_id")) if item.get("scope_id") is not None else parsed_scope_id
    )

    doc_id_raw = item.get("id", f"doc-{index}")
    return BenchmarkDocument(
        id=str(doc_id_raw),
        content=content,
        queries=queries,
        scope=scope,
        scope_id=scope_id,
    )


def _load_queries(dataset_path: Path, documents: list[BenchmarkDocument]) -> list[BenchmarkQuery]:
    text = dataset_path.read_text(encoding="utf-8")
    suffix = dataset_path.suffix.lower()
    doc_by_id = {doc.id: doc for doc in documents}
    queries: list[BenchmarkQuery] = []

    def _append_doc_default_queries(doc: BenchmarkDocument) -> None:
        for q in doc.queries:
            queries.append(
                BenchmarkQuery(
                    text=q,
                    relevant_ids={doc.id},
                    current_scope=doc.scope,
                    scope_id=doc.scope_id,
                )
            )

    if suffix == ".jsonl":
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("id", ""))
            doc = doc_by_id.get(doc_id)
            if doc is None:
                continue
            raw_queries = item.get("queries", [])
            if not isinstance(raw_queries, list):
                _append_doc_default_queries(doc)
                continue
            if not raw_queries:
                continue
            for raw_query in raw_queries:
                if isinstance(raw_query, dict):
                    text_value = str(raw_query.get("text", "")).strip()
                    if not text_value:
                        continue
                    relevant_ids_raw = raw_query.get("relevant_ids", [doc.id])
                    if not isinstance(relevant_ids_raw, list):
                        raise ValueError(f"relevant_ids must be a list for document {doc.id}.")
                    relevant_ids = {str(value) for value in relevant_ids_raw if str(value).strip()}
                    if not relevant_ids:
                        relevant_ids = {doc.id}
                    current_scope = _normalize_scope(str(raw_query.get("current_scope", doc.scope)))
                    scope_id_value = raw_query.get("scope_id", doc.scope_id)
                    scope_id = str(scope_id_value) if scope_id_value is not None else None
                    scope_id = _normalize_scope_id(current_scope, scope_id)
                    allowed_scopes_raw = raw_query.get("allowed_scopes")
                    if allowed_scopes_raw is not None and not isinstance(allowed_scopes_raw, list):
                        raise ValueError(
                            f"allowed_scopes must be a list for document {doc.id} query {text_value!r}."
                        )
                    allowed_scopes = _normalize_allowed_scopes(
                        allowed_scopes_raw if isinstance(allowed_scopes_raw, list) else None
                    )
                    queries.append(
                        BenchmarkQuery(
                            text=text_value,
                            relevant_ids=relevant_ids,
                            current_scope=current_scope,
                            scope_id=scope_id,
                            allowed_scopes=allowed_scopes,
                        )
                    )
                else:
                    text_value = str(raw_query).strip()
                    if text_value:
                        queries.append(
                            BenchmarkQuery(
                                text=text_value,
                                relevant_ids={doc.id},
                                current_scope=doc.scope,
                                scope_id=doc.scope_id,
                            )
                        )
        return queries

    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("JSON dataset must be a list of objects.")
    for item in payload:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("id", ""))
        doc = doc_by_id.get(doc_id)
        if doc is None:
            continue
        raw_queries = item.get("queries", [])
        if not isinstance(raw_queries, list):
            _append_doc_default_queries(doc)
            continue
        for raw_query in raw_queries:
            if isinstance(raw_query, dict):
                text_value = str(raw_query.get("text", "")).strip()
                if not text_value:
                    continue
                relevant_ids_raw = raw_query.get("relevant_ids", [doc.id])
                if not isinstance(relevant_ids_raw, list):
                    raise ValueError(f"relevant_ids must be a list for document {doc.id}.")
                relevant_ids = {str(value) for value in relevant_ids_raw if str(value).strip()}
                if not relevant_ids:
                    relevant_ids = {doc.id}
                current_scope = _normalize_scope(str(raw_query.get("current_scope", doc.scope)))
                scope_id_value = raw_query.get("scope_id", doc.scope_id)
                scope_id = str(scope_id_value) if scope_id_value is not None else None
                scope_id = _normalize_scope_id(current_scope, scope_id)
                allowed_scopes_raw = raw_query.get("allowed_scopes")
                if allowed_scopes_raw is not None and not isinstance(allowed_scopes_raw, list):
                    raise ValueError(
                        f"allowed_scopes must be a list for document {doc.id} query {text_value!r}."
                    )
                allowed_scopes = _normalize_allowed_scopes(
                    allowed_scopes_raw if isinstance(allowed_scopes_raw, list) else None
                )
                queries.append(
                    BenchmarkQuery(
                        text=text_value,
                        relevant_ids=relevant_ids,
                        current_scope=current_scope,
                        scope_id=scope_id,
                        allowed_scopes=allowed_scopes,
                    )
                )
            else:
                text_value = str(raw_query).strip()
                if text_value:
                    queries.append(
                        BenchmarkQuery(
                            text=text_value,
                            relevant_ids={doc.id},
                            current_scope=doc.scope,
                            scope_id=doc.scope_id,
                        )
                    )
    return queries


def load_documents(dataset_path: Path) -> list[BenchmarkDocument]:
    """Load dataset from .json or .jsonl."""
    suffix = dataset_path.suffix.lower()
    if suffix not in {".json", ".jsonl"}:
        raise ValueError("Dataset file must be .json or .jsonl")

    documents: list[BenchmarkDocument] = []
    if suffix == ".jsonl":
        for idx, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, dict):
                raise ValueError(f"JSONL line {idx + 1} must be an object.")
            documents.append(_coerce_document(item, idx))
        return documents

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("JSON dataset must be a list of objects.")
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"JSON dataset item at index {idx} must be an object.")
        documents.append(_coerce_document(item, idx))
    return documents


def _cleanup_local_store_artifacts(store_type: str, sqlite_path: Path | None) -> None:
    if store_type == "sqlite" and sqlite_path is not None and sqlite_path.exists():
        sqlite_path.unlink()


def _sqlite_path_for_run(
    sqlite_path: Path | None,
    *,
    repetition: int,
    store_type: str,
    retriever: str,
) -> Path | None:
    if sqlite_path is None or store_type != "sqlite":
        return sqlite_path
    stem = sqlite_path.stem
    suffix = sqlite_path.suffix or ".sqlite"
    return sqlite_path.with_name(f"{stem}-r{repetition}-{store_type}-{retriever}{suffix}")


def _build_store(
    *,
    store_type: str,
    sqlite_path: Path | None,
) -> MemoryStore:
    if store_type == "memory":
        return InMemoryStore(name="benchmark-memory")
    if store_type == "sqlite":
        if sqlite_path is None:
            raise ValueError("sqlite_path is required for sqlite benchmark runs.")
        return SQLiteStore(db_path=str(sqlite_path), name="benchmark-sqlite")
    raise ValueError(f"Unsupported store type: {store_type!r}")


def _close_store(store: MemoryStore) -> None:
    close_fn = getattr(store, "close", None)
    if callable(close_fn):
        close_fn()


async def _run_engine_roundtrip(
    *,
    engine: MnemosEngine,
    documents: list[BenchmarkDocument],
    queries: list[BenchmarkQuery],
    top_k: int,
) -> tuple[dict[str, str], list[list[str]], list[set[str]], list[float]]:
    id_map: dict[str, str] = {}
    for doc in documents:
        result = await engine.process(
            Interaction(
                role="user",
                content=doc.content,
                metadata={"source": "benchmark", "benchmark_id": doc.id},
            ),
            scope=doc.scope,
            scope_id=doc.scope_id,
        )
        if not result.stored or result.chunk is None:
            raise RuntimeError(f"Engine failed to store benchmark document: {doc.id}")
        id_map[doc.id] = result.chunk.id

    retrieved_ids: list[list[str]] = []
    relevant_ids: list[set[str]] = []
    latencies_ms: list[float] = []

    for query in queries:
        start = time.perf_counter()
        chunks = await engine.retrieve(
            query.text,
            top_k=top_k,
            reconsolidate=False,
            current_scope=query.current_scope,
            scope_id=query.scope_id,
            allowed_scopes=query.allowed_scopes,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)
        retrieved_ids.append([chunk.id for chunk in chunks])
        mapped_relevant_ids = {id_map[doc_id] for doc_id in query.relevant_ids if doc_id in id_map}
        if not mapped_relevant_ids:
            raise RuntimeError(f"Query {query.text!r} has no mapped relevant IDs.")
        relevant_ids.append(mapped_relevant_ids)

    return id_map, retrieved_ids, relevant_ids, latencies_ms


def _build_comparisons(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_store: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for result in results:
        store_type = str(result.get("store_type", ""))
        dataset_name = str(result.get("dataset", "built-in"))
        retriever = str(result.get("retriever", ""))
        by_store.setdefault((dataset_name, store_type), {})[retriever] = result

    comparisons: list[dict[str, Any]] = []
    for (dataset_name, store_type), store_results in by_store.items():
        baseline = store_results.get("baseline")
        engine = store_results.get("engine")
        if baseline is None or engine is None:
            continue

        comparisons.append(
            {
                "dataset": dataset_name,
                "store_type": store_type,
                "baseline": {
                    "recall_at_k": baseline["recall_at_k"],
                    "mrr": baseline["mrr"],
                    "latency_p95_ms": baseline["latency_p95_ms"],
                },
                "engine": {
                    "recall_at_k": engine["recall_at_k"],
                    "mrr": engine["mrr"],
                    "latency_p95_ms": engine["latency_p95_ms"],
                },
                "delta_engine_minus_baseline": {
                    "recall_at_k": engine["recall_at_k"] - baseline["recall_at_k"],
                    "mrr": engine["mrr"] - baseline["mrr"],
                    "latency_p95_ms": engine["latency_p95_ms"] - baseline["latency_p95_ms"],
                },
            }
        )
    return comparisons


def _resolve_dataset_pack(pack_name: str) -> list[tuple[str, Path]]:
    files = DATASET_PACKS.get(pack_name)
    if files is None:
        available = ", ".join(sorted(DATASET_PACKS))
        raise ValueError(f"Unknown dataset pack {pack_name!r}. Available: {available}")

    resolved: list[tuple[str, Path]] = []
    for filename in files:
        path = BENCHMARK_DATASET_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Dataset pack file not found: {path}")
        resolved.append((path.stem, path))
    return resolved


def evaluate_production_replacement_gate(
    comparisons: list[dict[str, Any]],
    *,
    min_mrr_lift: float = 0.15,
    max_latency_ratio: float | None = None,
    latency_floor_ms: float = 1.0,
) -> dict[str, Any]:
    """Evaluate replacement claim gate across baseline vs engine comparisons."""
    details: list[dict[str, Any]] = []

    for comp in comparisons:
        store_type = str(comp["store_type"])
        baseline_mrr = float(comp["baseline"]["mrr"])
        engine_mrr = float(comp["engine"]["mrr"])
        baseline_latency = float(comp["baseline"]["latency_p95_ms"])
        engine_latency = float(comp["engine"]["latency_p95_ms"])

        if baseline_mrr > 0:
            mrr_lift_ratio = (engine_mrr - baseline_mrr) / baseline_mrr
        else:
            mrr_lift_ratio = 1.0 if engine_mrr > 0 else 0.0

        applied_latency_floor_ms = (
            latency_floor_ms
            if max_latency_ratio is not None
            else DEFAULT_LATENCY_RATIO_FLOOR_MS_BY_STORE.get(store_type, latency_floor_ms)
        )
        latency_denominator = max(baseline_latency, applied_latency_floor_ms)
        latency_ratio = engine_latency / latency_denominator if latency_denominator > 0 else 0.0
        applied_max_latency_ratio = (
            max_latency_ratio
            if max_latency_ratio is not None
            else DEFAULT_MAX_P95_LATENCY_RATIO_BY_STORE.get(store_type, 2.0)
        )

        passed = mrr_lift_ratio >= min_mrr_lift and latency_ratio <= applied_max_latency_ratio
        details.append(
            {
                "dataset": comp["dataset"],
                "store_type": store_type,
                "mrr_lift_ratio": mrr_lift_ratio,
                "latency_p95_ratio": latency_ratio,
                "max_latency_p95_ratio": applied_max_latency_ratio,
                "latency_ratio_floor_ms": applied_latency_floor_ms,
                "passed": passed,
            }
        )

    failed = [item for item in details if not item["passed"]]
    passed_count = len(details) - len(failed)

    return {
        "required_mrr_lift_ratio": min_mrr_lift,
        "max_latency_p95_ratio": max_latency_ratio,
        "max_latency_p95_ratio_by_store": (
            None if max_latency_ratio is not None else DEFAULT_MAX_P95_LATENCY_RATIO_BY_STORE
        ),
        "latency_ratio_floor_ms": latency_floor_ms,
        "latency_ratio_floor_ms_by_store": (
            None if max_latency_ratio is not None else DEFAULT_LATENCY_RATIO_FLOOR_MS_BY_STORE
        ),
        "evaluated_pairs": len(details),
        "passed_pairs": passed_count,
        "failed_pairs": len(failed),
        "passed": len(details) > 0 and len(failed) == 0,
        "details": details,
    }


def summarize_repeat_runs(reports: list[dict[str, Any]]) -> dict[str, Any]:
    per_store: dict[str, dict[str, list[float]]] = {}
    passed_runs = 0

    for report in reports:
        gate = report.get("gates", {}).get("production_replacement", {})
        if gate.get("passed") is True:
            passed_runs += 1
        for detail in gate.get("details", []):
            store_type = str(detail.get("store_type", "unknown"))
            metrics = per_store.setdefault(
                store_type,
                {"latency_p95_ratio": [], "mrr_lift_ratio": []},
            )
            metrics["latency_p95_ratio"].append(float(detail["latency_p95_ratio"]))
            metrics["mrr_lift_ratio"].append(float(detail["mrr_lift_ratio"]))

    stores: dict[str, Any] = {}
    for store_type, metrics in per_store.items():
        stores[store_type] = {
            "latency_p95_ratio": {
                "min": min(metrics["latency_p95_ratio"]),
                "max": max(metrics["latency_p95_ratio"]),
            },
            "mrr_lift_ratio": {
                "min": min(metrics["mrr_lift_ratio"]),
                "max": max(metrics["mrr_lift_ratio"]),
            },
        }

    total_runs = len(reports)
    return {
        "runs": total_runs,
        "passed_runs": passed_runs,
        "failed_runs": total_runs - passed_runs,
        "all_passed": total_runs > 0 and passed_runs == total_runs,
        "stores": stores,
    }


def run_retrieval_benchmark(
    *,
    store_type: str,
    retriever: str = "baseline",
    top_k: int,
    documents: list[BenchmarkDocument],
    query_limit: int | None = None,
    sqlite_path: Path | None = None,
    queries: list[BenchmarkQuery] | None = None,
    baseline_scope_aware: bool = False,
) -> dict[str, Any]:
    """Run one benchmark pass for one storage backend and retriever mode."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if not documents:
        raise ValueError("Benchmark requires at least one document.")
    if retriever not in {"baseline", "engine"}:
        raise ValueError("retriever must be 'baseline' or 'engine'.")

    benchmark_queries = queries or build_queries(documents)
    if query_limit is not None and query_limit > 0:
        benchmark_queries = benchmark_queries[:query_limit]

    _cleanup_local_store_artifacts(store_type, sqlite_path)
    embedder = build_embedder_from_env(default_provider="simple")

    store = _build_store(
        store_type=store_type,
        sqlite_path=sqlite_path,
    )

    ingest_start = time.perf_counter()
    retrieved_ids: list[list[str]] = []
    relevant_ids: list[set[str]] = []
    latencies_ms: list[float] = []

    if retriever == "baseline":
        document_embeddings = embedder.embed_batch([doc.content for doc in documents])
        if not document_embeddings:
            raise ValueError("Unable to generate document embeddings.")
        id_map: dict[str, str] = {}
        for doc, embedding in zip(documents, document_embeddings):
            store_id = _benchmark_store_id(doc.id)
            id_map[doc.id] = store_id
            metadata: dict[str, Any] = {"source": "benchmark", "scope": doc.scope}
            if doc.scope_id is not None:
                metadata["scope_id"] = doc.scope_id
            store.store(
                MemoryChunk(
                    id=store_id,
                    content=doc.content,
                    embedding=embedding,
                    metadata=metadata,
                )
            )
        ingest_seconds = time.perf_counter() - ingest_start

        for query in benchmark_queries:
            start = time.perf_counter()
            query_embedding = embedder.embed(query.text)
            scope_filter = _scope_filter_for_query(query) if baseline_scope_aware else None
            chunks = store.retrieve(query_embedding, top_k=top_k, filter_fn=scope_filter)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)
            retrieved_ids.append([chunk.id for chunk in chunks])
            mapped_relevant_ids = {
                id_map[doc_id] for doc_id in query.relevant_ids if doc_id in id_map
            }
            if not mapped_relevant_ids:
                raise RuntimeError(f"Query {query.text!r} has no mapped relevant IDs.")
            relevant_ids.append(mapped_relevant_ids)
    else:
        engine = MnemosEngine(
            config=MnemosConfig(
                surprisal=SurprisalConfig(threshold=0.0, min_content_length=0),
                mutable_rag=MutableRAGConfig(enabled=False),
            ),
            llm=MockLLMProvider(),
            embedder=embedder,
            store=store,
        )
        _, retrieved_ids, relevant_ids, latencies_ms = asyncio.run(
            _run_engine_roundtrip(
                engine=engine,
                documents=documents,
                queries=benchmark_queries,
                top_k=top_k,
            )
        )
        ingest_seconds = time.perf_counter() - ingest_start

    metrics = compute_retrieval_metrics(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        latencies_ms=latencies_ms,
        top_k=top_k,
    )

    result = {
        "store_type": store_type,
        "retriever": retriever,
        "top_k": top_k,
        "document_count": len(documents),
        "query_count": int(metrics["query_count"]),
        "ingest_seconds": ingest_seconds,
        "recall_at_k": metrics["recall_at_k"],
        "mrr": metrics["mrr"],
        "latency_mean_ms": metrics["latency_mean_ms"],
        "latency_p95_ms": metrics["latency_p95_ms"],
        "store_stats": store.get_stats(),
    }

    _close_store(store)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mnemos-benchmark",
        description="Benchmark Mnemos retrieval quality and latency.",
    )
    parser.add_argument(
        "--stores",
        default="memory,sqlite",
        help="Comma-separated store backends to benchmark (memory,sqlite).",
    )
    parser.add_argument(
        "--retrievers",
        default="baseline,engine",
        help="Comma-separated retriever modes to benchmark (baseline,engine).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="k for Recall@k/MRR (default: 5).")
    parser.add_argument(
        "--dataset", type=str, default="", help="Optional .json/.jsonl dataset path."
    )
    parser.add_argument(
        "--dataset-pack",
        type=str,
        default="",
        help="Named dataset pack (e.g., claim-driving). Cannot be used with --dataset.",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=0,
        help="Optional max number of query variants to benchmark.",
    )
    parser.add_argument(
        "--sqlite-path",
        type=str,
        default=".mnemos_benchmark.sqlite",
        help="SQLite path for sqlite benchmark runs.",
    )
    parser.add_argument(
        "--output", type=str, default="", help="Optional path to write JSON report."
    )
    parser.add_argument(
        "--enforce-production-gate",
        action="store_true",
        help="Exit non-zero when production replacement gate fails.",
    )
    parser.add_argument(
        "--gate-min-mrr-lift",
        type=float,
        default=0.15,
        help="Minimum required MRR lift ratio for production replacement gate.",
    )
    parser.add_argument(
        "--gate-max-p95-latency-ratio",
        type=float,
        default=None,
        help=(
            "Override maximum allowed engine/baseline p95 latency ratio for production gate. "
            "If omitted, uses store defaults (memory=2.0, sqlite=4.0)."
        ),
    )
    parser.add_argument(
        "--gate-latency-floor-ms",
        type=float,
        default=1.0,
        help="Minimum baseline latency denominator used when computing p95 ratio.",
    )
    parser.add_argument(
        "--baseline-scope-aware",
        action="store_true",
        help="Apply query scope filters to baseline retriever (disabled by default).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Repeat the full benchmark this many times and emit an aggregate summary.",
    )
    args = parser.parse_args()

    if args.dataset and args.dataset_pack:
        raise ValueError("Use either --dataset or --dataset-pack, not both.")

    store_types = [item.strip().lower() for item in args.stores.split(",") if item.strip()]
    if not store_types:
        raise ValueError("No store types provided.")
    retrievers = [item.strip().lower() for item in args.retrievers.split(",") if item.strip()]
    if not retrievers:
        raise ValueError("No retrievers provided.")

    query_limit = args.query_limit if args.query_limit > 0 else None
    sqlite_path = Path(args.sqlite_path) if args.sqlite_path else None

    if args.dataset_pack:
        dataset_runs = []
        for name, path in _resolve_dataset_pack(args.dataset_pack):
            docs = load_documents(path)
            queries = _load_queries(path, docs)
            dataset_runs.append((name, docs, queries))
        dataset_label = args.dataset_pack
    elif args.dataset:
        dataset_path = Path(args.dataset)
        docs = load_documents(dataset_path)
        queries = _load_queries(dataset_path, docs)
        dataset_runs = [(dataset_path.stem, docs, queries)]
        dataset_label = args.dataset
    else:
        docs = default_benchmark_documents()
        dataset_runs = [("built-in", docs, build_queries(docs))]
        dataset_label = "built-in"

    reports: list[dict[str, Any]] = []
    for repetition in range(args.repetitions):
        results: list[dict[str, Any]] = []
        for dataset_name, documents, benchmark_queries in dataset_runs:
            for store_type in store_types:
                for retriever in retrievers:
                    run_sqlite_path = _sqlite_path_for_run(
                        sqlite_path,
                        repetition=repetition + 1,
                        store_type=store_type,
                        retriever=retriever,
                    )
                    result = run_retrieval_benchmark(
                        store_type=store_type,
                        retriever=retriever,
                        top_k=args.top_k,
                        documents=documents,
                        query_limit=query_limit,
                        sqlite_path=run_sqlite_path,
                        queries=benchmark_queries,
                        baseline_scope_aware=args.baseline_scope_aware,
                    )
                    result["dataset"] = dataset_name
                    results.append(result)

        comparisons = _build_comparisons(results)
        gate = evaluate_production_replacement_gate(
            comparisons,
            min_mrr_lift=args.gate_min_mrr_lift,
            max_latency_ratio=args.gate_max_p95_latency_ratio,
            latency_floor_ms=args.gate_latency_floor_ms,
        )

        reports.append(
            {
                "repetition": repetition + 1,
                "dataset": dataset_label,
                "stores": store_types,
                "retrievers": retrievers,
                "top_k": args.top_k,
                "results": results,
                "comparisons": comparisons,
                "gates": {
                    "production_replacement": gate,
                },
            }
        )

    report = (
        reports[0]
        if len(reports) == 1
        else {
            "dataset": dataset_label,
            "stores": store_types,
            "retrievers": retrievers,
            "top_k": args.top_k,
            "repetitions": args.repetitions,
            "runs": reports,
            "summary": summarize_repeat_runs(reports),
        }
    )

    rendered = json.dumps(report, indent=2)
    print(rendered)

    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")

    final_gate_passed = (
        report["gates"]["production_replacement"]["passed"]
        if len(reports) == 1
        else report["summary"]["all_passed"]
    )
    if args.enforce_production_gate and not final_gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
