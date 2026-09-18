"""Benchmark-only retrieval strategies over identical accepted SQLite snapshots."""

from __future__ import annotations

import hashlib
import random
import shutil
import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.retrieval import context_token_upper_bound, score_candidate
from mnemos.scopes import ScopeContext, scope_allows
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import embed_text_async
from mnemos.utils.storage import SQLiteStore

from .contracts import SessionScenario
from .harness import DEFAULT_PROFILES, ExperimentProfile, RunObservation
from .ollama_adapter import OllamaEvaluatorAdapter

STRATEGIES = ("vector", "lexical", "hybrid", "full")


class ComparisonEngine(MnemosEngine):
    """Keep experimental selection out of the production retrieval API."""

    strategy: str
    comparison_details: dict[str, Any]

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        reconsolidate: bool = True,
        *,
        current_scope: str = "project",
        scope_id: str | None = None,
        allowed_scopes: tuple[str, ...] | list[str] | None = None,
        context: ScopeContext | None = None,
        profile: str | None = None,
        max_context_tokens: int = 1200,
    ) -> list[MemoryChunk]:
        if self.strategy not in STRATEGIES:
            raise ValueError("Unknown retrieval strategy")
        self.comparison_details = {"strategy": self.strategy}
        if self.strategy in {"full", "hybrid"}:
            return await super().retrieve(
                query,
                top_k=top_k,
                reconsolidate=reconsolidate,
                current_scope=current_scope,
                scope_id=scope_id,
                allowed_scopes=allowed_scopes,
                context=context,
                profile="biomimetic" if self.strategy == "full" else "baseline",
                max_context_tokens=max_context_tokens,
            )
        if context is None or self._scope_registry is None:
            raise ValueError("Comparison retrieval requires a registered scope context")
        self._scope_registry.validate_context(context)
        scopes = tuple(allowed_scopes) if allowed_scopes is not None else ("project", "global")
        if any(scope not in {"project", "workspace", "global"} for scope in scopes):
            raise ValueError("Invalid allowed scope")
        if top_k <= 0 or max_context_tokens <= 0 or not query.strip():
            return []

        def scope_filter(chunk: MemoryChunk) -> bool:
            return scope_allows(chunk.metadata, context, scopes)

        setattr(
            scope_filter,
            "_mnemos_scope_filter",
            {
                "user_profile_id": context.user_profile_id,
                "project_id": context.project_id,
                "workspace_id": context.workspace_id,
                "allowed_scopes": scopes,
            },
        )
        pool = min(max(top_k * 6, 20), self.config.retrieval.max_candidates)
        scored: list[tuple[MemoryChunk, dict[str, Any]]] = []
        if self.strategy == "lexical":
            if not isinstance(self.store, SQLiteStore):
                raise TypeError("The lexical comparison requires SQLite FTS5")
            candidates = self.store.retrieve_lexical(query, top_k=pool, filter_fn=scope_filter)
            scored = [
                (
                    chunk,
                    {
                        "semantic": None,
                        "semantic_status": "not_measured",
                        "total": None,
                        "ranking_method": "sqlite_fts5_bm25",
                        "lexical_rank": rank,
                        "state_match": None,
                        "state_status": "disabled",
                        "graph_status": "disabled",
                        "graph_contribution": 0.0,
                    },
                )
                for rank, chunk in enumerate(candidates, 1)
            ]
            excluded: list[str] = []
        else:
            embedding = await embed_text_async(self._embedder, query, role="query")
            candidates = self.store.retrieve(embedding, top_k=pool, filter_fn=scope_filter)
            excluded = []
            for chunk in candidates:
                score = score_candidate(chunk, embedding, current_state=None, affective=False)
                if score.semantic < self.config.retrieval.relevance_floor:
                    excluded.append(chunk.id)
                    continue
                scored.append(
                    (
                        chunk,
                        {
                            **score.model_dump(),
                            "ranking_method": "cosine",
                            "graph_status": "disabled",
                        },
                    )
                )
            scored.sort(
                key=lambda row: (
                    -row[1]["total"],
                    row[0].metadata.get("scope") != current_scope,
                    row[0].id,
                )
            )
        selected: list[MemoryChunk] = []
        skipped: list[str] = []
        remaining = max_context_tokens
        for chunk, scores in scored:
            cost = context_token_upper_bound(chunk)
            if cost > remaining:
                skipped.append(chunk.id)
                continue
            returned = chunk.model_copy(deep=True)
            returned.metadata["retrieval_trace"] = {
                **scores,
                "profile": self.strategy,
                "scope_match": True,
                "context_token_upper_bound": cost,
                "token_count_method": "utf8_bytes_upper_bound",
                "candidate_backend": (
                    "sqlite_fts5"
                    if self.strategy == "lexical"
                    else getattr(self.store, "last_retrieval_backend", "unknown")
                ),
            }
            selected.append(returned)
            remaining -= cost
            if len(selected) == top_k:
                break
        self.store.mark_retrieved_many(
            [chunk.id for chunk in selected],
            retrieved_at=datetime.now(timezone.utc),
        )
        self.comparison_details.update(
            {
                "candidate_pool_limit": pool,
                "candidate_ids": [chunk.id for chunk in candidates],
                "relevance_excluded_ids": excluded,
                "budget_skipped_ids": skipped,
                "context_token_upper_bound": max_context_tokens - remaining,
            }
        )
        return selected


@contextmanager
def comparison_snapshots(
    service: LocalMemoryService,
) -> Iterator[tuple[str, dict[str, LocalMemoryService]]]:
    """Back up committed state once, then isolate every strategy's access writes."""
    source = service.engine.store
    if not isinstance(source, SQLiteStore):
        raise TypeError("Retrieval comparison requires an authoritative SQLite store")
    # These are disposable benchmark databases beside the attempt's owned DB.
    with TemporaryDirectory(prefix="retrieval-", dir=Path(source.db_path).parent) as directory:
        root = Path(directory)
        snapshot = root / "snapshot.sqlite"
        with source._lock:
            if source._conn.in_transaction:
                raise RuntimeError("Cannot compare an uncommitted snapshot")
            destination = sqlite3.connect(snapshot)
            try:
                source._conn.backup(destination)
            finally:
                destination.close()
        digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        with ExitStack() as stack:
            variants = {}
            for strategy in STRATEGIES:
                path = root / (strategy + ".sqlite")
                shutil.copyfile(snapshot, path)
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise RuntimeError("Retrieval snapshot copy changed")
                store = SQLiteStore(str(path))
                stack.callback(store.close)
                engine = ComparisonEngine(
                    config=service.engine.config.model_copy(deep=True),
                    llm=service.engine._llm,
                    embedder=service.engine._embedder,
                    store=store,
                    scope_registry=service.registry,
                    require_scope_context=True,
                )
                engine.strategy = strategy
                engine.comparison_details = {}
                variant = LocalMemoryService(engine, service.registry)
                setattr(
                    variant,
                    "_benchmark_provider_trace",
                    getattr(service, "_benchmark_provider_trace", {}),
                )
                variants[strategy] = variant
            yield digest, variants


class RetrievalComparatorAdapter(OllamaEvaluatorAdapter):
    """Ingest once with all modules, then compare four isolated read paths."""

    def evaluate_session(
        self,
        session: SessionScenario,
        profile: ExperimentProfile,
        repetition: int,
    ) -> RunObservation:
        if profile != DEFAULT_PROFILES[1]:
            raise ValueError("Retrieval comparison requires full-pipeline ingestion")
        self._comparison_session = session
        self._comparison_repetition = repetition
        self._comparisons: list[dict[str, Any]] = []
        observed = super().evaluate_session(session, profile, repetition)
        return replace(
            observed,
            raw_output={
                **observed.raw_output,
                "retrieval_comparisons": self._comparisons,
                "retrieval_comparison_notes": [
                    "Each query starts from the accepted post-lifecycle SQLite state.",
                    "Access writes are isolated; later queries do not inherit earlier retrievals.",
                    "Latency includes the retrieval dispatch, not snapshot creation or engine startup.",
                    "This is a fresh-clone comparison, not the T09 warm-latency gate.",
                ],
            },
        )

    def _quality(
        self,
        query_id: str,
        result: Mapping[str, Any] | None,
    ) -> tuple[str, bool | None]:
        session = self._comparison_session
        for query in session.queries:
            if query.query_id == query_id:
                return query.kind, (
                    self._answerable_hit(query, result, session)
                    if query.kind == "answerable"
                    else self._no_answer_correct(result)
                )
        for pair in session.state_pairs:
            for suffix in ("semantic", "state"):
                if query_id == pair.pair_id + ":" + suffix:
                    return "state_pair_" + suffix, self._state_hit(
                        pair,
                        result,
                        semantic=suffix == "semantic",
                    )
        raise ValueError("Unknown comparison query")

    def _run_retrieval(
        self,
        service: Any,
        query: str,
        context: Mapping[str, Any],
        profile: ExperimentProfile,
        operations: list[dict[str, Any]],
        errors: list[dict[str, Any]],
        query_id: str,
    ) -> Mapping[str, Any] | None:
        order = list(STRATEGIES)
        seed = f"{self._comparison_session.session_id}:{self._comparison_repetition}:{query_id}"
        random.Random(seed).shuffle(order)
        record: dict[str, Any] = {
            "query_id": query_id,
            "quality_kind": self._quality(query_id, None)[0],
            "strategy_order": order,
            "strategy_order_seed": seed,
            "source_database": str(
                getattr(getattr(getattr(service, "engine", None), "store", None), "db_path", "")
            ),
            "snapshot_sha256": None,
            "strategies": {},
        }
        self._comparisons.append(record)
        try:
            setup_started = time.perf_counter()
            with comparison_snapshots(service) as (digest, variants):
                record["snapshot_sha256"] = digest
                record["snapshot_setup_ms"] = (time.perf_counter() - setup_started) * 1000
                for strategy in order:
                    local_operations: list[dict[str, Any]] = []
                    local_errors: list[dict[str, Any]] = []
                    trace = self._provider_trace(service)
                    before = {
                        key: len(value) for key, value in trace.items() if isinstance(value, list)
                    }
                    started = time.perf_counter()
                    result = super()._run_retrieval(
                        variants[strategy],
                        query,
                        context,
                        profile,
                        local_operations,
                        local_errors,
                        query_id,
                    )
                    elapsed = (time.perf_counter() - started) * 1000
                    complete = result is not None and not local_errors
                    engine = variants[strategy].engine
                    details = getattr(engine, "comparison_details", {})
                    if result is not None:
                        result = {
                            **result,
                            "requested_profile": result.get("profile"),
                            "profile": strategy,
                        }
                    record["strategies"][strategy] = {
                        "status": "complete" if complete else "partial",
                        "quality_hit": self._quality(query_id, result)[1] if complete else None,
                        "latency_ms": elapsed,
                        "result": dict(result) if result is not None else None,
                        "errors": local_errors,
                        "snapshot_sha256": digest,
                        "details": {
                            **details,
                            "retrieval_features": {
                                "embedding": strategy != "lexical",
                                "affective": strategy == "full"
                                and engine.config.modules.affective
                                and engine.config.retrieval.affective_enabled,
                                "spreading": strategy == "full"
                                and engine.config.modules.spreading
                                and engine.config.retrieval.spreading_enabled,
                            },
                            "operations": local_operations,
                            "provider_trace_ranges": {
                                key: [before.get(key, 0), len(value)]
                                for key, value in self._provider_trace(service).items()
                                if isinstance(value, list)
                            },
                        },
                    }
                    errors.extend({**error, "strategy": strategy} for error in local_errors)
                    if strategy == "full":
                        operations.extend(local_operations)
        except Exception as exc:
            error = self._error("retrieval_comparison", exc, query_id)
            errors.append(error)
            for strategy in order:
                record["strategies"].setdefault(
                    strategy,
                    {
                        "status": "partial",
                        "quality_hit": None,
                        "latency_ms": None,
                        "result": None,
                        "errors": [error],
                        "snapshot_sha256": record["snapshot_sha256"],
                        "details": {},
                    },
                )
        full = record["strategies"]["full"]
        result = full["result"]
        return result if isinstance(result, Mapping) and full["status"] == "complete" else None
