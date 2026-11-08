"""One isolated local-service observation for live tokenizer integration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

from mnemos.config import MnemosConfig
from mnemos.context_budget import HostTokenizer
from mnemos.contracts import RetrievalRequest
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeContext, ScopeRegistry
from mnemos.utils.embeddings import EmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accepted_projection(store: SQLiteStore) -> str:
    """Access rows may change; accepted facts, sources, evidence and jobs may not."""
    values = {
        table: [list(row) for row in store._conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
        for table in ("memory_chunks", "fact_evidence", "episodic_events", "memory_jobs")
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


async def observe_retrieval(
    case: dict[str, Any],
    destination: Path,
    *,
    query: str,
    budget: int,
    embedder: EmbeddingProvider,
    tokenizer: HostTokenizer | None,
) -> dict[str, Any]:
    """Copy a verified snapshot, measure retrieval, and close it on every exit."""
    source = Path(case["database"])
    if destination.exists() or destination.resolve() == source.resolve():
        raise ValueError("Observation requires a new database destination")
    if file_hash(source) != case["database_sha256"]:
        raise ValueError("Original scenario database changed")
    config = MnemosConfig.model_validate(case["config"])
    if any(config.modules.model_dump().values()):
        raise ValueError("Experimental modules must remain disabled")
    registry = ScopeRegistry.model_validate(case["registry"])
    context = ScopeContext.model_validate(case["context"])
    registry.validate_context(context)
    shutil.copyfile(source, destination)
    store = SQLiteStore(str(destination))
    try:
        before = accepted_projection(store)
        if before != case["logical_sha256"]:
            raise ValueError("Accepted snapshot projection changed")
        engine = MnemosEngine(
            config=config,
            store=store,
            embedder=embedder,
            llm=UnavailableLLMProvider(),
            context_tokenizer=tokenizer,
        )
        service = LocalMemoryService(engine, registry)
        request = RetrievalRequest(
            query=query,
            context=context,
            profile="baseline",
            view="memory",
            allowed_scopes=["project", "global"],
            top_k=5,
            max_context_tokens=budget,
            response_format="full",
        )
        started = time.perf_counter()
        result = await service.retrieve(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if accepted_projection(store) != before:
            raise ValueError("Retrieval changed accepted facts, evidence, sources or jobs")
        if store._conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Observation database integrity failure")
        if store._conn.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Observation database foreign key failure")
        # Retain the ranked candidate trace, including budget omissions, so the
        # auditor can distinguish ranking differences from packing differences.
        traces = list(engine._retrieval_traces.values())
        return {
            "result": result,
            "elapsed_ms": elapsed_ms,
            "candidate_traces": traces,
            "accepted_projection": before,
            "integrity": "ok",
            "foreign_key_errors": 0,
        }
    finally:
        store.close()
        if file_hash(source) != case["database_sha256"]:
            raise ValueError("Original scenario changed during observation")
