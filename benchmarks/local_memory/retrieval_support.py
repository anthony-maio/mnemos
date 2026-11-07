"""Development-only retrieval followed by the fixed QA/contrast intersection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import time
from typing import Any, Literal

from mnemos.context_budget import pack_context
from mnemos.contracts import RetrievalRequest
from mnemos.memory_service import LocalMemoryService
from mnemos.retrieval import assertion_status, render_context_chunk
from mnemos.scopes import scope_allows
from mnemos.support_intersection import (
    IntersectionResult,
    SupportSelector,
    assess_support_intersection,
)
from mnemos.types import MemoryChunk
from mnemos.utils.reliability import MnemosExternalServiceError, MnemosTransientError


@dataclass(frozen=True)
class RetrievalSupportResult:
    status: Literal["assessed", "unavailable"]
    baseline: dict[str, Any]
    candidate_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    context: str
    context_bytes: int
    budget_omitted_ids: tuple[str, ...]
    retrieval_ms: float
    support_ms: float
    elapsed_ms: float
    support: IntersectionResult | None
    error_type: str | None


class _BoundaryChanged(Exception):
    pass


async def retrieve_with_support(
    service: LocalMemoryService,
    request: RetrievalRequest,
    *,
    extractive: SupportSelector,
    contrast: SupportSelector,
    reference_date: date,
) -> RetrievalSupportResult:
    """Retrieve real candidates, assess that cohort and return current whole facts.

    This adapter is deliberately outside the installed/default retrieval path.
    It accepts no candidate IDs or answer labels. The request contract fixes the
    agreed development profile, scopes and byte budget. Both model selectors
    retain their existing prompts, thresholds, deadlines and source policies.
    """
    request = request.model_copy(deep=True)
    if (
        request.profile != "baseline"
        or request.view != "memory"
        or request.response_format != "full"
        or request.allowed_scopes != ["project", "global"]
        or request.top_k != 5
        or request.max_context_tokens != 2048
        or type(reference_date) is not date
    ):
        raise ValueError("Use the frozen retrieval/support development request")
    started = time.perf_counter()
    baseline: dict[str, Any] = {}
    ids: tuple[str, ...] = ()
    retrieval_ms = support_ms = 0.0
    support: IntersectionResult | None = None

    def result(
        status: Literal["assessed", "unavailable"],
        *,
        selected: tuple[str, ...] = (),
        context: str = "",
        omitted: tuple[str, ...] = (),
        error: str | None = None,
    ) -> RetrievalSupportResult:
        return RetrievalSupportResult(
            status,
            baseline,
            ids,
            selected,
            context,
            len(context.encode("utf-8")),
            omitted,
            retrieval_ms,
            support_ms,
            (time.perf_counter() - started) * 1000,
            support,
            error,
        )

    try:
        baseline = await service.retrieve(request)
    except (MnemosExternalServiceError, MnemosTransientError) as exc:
        retrieval_ms = (time.perf_counter() - started) * 1000
        return result("unavailable", error=type(exc).__name__)
    retrieval_ms = (time.perf_counter() - started) * 1000
    if baseline.get("degraded") or baseline.get("status") in {"degraded", "unavailable"}:
        return result("unavailable", error="retrieval_unavailable")
    if (
        baseline.get("profile") != "baseline"
        or baseline.get("view") != "memory"
        or baseline.get("token_count_method") != "utf8_bytes_upper_bound"
    ):
        return result("unavailable", error="retrieval_contract_changed")
    facts = baseline.get("facts")
    if not isinstance(facts, list) or len(facts) > request.top_k:
        return result("unavailable", error="invalid_retrieval_candidates")
    if any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in facts):
        return result("unavailable", error="invalid_retrieval_candidates")
    ids = tuple(row["id"] for row in facts)
    if len(set(ids)) != len(ids) or any(not key for key in ids):
        return result("unavailable", error="invalid_retrieval_candidates")
    store, context = service.engine.store, request.context
    scopes = tuple(request.allowed_scopes)

    def cohort(*, compare_response: bool = False) -> tuple[str, tuple[MemoryChunk, ...]]:
        service.registry.validate_context(context)
        chunks, bindings = [], []
        for row in facts:
            chunk = store.get(row["id"])
            if (
                chunk is None
                or not scope_allows(chunk.metadata, context, scopes)
                or assertion_status(chunk) not in {"asserted", "conflict"}
                or chunk.metadata.get("source") == "captured_passthrough"
            ):
                raise _BoundaryChanged
            evidence = [item.model_dump(mode="json") for item in store.list_fact_evidence(chunk.id)]
            if compare_response:
                for key, value in {
                    "version": chunk.version,
                    "content": chunk.content,
                    "scope": chunk.metadata.get("scope"),
                    "scope_id": chunk.metadata.get("scope_id"),
                    "assertion_status": assertion_status(chunk),
                }.items():
                    if row.get(key) != value:
                        raise _BoundaryChanged
                retained = [
                    {key: value for key, value in source.items() if key != "original_source_status"}
                    for source in row.get("sources", [])
                ]
                if retained != evidence:
                    raise _BoundaryChanged
            bindings.append(
                {
                    "id": chunk.id,
                    "version": chunk.version,
                    "content": chunk.content,
                    "metadata": {
                        key: value
                        for key, value in chunk.metadata.items()
                        if key != "retrieval_trace"
                    },
                    "evidence": evidence,
                }
            )
            chunks.append(chunk.model_copy(deep=True))
        return json.dumps(bindings, sort_keys=True, ensure_ascii=False), tuple(chunks)

    try:
        original, chunks = cohort(compare_response=True)
        rendered = "".join(render_context_chunk(chunk) for chunk in chunks)
        if baseline.get("context") != rendered or len(rendered.encode("utf-8")) > 2048:
            return result("unavailable", error="retrieval_context_changed")
        support_started = time.perf_counter()
        support = await assess_support_intersection(
            request.query,
            ids,
            store=store,
            registry=service.registry,
            context=context,
            extractive=extractive,
            contrast=contrast,
            reference_date=reference_date,
            allowed_scopes=scopes,
        )
        support_ms = (time.perf_counter() - support_started) * 1000
        if support.status != "assessed":
            return result("unavailable", error=support.error_type or "support_snapshot_unavailable")
        current, chunks = cohort()
        if current != original:
            raise _BoundaryChanged
        selected = {item.fact_id for item in support.supported}
        if not selected <= set(ids):
            raise _BoundaryChanged
        packed = pack_context(
            (chunk for chunk in chunks if chunk.id in selected),
            max_tokens=request.max_context_tokens,
            top_k=request.top_k,
        )
        final, _ = cohort()
        if final != original:
            raise _BoundaryChanged
        return result(
            "assessed",
            selected=tuple(chunk.id for chunk in packed.chunks),
            context=packed.text,
            omitted=packed.omitted_ids,
        )
    except _BoundaryChanged:
        return result("unavailable", error="retrieval_support_cohort_changed")
