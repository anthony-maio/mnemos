"""Inactive complete-path capacity adapter; random outputs never select facts."""

import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def snapshot(memory, request, facts):
    from mnemos.scopes import scope_allows
    from mnemos.retrieval import assertion_status

    memory._refresh_registry()
    memory.registry.validate_context(request.context)
    store = memory.engine.store
    bindings, candidates = [], []
    with store.read_snapshot():
        for fact in facts:
            chunk = store.get(fact["id"])
            if chunk is None or not scope_allows(
                chunk.metadata, request.context, tuple(request.allowed_scopes)
            ):
                raise ValueError("Fact disappeared or became invisible")
            if chunk.metadata.get("source") == "captured_passthrough":
                raise ValueError("Expected accepted fact")
            expected = dict(
                id=chunk.id,
                version=chunk.version,
                content=chunk.content,
                scope=chunk.metadata["scope"],
                scope_id=chunk.metadata.get("scope_id"),
                assertion_status=assertion_status(chunk),
            )
            if any(fact.get(k) != v for k, v in expected.items()):
                raise ValueError("Retrieved fact changed before snapshot")
            evidence = sorted(store.list_fact_evidence(chunk.id), key=lambda e: e.evidence_id)
            if not evidence or len(evidence) > 8:
                raise ValueError("Expected one through eight complete owned sources")
            retained, sources, events = [], [], []
            for item in evidence:
                if item.fact_id != chunk.id or not item.excerpt.strip() or not item.source_event_id:
                    raise ValueError("Invalid evidence ownership")
                event = store.get_episodic_event(item.source_event_id)
                status = memory._original_source_status(
                    chunk, item, request.context, tuple(request.allowed_scopes)
                )
                if event is None or status != "retained_locally":
                    raise ValueError("Owned source unavailable")
                retained.append({**item.model_dump(mode="json"), "original_source_status": status})
                sources.append(
                    dict(
                        id=item.evidence_id,
                        source_event_id=item.source_event_id,
                        excerpt=item.excerpt,
                    )
                )
                events.append(event.model_dump(mode="json"))
            if sorted(fact["sources"], key=lambda e: e["evidence_id"]) != retained:
                raise ValueError("Retrieved evidence changed before snapshot")
            bindings.append(
                dict(
                    fact=expected,
                    metadata={k: v for k, v in chunk.metadata.items() if k != "retrieval_trace"},
                    evidence=retained,
                    events=events,
                )
            )
            candidates.append(
                dict(
                    id=chunk.id,
                    version=chunk.version,
                    content=chunk.content,
                    assertion_status=expected["assertion_status"],
                    sources=sources,
                )
            )
    return digest(bindings), dict(
        query=request.query, reference_date="2026-11-11", candidates=candidates
    )


def validate_outputs(response, target_ids):
    if not isinstance(response, dict) or response.get("target_ids") != target_ids:
        raise ValueError("Worker target binding mismatch")
    values = response.get("values")
    if (
        not isinstance(values, list)
        or len(values) != len(target_ids)
        or any(type(v) not in (float, int) or not math.isfinite(v) for v in values)
    ):
        raise ValueError("Invalid materialized worker outputs")
    if type(response.get("tokens")) is not int or not 2 <= response["tokens"] <= 8192:
        raise ValueError("Invalid worker token count")


async def capacity_retrieve(memory, request, infer, expected_ids):
    result = await memory.retrieve(request)
    if result.get("degraded") or [f["id"] for f in result["facts"]] != expected_ids:
        raise ValueError("Unexpected retrieval cohort or degraded operation")
    before, cohort = snapshot(memory, request, result["facts"])
    response = await infer(cohort)
    validate_outputs(response, expected_ids)
    after, current = snapshot(memory, request, result["facts"])
    if before != after or cohort != current:
        raise ValueError("Fact, source or authority changed during inference")
    result["capacity"] = {**response, "semantic_decisions": False, "snapshot_sha256": before}
    return result
