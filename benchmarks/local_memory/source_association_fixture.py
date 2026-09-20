"""Materialize authored source-association cases into a durable SQLite fixture.

This module is deliberately a fixture builder, not a graph builder.  It uses the
same episodic-event, durable-job, and atomic extraction commit APIs as the
runtime, while keeping embeddings as supplied test data.  No model, ranking, or
measurement work happens here.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from mnemos.jobs import DurableJob, FactEvidence
from mnemos.scopes import ScopeContext, scope_allows
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.storage import SQLiteStore

FIXTURE_VERSION = "source-association-development-v1"
_BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)
_InsertionOrder = Literal["forward", "reverse"]
_VALID_EVENT_ROLES = {"user", "assistant", "tool", "system"}


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _scope_from_item(item: Mapping[str, Any], field: str) -> dict[str, str | None]:
    raw = item.get("scope")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field}.scope must be an object")
    scope = _require_text(raw.get("scope"), f"{field}.scope.scope").strip().lower()
    if scope not in {"project", "workspace", "global"}:
        raise ValueError(f"{field}.scope.scope is invalid")
    scope_id_raw = raw.get("scope_id")
    scope_id = (
        None if scope_id_raw is None else _require_text(scope_id_raw, f"{field}.scope.scope_id")
    )
    profile = _require_text(raw.get("user_profile_id"), f"{field}.scope.user_profile_id")
    project_raw = raw.get("project_id")
    workspace_raw = raw.get("workspace_id")
    project_id = (
        None if project_raw is None else _require_text(project_raw, f"{field}.scope.project_id")
    )
    workspace_id = (
        None
        if workspace_raw is None
        else _require_text(workspace_raw, f"{field}.scope.workspace_id")
    )
    if scope == "global":
        scope_id = None
    elif scope == "project" and (project_id is None or scope_id != project_id):
        raise ValueError(f"{field}.scope project scope_id does not match project_id")
    elif scope == "workspace" and (workspace_id is None or scope_id != workspace_id):
        raise ValueError(f"{field}.scope workspace scope_id does not match workspace_id")
    return {
        "scope": scope,
        "scope_id": scope_id,
        "user_profile_id": profile,
        "project_id": project_id,
        "workspace_id": workspace_id,
    }


def _scope_from_event(event: Mapping[str, Any], field: str) -> dict[str, str | None]:
    scope = _require_text(event.get("scope"), f"{field}.scope").strip().lower()
    if scope not in {"project", "workspace", "global"}:
        raise ValueError(f"{field}.scope is invalid")
    scope_id_raw = event.get("scope_id")
    scope_id = None if scope_id_raw is None else _require_text(scope_id_raw, f"{field}.scope_id")
    profile = _require_text(event.get("user_profile_id"), f"{field}.user_profile_id")
    project_raw = event.get("project_id")
    workspace_raw = event.get("workspace_id")
    project_id = None if project_raw is None else _require_text(project_raw, f"{field}.project_id")
    workspace_id = (
        None if workspace_raw is None else _require_text(workspace_raw, f"{field}.workspace_id")
    )
    if scope == "global":
        scope_id = None
    elif scope == "project" and (project_id is None or scope_id != project_id):
        raise ValueError(f"{field} project scope_id does not match project_id")
    elif scope == "workspace" and (workspace_id is None or scope_id != workspace_id):
        raise ValueError(f"{field} workspace scope_id does not match workspace_id")
    return {
        "scope": scope,
        "scope_id": scope_id,
        "user_profile_id": profile,
        "project_id": project_id,
        "workspace_id": workspace_id,
    }


def _validate_vector(value: Any, field: str, dimension: int | None) -> tuple[list[float], int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a numeric vector")
    vector: list[float] = []
    for index, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise ValueError(f"{field}[{index}] must be numeric")
        numeric = float(component)
        if not isfinite(numeric):
            raise ValueError(f"{field}[{index}] must be finite")
        vector.append(numeric)
    if not vector:
        raise ValueError(f"{field} must not be empty")
    if not any(component != 0.0 for component in vector):
        raise ValueError(f"{field} must not be an all-zero vector")
    if dimension is not None and len(vector) != dimension:
        raise ValueError(f"{field} has dimension {len(vector)}; expected {dimension}")
    return vector, len(vector)


def _content_items(case: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any], bool]]:
    facts = case.get("facts")
    controls = case.get("scope_controls", [])
    if not isinstance(facts, list) or not isinstance(controls, list):
        raise ValueError("case facts and scope_controls must be lists")
    items: list[tuple[str, Mapping[str, Any], bool]] = []
    for item in facts:
        if not isinstance(item, Mapping):
            raise ValueError("case facts must contain objects")
        items.append(("fact", item, True))
    for item in controls:
        if not isinstance(item, Mapping):
            raise ValueError("case scope_controls must contain objects")
        items.append(("scope_control", item, False))
    return items


def _allowed_by_context(scope: Mapping[str, str | None], context: Mapping[str, Any]) -> bool:
    resolved = ScopeContext.model_validate(dict(context))
    return scope_allows(dict(scope), resolved, ("project", "workspace", "global"))


def _validate_case(
    case: Mapping[str, Any],
    vectors: Mapping[str, Sequence[float]],
    destination: Path,
) -> tuple[str, list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, list[float]], int]:
    case_id = _require_text(case.get("id"), "case.id")
    if destination.exists():
        raise FileExistsError(f"Fixture destination already exists: {destination}")
    events = case.get("events")
    if not isinstance(events, list):
        raise ValueError("case.events must be a list")
    event_ids: set[str] = set()
    normalized_events: list[Mapping[str, Any]] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"events[{index}] must be an object")
        event_id = _require_text(event.get("id"), f"events[{index}].id")
        if event_id in event_ids:
            raise ValueError(f"duplicate event id: {event_id}")
        event_ids.add(event_id)
        role = _require_text(event.get("role"), f"events[{index}].role").strip().lower()
        if role not in _VALID_EVENT_ROLES:
            raise ValueError(f"events[{index}].role is unsupported")
        _require_text(event.get("source_kind"), f"events[{index}].source_kind")
        _require_text(event.get("content"), f"events[{index}].content")
        _scope_from_event(event, f"events[{index}]")
        normalized_events.append(event)

    items = _content_items(case)
    item_ids: set[str] = set()
    evidence_ids: set[str] = set()
    used_vectors: dict[str, list[float]] = {}
    dimension: int | None = None
    query_count = 0
    context = case.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("case.context must be an object")
    for kind, item, expected_query_eligible in items:
        item_id = _require_text(item.get("id"), f"{kind}.id")
        if item_id in item_ids:
            raise ValueError(f"duplicate fact/control id: {item_id}")
        item_ids.add(item_id)
        content = _require_text(item.get("content"), f"{kind} {item_id}.content")
        source_event_id = _require_text(
            item.get("source_event_id"), f"{kind} {item_id}.source_event_id"
        )
        if source_event_id not in event_ids:
            raise ValueError(f"{kind} {item_id} references unknown source event {source_event_id}")
        evidence_id = _require_text(item.get("evidence_id"), f"{kind} {item_id}.evidence_id")
        if evidence_id in evidence_ids:
            raise ValueError(f"duplicate evidence id: {evidence_id}")
        evidence_ids.add(evidence_id)
        scope = _scope_from_item(item, f"{kind} {item_id}")
        query_eligible = item.get("query_eligible")
        if query_eligible is not expected_query_eligible:
            raise ValueError(f"{kind} {item_id}.query_eligible must be {expected_query_eligible}")
        actual_eligible = _allowed_by_context(scope, context)
        if actual_eligible != query_eligible:
            raise ValueError(
                f"{kind} {item_id}.query_eligible does not match the authored query context"
            )
        if query_eligible:
            query_count += 1
        raw_vector = vectors.get(content)
        if raw_vector is None:
            raise ValueError(f"missing frozen vector for {kind} {item_id}")
        vector, dimension = _validate_vector(raw_vector, f"vectors[{content!r}]", dimension)
        used_vectors[content] = vector
        if scope["scope"] == "global" and scope["scope_id"] is not None:
            raise ValueError(f"global {kind} {item_id} cannot have a scope_id")

    expected_count = case.get("query_eligible_count", 36)
    if expected_count != query_count or query_count != 36:
        raise ValueError(f"case must contain exactly 36 query-eligible facts, found {query_count}")
    if not item_ids:
        raise ValueError("case must contain at least one fact")
    return case_id, normalized_events, [item for _, item, _ in items], used_vectors, dimension or 0


def _vector_digest(vectors: Mapping[str, Sequence[float]]) -> str:
    payload = json.dumps(
        {
            content: [float(component) for component in vector]
            for content, vector in sorted(vectors.items())
        },
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _event(case_id: str, raw: Mapping[str, Any], index: int) -> EpisodicEvent:
    timestamp = _BASE_TIME + timedelta(seconds=index)
    scope = _scope_from_event(raw, f"event {raw.get('id')}")
    event_id = _require_text(raw.get("id"), "event.id")
    content = _require_text(raw.get("content"), f"event {event_id}.content")
    source_kind = _require_text(raw.get("source_kind"), f"event {event_id}.source_kind")
    metadata = {
        "fixture": FIXTURE_VERSION,
        "synthetic": True,
        "origin": raw.get("origin", "authored_development_event"),
        "source_kind": source_kind,
        "user_profile_id": scope["user_profile_id"],
        "project_id": scope["project_id"],
        "workspace_id": scope["workspace_id"],
    }
    return EpisodicEvent(
        id=event_id,
        event_fingerprint=hashlib.sha256(f"{case_id}\0{event_id}".encode("utf-8")).hexdigest(),
        role=_require_text(raw.get("role"), f"event {event_id}.role"),
        content=content,
        normalized_content=" ".join(content.lower().split()),
        scope=scope["scope"] or "global",
        scope_id=scope["scope_id"],
        source=source_kind,
        metadata=metadata,
        status="pending",
        created_at=timestamp,
        occurred_at=timestamp,
    )


def populate_source_association_fixture(
    case: Mapping[str, Any],
    vectors: Mapping[str, Sequence[float]],
    destination: str | Path,
    *,
    insertion_order: _InsertionOrder = "forward",
) -> dict[str, Any]:
    """Persist one authored graph case through the durable extraction APIs.

    All case, source-reference, and vector validation runs before the SQLite
    destination is created.  ``insertion_order`` exists only to test that source
    association diagnostics do not depend on write order.
    """

    if insertion_order not in {"forward", "reverse"}:
        raise ValueError("insertion_order must be 'forward' or 'reverse'")
    if not isinstance(vectors, Mapping):
        raise ValueError("vectors must be a content-to-vector mapping")
    destination_path = Path(destination).expanduser().resolve()
    case_id, raw_events, raw_items, used_vectors, dimension = _validate_case(
        case, vectors, destination_path
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    facts_by_event: dict[str, list[Mapping[str, Any]]] = {}
    for item in raw_items:
        source_event_id = _require_text(item.get("source_event_id"), "item.source_event_id")
        facts_by_event.setdefault(source_event_id, []).append(item)
    canonical_event_indexes = {str(event["id"]): index for index, event in enumerate(raw_events)}
    raw_events_ordered = list(raw_events)
    if insertion_order == "reverse":
        raw_events_ordered.reverse()

    store = SQLiteStore(db_path=str(destination_path))
    completed_jobs = 0
    job_ids: list[str] = []
    try:
        for raw_event in raw_events_ordered:
            canonical_index = canonical_event_indexes[str(raw_event["id"])]
            event = _event(case_id, raw_event, canonical_index)
            store.append_episodic_event(event)
            event_items = facts_by_event.get(event.id, [])
            if insertion_order == "reverse":
                event_items = list(reversed(event_items))
            if not event_items:
                continue
            job_id = f"{FIXTURE_VERSION}:{case_id}:{event.id}"
            owner = f"{FIXTURE_VERSION}:{case_id}:owner"
            timestamp = _BASE_TIME + timedelta(seconds=canonical_index)
            job = DurableJob(
                id=job_id,
                kind="extract",
                payload={
                    "event_id": event.id,
                    "case_id": case_id,
                    "fixture": FIXTURE_VERSION,
                    "synthetic_extraction": True,
                },
                created_at=timestamp,
                updated_at=timestamp,
                available_at=timestamp,
            )
            store.enqueue_job(job)
            claimed = store.claim_jobs(owner=owner, job_id=job_id, now=timestamp)[0]
            chunks: list[MemoryChunk] = []
            evidence: list[FactEvidence] = []
            for item in event_items:
                item_id = _require_text(item.get("id"), "item.id")
                content = _require_text(item.get("content"), f"item {item_id}.content")
                scope = _scope_from_item(item, f"item {item_id}")
                query_eligible = bool(item.get("query_eligible"))
                metadata = {
                    **scope,
                    "fixture": FIXTURE_VERSION,
                    "synthetic": True,
                    "synthetic_extraction": True,
                    "assertion_status": "asserted",
                    "source": "synthetic_extraction",
                    "source_kind": item.get("source_kind", "explicit_remember"),
                    "query_eligible": query_eligible,
                    "control_kind": item.get("control_kind"),
                }
                item_time = _BASE_TIME + timedelta(seconds=canonical_index)
                chunks.append(
                    MemoryChunk(
                        id=item_id,
                        content=content,
                        embedding=list(used_vectors[content]),
                        metadata=metadata,
                        created_at=item_time,
                        updated_at=item_time,
                    )
                )
                evidence.append(
                    FactEvidence(
                        evidence_id=_require_text(
                            item.get("evidence_id"), f"item {item_id}.evidence_id"
                        ),
                        fact_id=item_id,
                        source_event_id=event.id,
                        excerpt=content,
                        source_ref=f"{FIXTURE_VERSION}:{case_id}:{event.id}",
                        created_at=item_time,
                    )
                )
            committed = store.commit_fact_and_job(
                claimed.id,
                owner=owner,
                lease_token=claimed.lease_token,
                chunks=chunks,
                evidence=evidence,
                processed_event_ids=[event.id],
                result={
                    "decision": "facts",
                    "fixture": FIXTURE_VERSION,
                    "synthetic_extraction": True,
                },
                now=timestamp,
            )
            if not committed:
                raise RuntimeError(f"Fixture extraction job lease was not committed: {job_id}")
            completed_jobs += 1
            job_ids.append(job_id)
    finally:
        store.close()

    query_eligible_count = sum(1 for item in raw_items if item.get("query_eligible") is True)
    control_count = sum(1 for item in raw_items if item.get("query_eligible") is False)
    return {
        "schema_version": f"{FIXTURE_VERSION}-fixture-v1",
        "status": "materialized",
        "case_id": case_id,
        "db_path": str(destination_path),
        "insertion_order": insertion_order,
        "synthetic_extraction": True,
        "measurement": False,
        "counts": {
            "episodic_events": len(raw_events),
            "facts": len(raw_items) - control_count,
            "scope_controls": control_count,
            "query_eligible_facts": query_eligible_count,
            "evidence": len(raw_items),
            "jobs": len(job_ids),
            "completed_jobs": completed_jobs,
        },
        "job_ids": job_ids,
        "source_event_ids": [str(event.get("id")) for event in raw_events],
        "vector": {
            "dimension": dimension,
            "content_count": len(used_vectors),
            "sha256": _vector_digest(used_vectors),
        },
        "provenance": {
            "fixture": FIXTURE_VERSION,
            "label": "authored synthetic development facts and source events",
            "models_called": False,
            "ranking_performed": False,
            "measurement_performed": False,
        },
    }


__all__ = ["FIXTURE_VERSION", "populate_source_association_fixture"]
