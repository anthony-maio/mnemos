"""Durable, one-repetition graph measurement for the mechanism fixture.

This module deliberately uses the production engine and SQLite store while
keeping fixture population explicit.  It does not extract facts from a model.
The source rows are synthetic fixture evidence and are labelled as such in the
returned artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.embeddings import cosine_similarity
from mnemos.utils.storage import SQLiteStore

SCHEMA_VERSION = "local-memory-graph-measurement-v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "__dataclass_fields__"):
        return {name: _jsonable(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_component(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return result or "case"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _flush_close(store: SQLiteStore) -> None:
    store._conn.commit()
    store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    store.close()


def _scope_predicate(context: Any) -> Any:
    """Return a callable with the storage compiler metadata for SQLite FTS."""

    allowed = ("project", "workspace", "global")

    def predicate(chunk: MemoryChunk) -> bool:
        metadata = chunk.metadata
        if metadata.get("user_profile_id") != context.user_profile_id:
            return False
        scope = metadata.get("scope")
        if scope not in allowed:
            return False
        if scope == "global":
            return True
        wanted = context.project_id if scope == "project" else context.workspace_id
        return wanted is not None and metadata.get("scope_id") == wanted

    setattr(
        predicate,
        "_mnemos_scope_filter",
        {
            "user_profile_id": context.user_profile_id,
            "project_id": context.project_id,
            "workspace_id": context.workspace_id,
            "allowed_scopes": allowed,
        },
    )
    return predicate


def _fixture_event(case_id: str, fact: Any) -> EpisodicEvent:
    event_id = fact.evidence_id
    fingerprint = hashlib.sha256(f"{case_id}:{event_id}".encode("utf-8")).hexdigest()
    scope_id = fact.project_id if fact.scope == "project" else fact.workspace_id
    return EpisodicEvent(
        id=event_id,
        event_fingerprint=fingerprint,
        role="user",
        content=fact.content,
        normalized_content=fact.content.strip().lower(),
        scope=fact.scope,
        scope_id=scope_id,
        source="synthetic_fixture",
        hook_event="graph_measurement_fixture",
        metadata={
            "source_event_id": event_id,
            "user_profile_id": fact.user_profile_id,
            "session_id": case_id,
            "fixture": "synthetic_source_fact",
        },
        status="consolidated",
    )


def _build_graph_store(case: Any, facts: Iterable[Any], path: Path) -> tuple[Any, SQLiteStore]:
    from .module_mechanism import _engine_for, _insert_and_refresh

    materialized = list(facts)
    # Registry construction is independent of insertion order.  Put the
    # requested context first so a synthetic foreign control sharing a
    # fallback workspace cannot claim that workspace for its profile.
    registry_facts = sorted(
        materialized,
        key=lambda fact: (
            fact.user_profile_id != case.context.get("user_profile_id"),
            fact.project_id != case.context.get("project_id"),
            fact.workspace_id != case.context.get("workspace_id"),
        ),
    )
    engine, store = _engine_for(
        registry_facts,
        case.query,
        case.query_embedding,
        graph=True,
        affective=False,
        store_path=path,
    )
    for fact in materialized:
        store.append_episodic_event(_fixture_event(case.id, fact))
    _insert_and_refresh(engine, store, materialized)
    return engine, store


def _source_counts(store: SQLiteStore) -> dict[str, int]:
    event_count = int(store._conn.execute("SELECT COUNT(*) FROM episodic_events").fetchone()[0])
    evidence_count = int(store._conn.execute("SELECT COUNT(*) FROM fact_evidence").fetchone()[0])
    return {"episodic_events": event_count, "fact_evidence": evidence_count}


def _direct_pool(
    store: SQLiteStore, case: Any, predicate: Any
) -> tuple[list[str], list[str], list[str]]:
    limit = min(max(int(case.top_k) * 6, 20), 100)
    vector = store.retrieve(case.query_embedding, top_k=limit, filter_fn=predicate)
    lexical = store.retrieve_lexical(case.query, top_k=limit, filter_fn=predicate)
    vector_ids = [chunk.id for chunk in vector]
    lexical_ids = [chunk.id for chunk in lexical]
    return vector_ids, lexical_ids, sorted(set(vector_ids) | set(lexical_ids))


def _canonical_edges(edges: Mapping[str, Mapping[str, float]]) -> list[list[Any]]:
    return [
        [source, target, float(weight)]
        for source in sorted(edges)
        for target, weight in sorted(edges[source].items())
    ]


def _edge_digest(edges: Mapping[str, Mapping[str, float]]) -> str:
    return _digest(_canonical_edges(edges))


def _independent_activation(
    edges: Mapping[str, Mapping[str, float]],
    seed_id: str,
    allowed_ids: set[str],
    config: Any,
) -> dict[str, float]:
    """Enumerate paths instead of duplicating production's frontier algorithm."""
    if seed_id not in allowed_ids:
        return {}
    if len(allowed_ids) > int(config.max_visited):
        raise ValueError("Path oracle requires the fixture to fit the visit budget")
    best: dict[str, float] = {}
    for node_id in sorted(allowed_ids):
        paths = _paths(edges, seed_id, node_id, allowed_ids, config)
        if paths:
            best[node_id] = max(float(path["energy"]) for path in paths)
    return best


def _paths(
    edges: Mapping[str, Mapping[str, float]],
    seed_id: str,
    target_id: str,
    allowed_ids: set[str],
    config: Any,
) -> list[dict[str, Any]]:
    """Enumerate bounded simple paths using the live activation policy."""

    threshold = float(config.activation_threshold)
    attenuation = 1.0 - float(config.decay_rate)
    hop_limit = max(0, int(config.max_hops))
    paths: list[dict[str, Any]] = []

    def visit(node_id: str, path: list[str], energy: float) -> None:
        if node_id == target_id:
            paths.append({"nodes": list(path), "energy": energy, "hops": len(path) - 1})
            return
        if len(path) - 1 >= hop_limit:
            return
        for neighbor, raw_weight in sorted(
            edges.get(node_id, {}).items(), key=lambda item: (-float(item[1]), item[0])
        ):
            if neighbor not in allowed_ids or neighbor in path:
                continue
            weight = float(raw_weight)
            if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
                continue
            transferred = energy * attenuation * weight
            if transferred < threshold:
                continue
            visit(neighbor, [*path, neighbor], transferred)

    if seed_id in allowed_ids:
        visit(seed_id, [seed_id], float(config.initial_energy))
    paths.sort(key=lambda row: (-float(row["energy"]), row["nodes"]))
    return paths


def _has_cycle(edges: Mapping[str, Mapping[str, float]], allowed_ids: set[str]) -> bool:
    seen: set[str] = set()

    def visit(node_id: str, parent_id: str | None) -> bool:
        seen.add(node_id)
        for neighbor in edges.get(node_id, {}):
            if neighbor not in allowed_ids:
                continue
            if neighbor not in seen:
                if visit(neighbor, node_id):
                    return True
            elif neighbor != parent_id:
                return True
        return False

    return any(node_id not in seen and visit(node_id, None) for node_id in sorted(allowed_ids))


def _query(engine: Any, case: Any, context: Any, profile: str) -> tuple[list[Any], float]:
    from .module_mechanism import _run

    started = time.perf_counter()
    result = _run(
        engine.retrieve(
            case.query,
            top_k=int(case.top_k),
            reconsolidate=False,
            context=context,
            allowed_scopes=("project", "workspace", "global"),
            profile=profile,
            max_context_tokens=1200,
        )
    )
    return result, (time.perf_counter() - started) * 1000.0


def _activation(
    engine: Any, case: Any, predicate: Any, seed_id: str
) -> tuple[dict[str, float], float]:
    started = time.perf_counter()
    rows = engine.spreading_activation.retrieve(
        case.query_embedding,
        top_k=100,
        filter_fn=predicate,
        seed_ids=[seed_id],
        max_hops=int(engine.config.spreading.max_hops),
        max_visited=int(engine.config.spreading.max_visited),
    )
    return {row.id: float(row.energy) for row in rows}, (time.perf_counter() - started) * 1000.0


def _close_engines(*stores: Any) -> None:
    for store in stores:
        if store is not None:
            _flush_close(store)


def _snapshot(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _file_digest(path)}


def _validate_snapshot(snapshot: Mapping[str, Any]) -> None:
    path = Path(str(snapshot.get("path", "")))
    if not path.exists() or not path.is_file():
        raise ValueError(f"measurement artifact is missing: {path}")
    actual_bytes = path.stat().st_size
    actual_digest = _file_digest(path)
    if actual_bytes != int(snapshot.get("bytes", -1)) or actual_digest != snapshot.get("sha256"):
        raise ValueError(f"measurement artifact changed: {path}")


def _validate_existing_report(report_path: Path, binding: str) -> dict[str, Any]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid completed graph measurement: {report_path}") from exc
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("measurement_status") != "complete"
    ):
        raise ValueError(f"graph measurement is incomplete: {report_path}")
    if report.get("binding") != binding:
        raise ValueError(f"graph measurement binding changed: {report_path}")
    artifacts = report.get("artifacts", {})
    initial_snapshots = artifacts.get("initial_snapshots", {})
    if not isinstance(initial_snapshots, Mapping) or not initial_snapshots:
        raise ValueError(f"missing initial_snapshots: {report_path}")
    # Retrieval updates access metadata in the paired databases, so initial
    # hashes are historical evidence rather than current-file checks.
    final_snapshots = artifacts.get("final_snapshots", {})
    if not isinstance(final_snapshots, Mapping) or not final_snapshots:
        raise ValueError(f"missing final_snapshots: {report_path}")
    for snapshot in final_snapshots.values():
        if not isinstance(snapshot, Mapping):
            raise ValueError(f"invalid snapshot record: {report_path}")
        _validate_snapshot(snapshot)
    measurement_path = Path(str(artifacts.get("measurement_json", "")))
    if measurement_path != report_path:
        raise ValueError(f"measurement report path mismatch: {report_path}")
    return cast(dict[str, Any], report)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = {
        "measurement": Path(__file__).resolve(),
        "mechanism_adapter": root / "benchmarks/local_memory/module_mechanism.py",
        "engine": root / "mnemos/engine.py",
        "storage": root / "mnemos/utils/storage.py",
    }
    return {name: _file_digest(path) for name, path in paths.items() if path.exists()}


def measure_graph_repetition(case: Any, *, work_dir: Path, repetition: int) -> dict[str, Any]:
    """Measure one graph case repetition and retain every database artifact."""

    if repetition < 0:
        raise ValueError("repetition must be non-negative")
    if case.query_embedding is None:
        raise ValueError("graph case has no frozen query embedding")
    all_facts = (*case.facts, *case.scope_controls)
    case_digest = _digest(_jsonable(case))
    source_hashes = _source_hashes()
    binding_payload = {
        "schema_version": SCHEMA_VERSION,
        "case_sha256": case_digest,
        "repetition": repetition,
        "source_hashes": source_hashes,
        "python": platform.python_version(),
    }
    binding = _digest(binding_payload)
    repetition_root = work_dir / f"case-{_safe_component(case.id)}" / f"repetition-{repetition:03d}"
    repetition_root.mkdir(parents=True, exist_ok=True)
    existing_reports = sorted(repetition_root.glob("attempt-*/measurement.json"))
    if existing_reports:
        latest_report = existing_reports[-1]
        try:
            prior_payload = json.loads(latest_report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior_payload = {}
        if prior_payload.get("measurement_status") == "complete":
            return _validate_existing_report(latest_report, binding)
    attempt_numbers = [
        int(path.name.removeprefix("attempt-"))
        for path in repetition_root.glob("attempt-*")
        if path.is_dir() and path.name.removeprefix("attempt-").isdigit()
    ]
    run_dir = repetition_root / f"attempt-{max(attempt_numbers, default=-1) + 1:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "measurement.json"
    canonical_path = run_dir / "canonical.sqlite"
    baseline_path = run_dir / "baseline.sqlite"
    graph_path = run_dir / "graph.sqlite"
    cut_path = run_dir / "edge-cut.sqlite"
    reverse_path = run_dir / "reverse-insertion.sqlite"

    graph_engine, canonical_store = _build_graph_store(case, all_facts, canonical_path)
    source_counts = _source_counts(canonical_store)
    _close_engines(canonical_store)
    shutil.copy2(canonical_path, baseline_path)
    shutil.copy2(canonical_path, graph_path)
    initial_snapshots = {
        "canonical": _snapshot(canonical_path),
        "baseline": _snapshot(baseline_path),
        "graph": _snapshot(graph_path),
    }
    initial_hashes = {item["sha256"] for item in initial_snapshots.values()}
    if len(initial_hashes) != 1:
        raise RuntimeError("SQLite copy2 snapshots are not byte-identical")

    from .module_mechanism import _actual_seed, _context_model

    baseline_engine, baseline_store = _build_engine(case, all_facts, baseline_path, graph=False)
    graph_engine, graph_store = _build_engine(case, all_facts, graph_path, graph=True)
    context = _context_model(case.context)
    predicate = _scope_predicate(context)
    vector_ids, lexical_ids, direct_ids = _direct_pool(graph_store, case, predicate)
    # The seed is selected from the persisted vector pool before production
    # retrieval.  This avoids a hidden warm-up query and keeps the graph call
    # count in the artifact one baseline plus one graph retrieval.
    nodes = {chunk.id: chunk for chunk in graph_store.get_all()}
    selected_seed = _actual_seed(case.query_embedding, vector_ids, nodes)
    if selected_seed is None:
        raise RuntimeError("production vector pool did not provide a usable seed")

    baseline_results, baseline_latency = _query(baseline_engine, case, context, "baseline")
    graph_results, graph_latency = _query(graph_engine, case, context, "biomimetic")
    activation, activation_latency = _activation(graph_engine, case, predicate, selected_seed)
    graph_edges = graph_store.get_graph_edges()
    chunks = {chunk.id: chunk for chunk in graph_store.get_all()}
    allowed_ids = {chunk.id for chunk in chunks.values() if predicate(chunk)}
    independent = _independent_activation(
        graph_edges, selected_seed, allowed_ids, graph_engine.config.spreading
    )
    paths = _paths(
        graph_edges,
        selected_seed,
        case.target_id,
        allowed_ids,
        graph_engine.config.spreading,
    )
    target_chunk = chunks.get(case.target_id)
    target_similarity = (
        float(cosine_similarity(case.query_embedding, target_chunk.embedding or []))
        if target_chunk is not None
        else 0.0
    )
    production_trace = next(
        (
            chunk.metadata.get("retrieval_trace")
            for chunk in graph_results
            if chunk.id == case.target_id
        ),
        None,
    )
    energy_union = set(activation) | set(independent)
    energy_delta = max(
        (
            abs(float(activation.get(node_id, 0.0)) - float(independent.get(node_id, 0.0)))
            for node_id in energy_union
        ),
        default=0.0,
    )
    edge_digest = _edge_digest(graph_edges)
    baseline_ids = [chunk.id for chunk in baseline_results]
    graph_ids = [chunk.id for chunk in graph_results]

    _close_engines(baseline_store, graph_store)
    shutil.copy2(graph_path, cut_path)
    cut_store = SQLiteStore(str(cut_path))
    cut_store._conn.execute(
        "DELETE FROM memory_edges WHERE target_id = ? OR source_id = ?",
        (case.target_id, case.target_id),
    )
    cut_store._conn.commit()
    _close_engines(cut_store)
    cut_engine, cut_store = _build_engine(case, all_facts, cut_path, graph=True)
    cut_results, cut_latency = _query(cut_engine, case, context, "biomimetic")
    cut_direct_vector, _, _ = _direct_pool(cut_store, case, predicate)
    cut_seed = (
        selected_seed
        if selected_seed in cut_direct_vector
        else (cut_direct_vector[0] if cut_direct_vector else None)
    )
    cut_activation: dict[str, float] = {}
    cut_paths: list[dict[str, Any]] = []
    if cut_seed is not None:
        cut_activation, _ = _activation(cut_engine, case, predicate, cut_seed)
        cut_chunks = {chunk.id: chunk for chunk in cut_store.get_all()}
        cut_allowed_ids = {chunk.id for chunk in cut_chunks.values() if predicate(chunk)}
        cut_paths = _paths(
            cut_store.get_graph_edges(),
            cut_seed,
            case.target_id,
            cut_allowed_ids,
            cut_engine.config.spreading,
        )
    cut_ids = [chunk.id for chunk in cut_results]
    _close_engines(cut_store)

    reverse_engine, reverse_store = _build_graph_store(case, reversed(all_facts), reverse_path)
    reverse_results, reverse_latency = _query(reverse_engine, case, context, "biomimetic")
    reverse_edges = reverse_store.get_graph_edges()
    reverse_ids = [chunk.id for chunk in reverse_results]
    _close_engines(reverse_store)

    final_snapshots = {
        "canonical": _snapshot(canonical_path),
        "baseline": _snapshot(baseline_path),
        "graph": _snapshot(graph_path),
        "edge_cut": _snapshot(cut_path),
        "reverse_insertion": _snapshot(reverse_path),
    }
    semantic_floor = target_similarity >= float(graph_engine.config.retrieval.relevance_floor)
    valid_path = bool(paths)
    target_energy = float(activation.get(case.target_id, 0.0))
    target_absent_after_cut = case.target_id not in cut_activation
    eligible = case.target_id not in direct_ids and semantic_floor and valid_path
    success = (
        eligible
        and case.target_id in graph_ids
        and target_energy >= float(graph_engine.config.spreading.activation_threshold)
    )
    controls = {
        "snapshot": {
            "byte_identical_before_open": True,
            "canonical_baseline_graph_sha256": initial_snapshots["canonical"]["sha256"],
        },
        "activation_vs_independent_path": {
            "production_activation": activation,
            "independent_path_max": independent,
            "max_abs_energy_delta": energy_delta,
            "matches": energy_delta <= 1e-9,
        },
        "edge_cut": {
            "query_executed": True,
            "result_ids_after_cut": cut_ids,
            "alternate_paths_before": paths,
            "alternate_paths_after": cut_paths,
            "target_energy_after": float(cut_activation.get(case.target_id, 0.0)),
            "target_present_after": case.target_id in cut_activation,
        },
        "insertion_order": {
            "reverse_query_executed": True,
            "forward_edge_sha256": edge_digest,
            "reverse_edge_sha256": _edge_digest(reverse_edges),
            "edges_equal": _canonical_edges(graph_edges) == _canonical_edges(reverse_edges),
            "forward_result_ids": graph_ids,
            "reverse_result_ids": reverse_ids,
            "results_equal": graph_ids == reverse_ids,
            "forward_inserted_fact_ids": [fact.id for fact in all_facts],
            "reverse_inserted_fact_ids": [fact.id for fact in reversed(all_facts)],
            "status": "measured",
        },
    }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "binding": binding,
        "binding_payload": binding_payload,
        "case_id": case.id,
        "repetition": repetition,
        "eligible": eligible,
        "success": success,
        "status": "observed" if eligible else "ineligible",
        "measurement_status": "complete",
        "source_fixture": {
            "mode": "direct_fixture_insertion",
            "synthetic_source_events": True,
            "facts_and_controls": len(all_facts),
            **source_counts,
        },
        "query": {
            "text": case.query,
            "allowed_scopes": ["project", "workspace", "global"],
            "vector_ids": vector_ids,
            "lexical_ids": lexical_ids,
            "direct_union_ids": direct_ids,
            "actual_seed_ids": [selected_seed],
            "actual_selected_seed_id": selected_seed,
        },
        "production": {
            "baseline_result_ids": baseline_ids,
            "graph_result_ids": graph_ids,
            "baseline_latency_ms": round(baseline_latency, 6),
            "graph_latency_ms": round(graph_latency, 6),
            "activation_latency_ms": round(activation_latency, 6),
            "graph_target_trace": production_trace,
            "target_id": case.target_id,
            "target_in_direct_pool": case.target_id in direct_ids,
            "target_vector_candidate_rank": (
                vector_ids.index(case.target_id) + 1 if case.target_id in vector_ids else None
            ),
            "target_similarity": target_similarity,
            "semantic_floor": semantic_floor,
            "target_energy": target_energy,
            "target_in_graph_results": case.target_id in graph_ids,
            "valid_path": valid_path,
            "target_absent_after_edge_cut": target_absent_after_cut,
            "scope_control_ids_permitted_observed": [
                fact.id
                for fact in case.scope_controls
                if predicate(fact.chunk()) and (fact.id in direct_ids or fact.id in graph_ids)
            ],
            "scope_control_ids_forbidden_observed": [
                fact.id
                for fact in case.scope_controls
                if not predicate(fact.chunk())
                and (fact.id in direct_ids or fact.id in graph_ids or fact.id in activation)
            ],
            "scope_control_ids_in_direct_union": [
                fact.id for fact in case.scope_controls if fact.id in direct_ids
            ],
            "scope_control_ids_in_graph_results": [
                fact.id for fact in case.scope_controls if fact.id in graph_ids
            ],
            "graph_edge_count": len(_canonical_edges(graph_edges)),
            "graph_edge_sha256": edge_digest,
            "persisted_edges": graph_edges,
            "activated_node_count": len(activation),
        },
        "negative_cases": {
            "target_directly_eligible": case.target_id in direct_ids,
            "below_semantic_floor": not semantic_floor,
            "no_valid_path": not valid_path,
            "missing_from_graph_results": case.target_id not in graph_ids,
            "below_activation_threshold": target_energy
            < float(graph_engine.config.spreading.activation_threshold),
        },
        "correctness_controls": controls,
        "artifacts": {
            "directory": str(run_dir),
            "initial_snapshots": initial_snapshots,
            "final_snapshots": final_snapshots,
            "canonical_sqlite": str(canonical_path),
            "baseline_sqlite": str(baseline_path),
            "graph_sqlite": str(graph_path),
            "edge_cut_sqlite": str(cut_path),
            "reverse_insertion_sqlite": str(reverse_path),
        },
        "latencies_ms": {
            "baseline_retrieve": round(baseline_latency, 6),
            "graph_retrieve": round(graph_latency, 6),
            "edge_cut_retrieve": round(cut_latency, 6),
            "reverse_retrieve": round(reverse_latency, 6),
        },
        "coverage": {
            "cycle_observed": bool(_has_cycle(graph_edges, allowed_ids)),
            "activation_comparison_measured": True,
            "activation_comparison_matches": controls["activation_vs_independent_path"]["matches"],
            "edge_cut_query_measured": controls["edge_cut"]["query_executed"],
            "edge_cut_target_absent": target_absent_after_cut,
            "insertion_order_measured": controls["insertion_order"]["status"] == "measured",
        },
    }
    artifact["artifacts"]["measurement_json"] = str(report_path)
    _write_json(report_path, artifact)
    return artifact


def _build_engine(
    case: Any, facts: Iterable[Any], path: Path, *, graph: bool
) -> tuple[Any, SQLiteStore]:
    from .module_mechanism import _engine_for

    return _engine_for(
        facts,
        case.query,
        case.query_embedding,
        graph=graph,
        affective=False,
        store_path=path,
    )
