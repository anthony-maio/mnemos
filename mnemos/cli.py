"""
mnemos/cli.py — Command-line interface for mnemos memory operations.

Provides shell-friendly commands that work with Claude Code hooks,
cron jobs, and other automation. Unlike the MCP server (stdio transport),
these commands run as one-shot processes and exit.

Usage:
    mnemos-cli store "I prefer dark mode"
    mnemos-cli retrieve "user preferences" --top-k 5
    mnemos-cli consolidate
    mnemos-cli stats
    mnemos-cli doctor
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, cast

from .antigravity import (
    ANTIGRAVITY_HOST_CHOICES,
    ANTIGRAVITY_TARGET_CHOICES,
    build_antigravity_artifact,
    build_antigravity_policy,
)
from .config import MemoryGovernanceConfig, MemorySafetyConfig, MnemosConfig, SurprisalConfig
from .engine import MnemosEngine
from .health import run_health_checks
from .hook_autostore import SUPPORTED_HOOK_EVENTS, decide_autostore, parse_hook_payload
from .inspectability import build_chunk_inspection
from .observability import configure_logging, log_event
from .runtime import (
    build_embedder_from_env,
    build_llm_from_env,
    build_mnemos_config_from_env,
    build_store_from_env,
)
from .types import Interaction, RetrievalFeedbackEvent
from .utils.llm import MockLLMProvider, LLMProvider
from .utils.storage import SQLiteStore

PROFILE_CHOICES = ("default",)
VALID_SCOPES = ("project", "workspace", "global")
AUDIT_SCOPES = ("all", "project", "workspace", "global")
MIGRATION_SOURCE_STORE_CHOICES = ("sqlite",)
MIGRATION_TARGET_CHOICES = ("sqlite",)
FEEDBACK_EVENT_CHOICES = ("helpful", "not_helpful", "missed_memory")
MemoryAction = Literal["allow", "redact", "block"]
CaptureMode = Literal["all", "manual_only", "hooks_only"]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _memory_action_from_env(name: str, default: MemoryAction) -> MemoryAction:
    raw = (os.getenv(name, default) or default).strip().lower()
    if raw in {"allow", "redact", "block"}:
        return cast(MemoryAction, raw)
    return default


def _capture_mode_from_env(name: str, default: CaptureMode) -> CaptureMode:
    raw = (os.getenv(name, default) or default).strip().lower()
    if raw in {"all", "manual_only", "hooks_only"}:
        return cast(CaptureMode, raw)
    return default


def _infer_embedding_provider(llm_provider: str) -> str:
    provider = llm_provider.lower()
    if provider in {"openai", "openclaw", "ollama"}:
        return provider
    return "simple"


def _build_profile_env(
    *,
    profile: str,
    llm_provider: str,
    embedding_provider: str | None,
    model: str | None,
    sqlite_path: str,
) -> dict[str, str]:
    if profile not in PROFILE_CHOICES:
        raise ValueError(f"Unsupported profile: {profile!r}")

    env: dict[str, str] = {
        "MNEMOS_LLM_PROVIDER": llm_provider,
        "MNEMOS_EMBEDDING_PROVIDER": embedding_provider or _infer_embedding_provider(llm_provider),
    }
    if model:
        env["MNEMOS_LLM_MODEL"] = model

    env["MNEMOS_STORE_TYPE"] = "sqlite"
    env["MNEMOS_SQLITE_PATH"] = sqlite_path

    return env


def _render_profile_env(profile: str, env: dict[str, str], output_format: str) -> str:
    ordered = {key: env[key] for key in sorted(env)}
    if output_format == "json":
        return json.dumps({"profile": profile, "env": ordered}, indent=2)
    if output_format == "dotenv":
        return "\n".join(f"{key}={value}" for key, value in ordered.items())
    if output_format == "bash":
        return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in ordered.items())
    if output_format == "powershell":
        lines: list[str] = []
        for key, value in ordered.items():
            escaped = value.replace("'", "''")
            lines.append(f"$env:{key}='{escaped}'")
        return "\n".join(lines)
    raise ValueError(f"Unsupported output format: {output_format!r}")


def _parse_allowed_scopes(raw: str) -> tuple[str, ...]:
    scopes = [scope.strip().lower() for scope in raw.split(",") if scope.strip()]
    if not scopes:
        return VALID_SCOPES
    invalid = [scope for scope in scopes if scope not in VALID_SCOPES]
    if invalid:
        raise ValueError(
            f"Invalid allowed scope(s): {', '.join(invalid)}. "
            f"Expected one or more of: {', '.join(VALID_SCOPES)}."
        )
    deduped: list[str] = []
    for scope in scopes:
        if scope not in deduped:
            deduped.append(scope)
    return tuple(deduped)


def _build_antigravity_policy(host: str) -> str:
    return build_antigravity_policy(cast(Any, host))


def _build_antigravity_artifact(host: str, target: str) -> str:
    return build_antigravity_artifact(cast(Any, host), cast(Any, target))


def _chunk_scope(chunk: Any) -> tuple[str, str | None]:
    scope_raw = chunk.metadata.get("scope")
    scope = str(scope_raw).lower() if isinstance(scope_raw, str) else "global"
    if scope not in VALID_SCOPES:
        scope = "global"
    if scope == "global":
        return scope, None
    scope_id_raw = chunk.metadata.get("scope_id")
    scope_id = str(scope_id_raw).strip() if isinstance(scope_id_raw, str) else "default"
    if not scope_id:
        scope_id = "default"
    return scope, scope_id


def _scope_filter(chunk: Any, scope: str, scope_id: str) -> bool:
    if scope == "all":
        return True
    chunk_scope, chunk_scope_id = _chunk_scope(chunk)
    if chunk_scope != scope:
        return False
    if scope in {"project", "workspace"}:
        target = scope_id.strip() if scope_id.strip() else "default"
        return chunk_scope_id == target
    return True


def _query_filter(chunk: Any, query: str) -> bool:
    if not query.strip():
        return True
    q = query.lower()
    if q in chunk.content.lower():
        return True
    for value in chunk.metadata.values():
        if isinstance(value, str) and q in value.lower():
            return True
    return False


def _age_filter(chunk: Any, older_than_days: int) -> bool:
    if older_than_days <= 0:
        return True
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    updated = chunk.updated_at
    created = chunk.created_at
    return bool(updated < cutoff and created < cutoff)


def _sort_chunks(chunks: list[Any], sort_by: str) -> list[Any]:
    if sort_by == "salience":
        return sorted(chunks, key=lambda chunk: chunk.salience, reverse=True)
    if sort_by == "access_count":
        return sorted(chunks, key=lambda chunk: chunk.access_count, reverse=True)
    if sort_by == "updated_at":
        return sorted(chunks, key=lambda chunk: chunk.updated_at, reverse=True)
    return sorted(chunks, key=lambda chunk: chunk.created_at, reverse=True)


def _serialize_chunk(chunk: Any) -> dict[str, Any]:
    scope, scope_id = _chunk_scope(chunk)
    return {
        "id": chunk.id,
        "content": chunk.content,
        "scope": scope,
        "scope_id": scope_id,
        "salience": round(chunk.salience, 4),
        "version": chunk.version,
        "access_count": chunk.access_count,
        "created_at": chunk.created_at.isoformat(),
        "updated_at": chunk.updated_at.isoformat(),
        "metadata": chunk.metadata,
    }


def _normalize_feedback_event_type(event_type: str | None) -> str | None:
    value = (event_type or "").strip()
    if not value:
        return None
    if value not in FEEDBACK_EVENT_CHOICES:
        raise ValueError(
            f"Invalid feedback event type: {value!r}. "
            f"Expected one of: {', '.join(FEEDBACK_EVENT_CHOICES)}."
        )
    return value


def _normalize_feedback_scope(scope: str, scope_id: str) -> tuple[str | None, str | None]:
    if scope == "all":
        return None, None
    if scope == "global":
        return "global", None
    normalized_scope_id = scope_id.strip() if scope_id.strip() else "default"
    return scope, normalized_scope_id


def _serialize_feedback_event(event: RetrievalFeedbackEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "event_type": event.event_type,
        "query": event.query,
        "scope": event.scope,
        "scope_id": event.scope_id if event.scope != "global" else None,
        "chunk_ids": list(event.chunk_ids),
        "notes": event.notes,
        "created_at": event.created_at.isoformat(),
    }


def _filtered_feedback_events(
    engine: MnemosEngine,
    *,
    event_type: str | None,
    scope: str,
    scope_id: str,
) -> list[RetrievalFeedbackEvent]:
    normalized_event_type = _normalize_feedback_event_type(event_type)
    normalized_scope, normalized_scope_id = _normalize_feedback_scope(scope, scope_id)
    events = engine.store.list_feedback_events(
        event_type=normalized_event_type,
        scope=normalized_scope,
        scope_id=normalized_scope_id,
    )
    return sorted(events, key=lambda event: event.created_at, reverse=True)


def _build_store_for_migration(
    *,
    store_type: str,
    sqlite_path: str = "",
) -> Any:
    if store_type == "sqlite":
        return SQLiteStore(db_path=sqlite_path or "mnemos_memory.db")
    raise ValueError(f"Unsupported migration store: {store_type!r}")


def _migrate_chunks(*, source_store: Any, target_store: Any, dry_run: bool) -> dict[str, Any]:
    chunks = source_store.get_all()
    migrated = 0
    edge_sets_migrated = 0
    if not dry_run:
        for chunk in chunks:
            target_store.store(chunk)
            migrated += 1
        source_edges: dict[str, dict[str, float]] = getattr(
            source_store, "get_graph_edges", lambda: {}
        )()
        replace_neighbors = getattr(target_store, "replace_graph_neighbors", None)
        if callable(replace_neighbors):
            for chunk_id, neighbors in source_edges.items():
                replace_neighbors(chunk_id, neighbors)
                edge_sets_migrated += 1
    return {
        "scanned": len(chunks),
        "migrated": migrated,
        "edge_sets_migrated": edge_sets_migrated,
        "skipped": 0,
        "dry_run": dry_run,
    }


def _close_store_quietly(store: Any) -> None:
    close = getattr(store, "close", None)
    if callable(close):
        close()


def _filtered_chunks(
    engine: MnemosEngine,
    *,
    scope: str,
    scope_id: str,
    query: str = "",
    older_than_days: int = 0,
) -> list[Any]:
    chunks = engine.store.get_all()
    filtered = [
        chunk
        for chunk in chunks
        if _scope_filter(chunk, scope, scope_id)
        and _query_filter(chunk, query)
        and _age_filter(chunk, older_than_days)
    ]
    return filtered


def _build_engine() -> MnemosEngine:
    """Build engine from environment, defaulting to SQLite for persistence."""
    try:
        llm = build_llm_from_env()
        config = build_mnemos_config_from_env(default_store_type="sqlite")
        engine = MnemosEngine(
            config=config,
            llm=llm,
            embedder=build_embedder_from_env(default_provider="simple"),
            store=build_store_from_env(default_store_type="sqlite"),
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    health = run_health_checks()
    log_event(
        "mnemos.startup",
        transport="cli",
        status=health["status"],
        profile=health["profile"],
        store_type=health["store_type"],
        llm_provider=health["llm_provider"],
        embedding_provider=health["embedding_provider"],
    )
    if health["status"] != "ready":
        log_event(
            "mnemos.degraded_mode",
            transport="cli",
            status=health["status"],
            summary=health["summary"],
        )
    return engine


def _build_hook_engine() -> MnemosEngine:
    """
    Build a fast-path engine for deterministic hook ingestion.

    Claude hook capture should not block on slow background cognition calls.
    We keep the real embedder/store/config so persisted memory still behaves
    normally, but swap in a local mock LLM so surprisal/affective tagging
    degrade to deterministic neutral behavior instead of timing out.
    """
    try:
        engine = MnemosEngine(
            config=build_mnemos_config_from_env(default_store_type="sqlite"),
            llm=MockLLMProvider(),
            embedder=build_embedder_from_env(default_provider="simple"),
            store=build_store_from_env(default_store_type="sqlite"),
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    return engine


async def _cmd_store(args: argparse.Namespace) -> None:
    engine = _build_engine()
    interaction = Interaction(role=args.role, content=args.content)
    result = await engine.process(
        interaction,
        scope=args.scope,
        scope_id=(args.scope_id or None),
    )
    output = {
        "stored": result.stored,
        "salience": round(result.salience, 4),
        "reason": result.reason,
        "chunk_id": result.chunk.id if result.chunk else None,
        "scope": (result.chunk.metadata.get("scope") if result.chunk else None),
        "scope_id": (result.chunk.metadata.get("scope_id") if result.chunk else None),
    }
    print(json.dumps(output, indent=2))


async def _cmd_retrieve(args: argparse.Namespace) -> None:
    engine = _build_engine()
    allowed_scopes = _parse_allowed_scopes(args.allowed_scopes)
    chunks = await engine.retrieve(
        args.query,
        top_k=args.top_k,
        reconsolidate=args.reconsolidate,
        current_scope=args.current_scope,
        scope_id=(args.scope_id or None),
        allowed_scopes=allowed_scopes,
    )
    results = []
    for chunk in chunks:
        results.append(
            {
                "id": chunk.id,
                "content": chunk.content,
                "salience": round(chunk.salience, 4),
                "version": chunk.version,
                "scope": chunk.metadata.get("scope", "global"),
                "scope_id": chunk.metadata.get("scope_id"),
            }
        )
    print(json.dumps(results, indent=2))


async def _cmd_consolidate(args: argparse.Namespace) -> None:
    engine = _build_engine()
    result = await engine.consolidate()
    output = {
        "facts_extracted": result.facts_extracted,
        "chunks_pruned": result.chunks_pruned,
        "duration_seconds": round(result.duration_seconds, 3),
    }
    print(json.dumps(output, indent=2))


async def _cmd_feedback(args: argparse.Namespace) -> None:
    engine = _build_engine()
    event = RetrievalFeedbackEvent(
        event_type=args.event_type,
        query=args.query,
        scope=args.scope,
        scope_id=(args.scope_id or None),
        chunk_ids=list(args.chunk_ids or []),
        notes=args.notes,
    )
    engine.store.store_feedback_event(event)
    print(
        json.dumps(
            {
                "stored": True,
                "event_id": event.id,
                "event_type": event.event_type,
                "query": event.query,
                "scope": event.scope,
                "scope_id": event.scope_id,
                "chunk_ids": event.chunk_ids,
                "notes": event.notes,
                "created_at": event.created_at.isoformat(),
            },
            indent=2,
        )
    )


async def _cmd_feedback_list(args: argparse.Namespace) -> None:
    engine = _build_engine()
    events = _filtered_feedback_events(
        engine,
        event_type=args.event_type,
        scope=args.scope,
        scope_id=args.scope_id,
    )
    limited = events[: args.limit] if args.limit > 0 else events
    print(
        json.dumps(
            {
                "total": len(events),
                "showing": len(limited),
                "event_type": _normalize_feedback_event_type(args.event_type),
                "scope": args.scope,
                "scope_id": args.scope_id if args.scope not in {"all", "global"} else None,
                "events": [_serialize_feedback_event(event) for event in limited],
            },
            indent=2,
        )
    )


async def _cmd_feedback_export(args: argparse.Namespace) -> None:
    engine = _build_engine()
    events = _filtered_feedback_events(
        engine,
        event_type=args.event_type,
        scope=args.scope,
        scope_id=args.scope_id,
    )
    limited = events[: args.limit] if args.limit > 0 else events
    serialized = [_serialize_feedback_event(event) for event in limited]

    if args.format == "jsonl":
        output = "\n".join(json.dumps(item) for item in serialized)
    else:
        output = json.dumps(
            {
                "event_type": _normalize_feedback_event_type(args.event_type),
                "scope": args.scope,
                "scope_id": args.scope_id if args.scope not in {"all", "global"} else None,
                "total": len(serialized),
                "events": serialized,
            },
            indent=2,
        )
    if args.output:
        text = output if output.endswith("\n") else f"{output}\n"
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(output)


async def _cmd_stats(args: argparse.Namespace) -> None:
    engine = _build_engine()
    stats = engine.get_stats()
    print(json.dumps(stats, indent=2, default=str))


async def _cmd_inspect(args: argparse.Namespace) -> None:
    engine = _build_engine()
    payload = build_chunk_inspection(
        engine,
        args.chunk_id,
        query=args.query,
        current_scope=args.current_scope,
        scope_id=args.scope_id,
        allowed_scopes=_parse_allowed_scopes(args.allowed_scopes),
    )
    if payload is None:
        print(json.dumps({"error": f"Memory chunk {args.chunk_id!r} not found."}, indent=2))
        return
    print(json.dumps(payload, indent=2))


async def _cmd_list(args: argparse.Namespace) -> None:
    engine = _build_engine()
    chunks = _filtered_chunks(
        engine,
        scope=args.scope,
        scope_id=args.scope_id,
        query=args.query,
    )
    chunks = _sort_chunks(chunks, args.sort_by)
    limited = chunks[: args.limit] if args.limit > 0 else chunks
    print(
        json.dumps(
            {
                "total": len(chunks),
                "showing": len(limited),
                "scope": args.scope,
                "scope_id": args.scope_id if args.scope != "global" else None,
                "query": args.query,
                "memories": [_serialize_chunk(chunk) for chunk in limited],
            },
            indent=2,
        )
    )


async def _cmd_search(args: argparse.Namespace) -> None:
    engine = _build_engine()
    chunks = _filtered_chunks(
        engine,
        scope=args.scope,
        scope_id=args.scope_id,
        query=args.query,
    )
    chunks = _sort_chunks(chunks, args.sort_by)
    limited = chunks[: args.limit] if args.limit > 0 else chunks
    print(
        json.dumps(
            {
                "query": args.query,
                "total": len(chunks),
                "showing": len(limited),
                "scope": args.scope,
                "scope_id": args.scope_id if args.scope != "global" else None,
                "results": [_serialize_chunk(chunk) for chunk in limited],
            },
            indent=2,
        )
    )


async def _cmd_export(args: argparse.Namespace) -> None:
    engine = _build_engine()
    chunks = _filtered_chunks(
        engine,
        scope=args.scope,
        scope_id=args.scope_id,
        query=args.query,
    )
    chunks = _sort_chunks(chunks, args.sort_by)
    limited = chunks[: args.limit] if args.limit > 0 else chunks
    serialized = [_serialize_chunk(chunk) for chunk in limited]

    if args.format == "jsonl":
        output = "\n".join(json.dumps(item) for item in serialized)
    else:
        output = json.dumps(
            {
                "scope": args.scope,
                "scope_id": args.scope_id if args.scope != "global" else None,
                "query": args.query,
                "total": len(serialized),
                "memories": serialized,
            },
            indent=2,
        )
    if args.output:
        text = output if output.endswith("\n") else f"{output}\n"
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(output)


async def _cmd_purge(args: argparse.Namespace) -> None:
    engine = _build_engine()
    chunks = _filtered_chunks(
        engine,
        scope=args.scope,
        scope_id=args.scope_id,
        query=args.query,
        older_than_days=args.older_than_days,
    )
    chunks = _sort_chunks(chunks, "created_at")
    limited = chunks[: args.limit] if args.limit > 0 else chunks
    ids = [chunk.id for chunk in limited]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "deleted": 0,
                    "matched": len(ids),
                    "dry_run": True,
                    "scope": args.scope,
                    "scope_id": args.scope_id if args.scope != "global" else None,
                    "query": args.query,
                    "older_than_days": args.older_than_days,
                    "ids": ids,
                },
                indent=2,
            )
        )
        return

    if not args.yes:
        print(
            json.dumps(
                {
                    "deleted": 0,
                    "matched": len(ids),
                    "dry_run": False,
                    "reason": "Refusing purge without --yes confirmation.",
                },
                indent=2,
            )
        )
        return

    deleted = 0
    for chunk_id in ids:
        if engine.store.delete(chunk_id):
            deleted += 1
            node = engine.spreading_activation.get_node(chunk_id)
            if node:
                engine.spreading_activation.remove_node(chunk_id)

    print(
        json.dumps(
            {
                "deleted": deleted,
                "matched": len(ids),
                "scope": args.scope,
                "scope_id": args.scope_id if args.scope != "global" else None,
                "query": args.query,
                "older_than_days": args.older_than_days,
            },
            indent=2,
        )
    )


async def _cmd_doctor(args: argparse.Namespace) -> None:
    doctor_env = dict(os.environ)
    doctor_env["MNEMOS_DOCTOR_CHUNK_THRESHOLD"] = str(args.chunk_threshold)
    doctor_env["MNEMOS_DOCTOR_LATENCY_P95_THRESHOLD_MS"] = str(args.latency_p95_threshold_ms)
    if args.observed_p95_ms is not None:
        doctor_env["MNEMOS_DOCTOR_OBSERVED_P95_MS"] = str(args.observed_p95_ms)
    report = run_health_checks(env=doctor_env)
    log_event(
        "mnemos.health_check",
        status=report["status"],
        profile=report["profile"],
        summary=report["summary"],
    )
    print(json.dumps(report, indent=2))


async def _cmd_profile(args: argparse.Namespace) -> None:
    env = _build_profile_env(
        profile=args.profile,
        llm_provider=args.llm_provider,
        embedding_provider=args.embedding_provider or None,
        model=args.model or None,
        sqlite_path=args.sqlite_path,
    )
    rendered = _render_profile_env(args.profile, env, args.format)
    print(rendered)
    if args.write:
        text = rendered if rendered.endswith("\n") else f"{rendered}\n"
        with open(args.write, "w", encoding="utf-8") as handle:
            handle.write(text)


async def _cmd_migrate_store(args: argparse.Namespace) -> None:
    if args.source_store not in MIGRATION_SOURCE_STORE_CHOICES:
        raise ValueError(f"Unsupported source store: {args.source_store!r}")
    if args.target_store not in MIGRATION_TARGET_CHOICES:
        raise ValueError(f"Unsupported target store: {args.target_store!r}")
    if not args.source_sqlite_path or not args.target_sqlite_path:
        raise ValueError(
            "SQLite migration requires both --source-sqlite-path and --target-sqlite-path."
        )
    if Path(args.source_sqlite_path).resolve() == Path(args.target_sqlite_path).resolve():
        raise ValueError("Source and target SQLite paths must differ.")

    source_store = _build_store_for_migration(
        store_type=args.source_store,
        sqlite_path=args.source_sqlite_path,
    )
    target_store = _build_store_for_migration(
        store_type=args.target_store,
        sqlite_path=args.target_sqlite_path,
    )

    try:
        summary = _migrate_chunks(
            source_store=source_store,
            target_store=target_store,
            dry_run=args.dry_run,
        )
    finally:
        _close_store_quietly(source_store)
        _close_store_quietly(target_store)

    summary["source_store"] = args.source_store
    summary["target_store"] = args.target_store
    print(json.dumps(summary, indent=2))


async def _cmd_antigravity(args: argparse.Namespace) -> None:
    policy = _build_antigravity_artifact(args.host, args.target)
    if args.format == "json":
        rendered = json.dumps(
            {"host": args.host, "target": args.target, "artifact": policy},
            indent=2,
        )
    else:
        rendered = policy
    print(rendered)
    if args.write:
        text = rendered if rendered.endswith("\n") else f"{rendered}\n"
        with open(args.write, "w", encoding="utf-8") as handle:
            handle.write(text)


def _read_hook_payload(raw_payload_arg: str) -> dict[str, Any]:
    if raw_payload_arg:
        return parse_hook_payload(raw_payload_arg)
    if sys.stdin is None:
        return {}
    try:
        if sys.stdin.isatty():
            return {}
        return parse_hook_payload(sys.stdin.read())
    except OSError:
        return {}


async def _cmd_autostore_hook(args: argparse.Namespace) -> None:
    payload = _read_hook_payload(args.payload)
    decision = decide_autostore(
        event=args.event,
        payload=payload,
        default_scope=args.scope,
        default_scope_id=(args.scope_id or None),
        max_chars=args.max_chars,
    )

    if not decision.should_store or decision.interaction is None:
        print(
            json.dumps(
                {
                    "stored": False,
                    "reason": decision.reason,
                    "event": args.event,
                    "scope": decision.scope,
                    "scope_id": decision.scope_id,
                },
                indent=2,
            )
        )
        return

    if args.dry_run:
        print(
            json.dumps(
                {
                    "stored": False,
                    "reason": f"Dry run: {decision.reason}",
                    "event": args.event,
                    "scope": decision.scope,
                    "scope_id": decision.scope_id,
                    "content_preview": decision.interaction.content[:200],
                },
                indent=2,
            )
        )
        return

    engine = _build_hook_engine()
    result = await engine.process(
        decision.interaction,
        scope=decision.scope,
        scope_id=decision.scope_id,
    )
    print(
        json.dumps(
            {
                "stored": result.stored,
                "reason": result.reason,
                "event": args.event,
                "scope": decision.scope,
                "scope_id": decision.scope_id,
                "chunk_id": result.chunk.id if result.chunk else None,
            },
            indent=2,
        )
    )


def _cmd_ui(args: argparse.Namespace) -> None:
    from .control_plane import ControlPlaneService
    from .ui_server import run_ui_server

    service = ControlPlaneService(
        cwd=Path.cwd(),
        env=os.environ,
        global_config_path=(args.config_path or None),
    )
    run_ui_server(
        service=service,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        description="Mnemos memory system CLI — shell-friendly commands for hooks and automation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # store
    sp_store = subparsers.add_parser("store", help="Store a memory through the full pipeline")
    sp_store.add_argument("content", help="Text content to memorize")
    sp_store.add_argument("--role", default="user", help="Speaker role (default: user)")
    sp_store.add_argument(
        "--scope",
        choices=VALID_SCOPES,
        default="project",
        help="Memory scope (default: project).",
    )
    sp_store.add_argument(
        "--scope-id",
        default="default",
        help="Scope identifier (ignored for global scope).",
    )

    # retrieve
    sp_retrieve = subparsers.add_parser("retrieve", help="Retrieve memories by query")
    sp_retrieve.add_argument("query", help="Search query")
    sp_retrieve.add_argument("--top-k", type=int, default=5, help="Max results (default: 5)")
    sp_retrieve.add_argument(
        "--current-scope",
        choices=VALID_SCOPES,
        default="project",
        help="Current runtime scope used for scoped retrieval (default: project).",
    )
    sp_retrieve.add_argument(
        "--scope-id",
        default="default",
        help="Current scope identifier used to filter project/workspace memories.",
    )
    sp_retrieve.add_argument(
        "--allowed-scopes",
        default="project,workspace,global",
        help="Comma-separated scopes to include (default: project,workspace,global).",
    )
    sp_retrieve.add_argument(
        "--no-reconsolidate",
        action="store_true",
        help="Disable post-retrieval background reconsolidation.",
    )

    # consolidate
    subparsers.add_parser("consolidate", help="Trigger sleep consolidation")

    sp_feedback = subparsers.add_parser(
        "feedback",
        help="Record feedback about whether a retrieval was useful, wrong, or missed.",
    )
    sp_feedback.add_argument("event_type", choices=("helpful", "not_helpful", "missed_memory"))
    sp_feedback.add_argument("--query", required=True, help="Query or task the feedback refers to.")
    sp_feedback.add_argument("--scope", choices=VALID_SCOPES, default="project")
    sp_feedback.add_argument("--scope-id", default="default")
    sp_feedback.add_argument(
        "--chunk-id",
        dest="chunk_ids",
        action="append",
        default=[],
        help="Retrieved chunk ID associated with this feedback event. Repeatable.",
    )
    sp_feedback.add_argument("--notes", default="", help="Optional explanation or annotation.")

    sp_feedback_list = subparsers.add_parser(
        "feedback-list",
        help="List recorded retrieval feedback events for maintainer review.",
    )
    sp_feedback_list.add_argument("--event-type", choices=FEEDBACK_EVENT_CHOICES, default=None)
    sp_feedback_list.add_argument("--scope", choices=AUDIT_SCOPES, default="all")
    sp_feedback_list.add_argument("--scope-id", default="default")
    sp_feedback_list.add_argument("--limit", type=int, default=50)

    sp_feedback_export = subparsers.add_parser(
        "feedback-export",
        help="Export recorded retrieval feedback events for offline analysis.",
    )
    sp_feedback_export.add_argument("--event-type", choices=FEEDBACK_EVENT_CHOICES, default=None)
    sp_feedback_export.add_argument("--scope", choices=AUDIT_SCOPES, default="all")
    sp_feedback_export.add_argument("--scope-id", default="default")
    sp_feedback_export.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    sp_feedback_export.add_argument("--format", choices=("json", "jsonl"), default="json")
    sp_feedback_export.add_argument("--output", default="", help="Optional output path.")

    # stats
    subparsers.add_parser("stats", help="Show system statistics")

    sp_inspect = subparsers.add_parser("inspect", help="Inspect one stored memory chunk")
    sp_inspect.add_argument("chunk_id", help="Memory chunk ID to inspect")
    sp_inspect.add_argument(
        "--query", default="", help="Optional query to explain why this memory would be retrieved."
    )
    sp_inspect.add_argument("--current-scope", choices=VALID_SCOPES, default="project")
    sp_inspect.add_argument("--scope-id", default="default")
    sp_inspect.add_argument("--allowed-scopes", default="project,workspace,global")

    sp_list = subparsers.add_parser(
        "list", help="List stored memories with optional scope/query filters"
    )
    sp_list.add_argument("--scope", choices=AUDIT_SCOPES, default="all")
    sp_list.add_argument("--scope-id", default="default")
    sp_list.add_argument("--query", default="", help="Optional case-insensitive substring filter.")
    sp_list.add_argument(
        "--sort-by",
        choices=("created_at", "updated_at", "salience", "access_count"),
        default="created_at",
    )
    sp_list.add_argument("--limit", type=int, default=50)

    sp_search = subparsers.add_parser(
        "search", help="Search memory content/metadata with scope filters."
    )
    sp_search.add_argument("query")
    sp_search.add_argument("--scope", choices=AUDIT_SCOPES, default="all")
    sp_search.add_argument("--scope-id", default="default")
    sp_search.add_argument(
        "--sort-by",
        choices=("created_at", "updated_at", "salience", "access_count"),
        default="created_at",
    )
    sp_search.add_argument("--limit", type=int, default=50)

    sp_export = subparsers.add_parser(
        "export", help="Export stored memories for audit/backup with optional filters."
    )
    sp_export.add_argument("--scope", choices=AUDIT_SCOPES, default="all")
    sp_export.add_argument("--scope-id", default="default")
    sp_export.add_argument(
        "--query", default="", help="Optional case-insensitive substring filter."
    )
    sp_export.add_argument(
        "--sort-by",
        choices=("created_at", "updated_at", "salience", "access_count"),
        default="created_at",
    )
    sp_export.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    sp_export.add_argument("--format", choices=("json", "jsonl"), default="json")
    sp_export.add_argument("--output", default="", help="Optional output path.")

    sp_purge = subparsers.add_parser(
        "purge", help="Delete memories matching filters (requires --yes unless --dry-run)."
    )
    sp_purge.add_argument("--scope", choices=AUDIT_SCOPES, default="all")
    sp_purge.add_argument("--scope-id", default="default")
    sp_purge.add_argument("--query", default="", help="Optional case-insensitive substring filter.")
    sp_purge.add_argument(
        "--older-than-days",
        type=int,
        default=0,
        help="Only purge memories older than this many days (0 disables age filter).",
    )
    sp_purge.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    sp_purge.add_argument("--dry-run", action="store_true")
    sp_purge.add_argument("--yes", action="store_true", help="Confirm destructive purge.")

    sp_doctor = subparsers.add_parser("doctor", help="Run readiness and dependency checks")
    sp_doctor.add_argument(
        "--chunk-threshold",
        type=int,
        default=5000,
        help="Flag large SQLite datasets once chunk count reaches this threshold.",
    )
    sp_doctor.add_argument(
        "--latency-p95-threshold-ms",
        type=float,
        default=250.0,
        help="Flag slow retrieval once observed p95 latency exceeds this threshold.",
    )
    sp_doctor.add_argument(
        "--observed-p95-ms",
        type=float,
        default=None,
        help="Optional observed p95 retrieval latency to evaluate against threshold.",
    )

    sp_profile = subparsers.add_parser(
        "profile",
        help="Generate ready-to-use env configuration for the default local SQLite setup.",
    )
    sp_profile.add_argument("profile", choices=PROFILE_CHOICES)
    sp_profile.add_argument(
        "--format",
        choices=("dotenv", "json", "bash", "powershell"),
        default="dotenv",
        help="Output format for generated profile config.",
    )
    sp_profile.add_argument(
        "--write",
        default="",
        help="Optional file path to write generated config output.",
    )
    sp_profile.add_argument(
        "--llm-provider",
        default="openclaw",
        help="LLM provider to encode in profile (openclaw/openai/ollama/mock).",
    )
    sp_profile.add_argument(
        "--embedding-provider",
        default="",
        help="Optional explicit embedding provider; defaults to inferred from LLM provider.",
    )
    sp_profile.add_argument(
        "--model", default="", help="Optional model value for MNEMOS_LLM_MODEL."
    )
    sp_profile.add_argument(
        "--sqlite-path",
        default=".mnemos/memory.db",
        help="SQLite path used by the default profile.",
    )

    sp_migrate = subparsers.add_parser(
        "migrate-store",
        help="Copy memories between SQLite databases for upgrades or path moves.",
    )
    sp_migrate.add_argument("--source-store", required=True, choices=MIGRATION_SOURCE_STORE_CHOICES)
    sp_migrate.add_argument("--target-store", required=True, choices=MIGRATION_TARGET_CHOICES)
    sp_migrate.add_argument("--source-sqlite-path", default="")
    sp_migrate.add_argument("--target-sqlite-path", default="")
    sp_migrate.add_argument("--dry-run", action="store_true")

    sp_antigravity = subparsers.add_parser(
        "antigravity",
        help="Generate host autopilot policy text for automatic Mnemos tool use.",
    )
    sp_antigravity.add_argument("host", choices=ANTIGRAVITY_HOST_CHOICES)
    sp_antigravity.add_argument(
        "--target",
        choices=ANTIGRAVITY_TARGET_CHOICES,
        default="policy",
        help="Artifact to generate: generic policy text, Cursor rule, or Codex helpers.",
    )
    sp_antigravity.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for generated autopilot policy.",
    )
    sp_antigravity.add_argument(
        "--write",
        default="",
        help="Optional file path to write generated autopilot policy.",
    )

    sp_autostore = subparsers.add_parser(
        "autostore-hook",
        help="Ingest Claude Code hook payload from stdin and auto-store high-signal memory.",
    )
    sp_autostore.add_argument("event", choices=SUPPORTED_HOOK_EVENTS)
    sp_autostore.add_argument(
        "--payload",
        default="",
        help="Optional raw JSON payload (stdin is used when omitted).",
    )
    sp_autostore.add_argument(
        "--scope",
        choices=VALID_SCOPES,
        default="project",
        help="Scope to store auto-ingested memory under (default: project).",
    )
    sp_autostore.add_argument(
        "--scope-id",
        default="",
        help="Optional explicit scope id; defaults to payload cwd basename.",
    )
    sp_autostore.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Maximum stored content length for hook ingests.",
    )
    sp_autostore.add_argument(
        "--dry-run",
        action="store_true",
        help="Print decision without storing.",
    )

    sp_ui = subparsers.add_parser(
        "ui",
        help="Launch the local Mnemos onboarding and control-plane UI.",
    )
    sp_ui.add_argument("--host", default="127.0.0.1")
    sp_ui.add_argument("--port", type=int, default=8765)
    sp_ui.add_argument("--config-path", default="")
    sp_ui.add_argument("--no-browser", action="store_true")

    args = parser.parse_args()
    if args.command == "retrieve":
        args.reconsolidate = not args.no_reconsolidate

    if args.command == "store":
        asyncio.run(_cmd_store(args))
    elif args.command == "retrieve":
        asyncio.run(_cmd_retrieve(args))
    elif args.command == "consolidate":
        asyncio.run(_cmd_consolidate(args))
    elif args.command == "feedback":
        asyncio.run(_cmd_feedback(args))
    elif args.command == "feedback-list":
        asyncio.run(_cmd_feedback_list(args))
    elif args.command == "feedback-export":
        asyncio.run(_cmd_feedback_export(args))
    elif args.command == "stats":
        asyncio.run(_cmd_stats(args))
    elif args.command == "inspect":
        asyncio.run(_cmd_inspect(args))
    elif args.command == "list":
        asyncio.run(_cmd_list(args))
    elif args.command == "search":
        asyncio.run(_cmd_search(args))
    elif args.command == "export":
        asyncio.run(_cmd_export(args))
    elif args.command == "purge":
        asyncio.run(_cmd_purge(args))
    elif args.command == "doctor":
        asyncio.run(_cmd_doctor(args))
    elif args.command == "profile":
        asyncio.run(_cmd_profile(args))
    elif args.command == "migrate-store":
        asyncio.run(_cmd_migrate_store(args))
    elif args.command == "antigravity":
        asyncio.run(_cmd_antigravity(args))
    elif args.command == "autostore-hook":
        asyncio.run(_cmd_autostore_hook(args))
    elif args.command == "ui":
        _cmd_ui(args)


if __name__ == "__main__":
    main()
