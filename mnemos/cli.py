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
from typing import Any, Literal, cast

from .config import MemoryGovernanceConfig, MemorySafetyConfig, MnemosConfig, SurprisalConfig
from .engine import MnemosEngine
from .health import run_health_checks
from .hook_autostore import SUPPORTED_HOOK_EVENTS, decide_autostore, parse_hook_payload
from .observability import configure_logging, log_event
from .runtime import build_embedder_from_env, build_store_from_env
from .types import Interaction
from .utils.llm import MockLLMProvider, LLMProvider

PROFILE_CHOICES = ("starter", "local-performance", "scale")
VALID_SCOPES = ("project", "workspace", "global")
AUDIT_SCOPES = ("all", "project", "workspace", "global")
ANTIGRAVITY_HOST_CHOICES = ("cursor", "generic-mcp", "codex")
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
    qdrant_path: str,
    qdrant_url: str,
    qdrant_collection: str,
) -> dict[str, str]:
    if profile not in PROFILE_CHOICES:
        raise ValueError(f"Unsupported profile: {profile!r}")

    env: dict[str, str] = {
        "MNEMOS_LLM_PROVIDER": llm_provider,
        "MNEMOS_EMBEDDING_PROVIDER": embedding_provider or _infer_embedding_provider(llm_provider),
    }
    if model:
        env["MNEMOS_LLM_MODEL"] = model

    if profile == "starter":
        env["MNEMOS_STORE_TYPE"] = "sqlite"
        env["MNEMOS_SQLITE_PATH"] = sqlite_path
    elif profile == "local-performance":
        env["MNEMOS_STORE_TYPE"] = "qdrant"
        env["MNEMOS_QDRANT_PATH"] = qdrant_path
        env["MNEMOS_QDRANT_COLLECTION"] = qdrant_collection
    else:
        env["MNEMOS_STORE_TYPE"] = "qdrant"
        env["MNEMOS_QDRANT_URL"] = qdrant_url
        env["MNEMOS_QDRANT_COLLECTION"] = qdrant_collection

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
    if host not in ANTIGRAVITY_HOST_CHOICES:
        raise ValueError(f"Unsupported host: {host!r}")

    if host == "cursor":
        host_label = "Cursor"
    elif host == "codex":
        host_label = "Codex"
    else:
        host_label = "Generic MCP host"
    codex_note = (
        "0. Add this workflow to your repo-level `AGENTS.md` so Codex uses Mnemos via MCP consistently.\n"
        if host == "codex"
        else ""
    )
    return (
        f"Mnemos Antigravity Autopilot Policy ({host_label})\n\n"
        "Always use Mnemos memory tools automatically.\n"
        f"{codex_note}"
        "1. At the start of every new user task, call `mnemos_retrieve` using a focused query.\n"
        "2. Use `current_scope=project` and set `scope_id` to the current repository/workspace name.\n"
        "3. Include `allowed_scopes=project,global` unless the user asks for broader scope.\n"
        "4. During execution, call `mnemos_store` for durable facts only:\n"
        "   - stable user preferences\n"
        "   - project architecture decisions\n"
        "   - environment/tooling setup facts\n"
        "   - recurring bug patterns and fixes\n"
        "5. Before finishing substantial work, call `mnemos_consolidate`.\n"
        "6. Never store secrets, tokens, credentials, or one-off transient chatter.\n"
        "7. If retrieval returns nothing useful, continue normally and seed memory as facts emerge.\n"
    )


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
    import os

    provider_name = os.getenv("MNEMOS_LLM_PROVIDER", "mock").lower()
    llm: LLMProvider

    if provider_name == "ollama":
        from .utils.llm import OllamaProvider

        llm = OllamaProvider(
            base_url=os.getenv("MNEMOS_OLLAMA_URL", "http://localhost:11434"),
            model=os.getenv("MNEMOS_LLM_MODEL", "llama3"),
        )
    elif provider_name in ("openai", "openclaw"):
        from .utils.llm import OpenAIProvider

        if provider_name == "openclaw":
            api_key = os.getenv("MNEMOS_OPENCLAW_API_KEY", "") or os.getenv(
                "MNEMOS_OPENAI_API_KEY", ""
            )
            base_url = os.getenv("MNEMOS_OPENCLAW_URL", "") or os.getenv(
                "MNEMOS_OPENAI_URL", "https://api.openai.com/v1"
            )
            key_error = "Error: MNEMOS_OPENCLAW_API_KEY or MNEMOS_OPENAI_API_KEY required for openclaw provider"
        else:
            api_key = os.getenv("MNEMOS_OPENAI_API_KEY", "")
            base_url = os.getenv("MNEMOS_OPENAI_URL", "https://api.openai.com/v1")
            key_error = "Error: MNEMOS_OPENAI_API_KEY required for openai provider"

        if not api_key:
            print(key_error, file=sys.stderr)
            sys.exit(1)

        llm = OpenAIProvider(
            api_key=api_key,
            base_url=base_url,
            model=os.getenv("MNEMOS_LLM_MODEL", "gpt-4o-mini"),
        )
    else:
        llm = MockLLMProvider()

    threshold = float(os.getenv("MNEMOS_SURPRISAL_THRESHOLD", "0.3"))

    config = MnemosConfig(
        surprisal=SurprisalConfig(threshold=threshold),
        safety=MemorySafetyConfig(
            enabled=_env_bool("MNEMOS_MEMORY_SAFETY_ENABLED", True),
            secret_action=_memory_action_from_env("MNEMOS_MEMORY_SECRET_ACTION", "block"),
            pii_action=_memory_action_from_env("MNEMOS_MEMORY_PII_ACTION", "redact"),
        ),
        governance=MemoryGovernanceConfig(
            capture_mode=_capture_mode_from_env("MNEMOS_MEMORY_CAPTURE_MODE", "all"),
            retention_ttl_days=int(os.getenv("MNEMOS_MEMORY_RETENTION_TTL_DAYS", "0")),
            max_chunks_per_scope=int(os.getenv("MNEMOS_MEMORY_MAX_CHUNKS_PER_SCOPE", "0")),
        ),
        debug=os.getenv("MNEMOS_DEBUG", "").lower() in ("true", "1", "yes"),
    )

    engine = MnemosEngine(
        config=config,
        llm=llm,
        embedder=build_embedder_from_env(default_provider="simple"),
        store=build_store_from_env(default_store_type="sqlite"),
    )
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


async def _cmd_stats(args: argparse.Namespace) -> None:
    engine = _build_engine()
    stats = engine.get_stats()
    print(json.dumps(stats, indent=2, default=str))


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
    doctor_env["MNEMOS_DOCTOR_QDRANT_CHUNK_THRESHOLD"] = str(args.qdrant_chunk_threshold)
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
        qdrant_path=args.qdrant_path,
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.qdrant_collection,
    )
    rendered = _render_profile_env(args.profile, env, args.format)
    print(rendered)
    if args.write:
        text = rendered if rendered.endswith("\n") else f"{rendered}\n"
        with open(args.write, "w", encoding="utf-8") as handle:
            handle.write(text)


async def _cmd_antigravity(args: argparse.Namespace) -> None:
    policy = _build_antigravity_policy(args.host)
    if args.format == "json":
        rendered = json.dumps({"host": args.host, "policy": policy}, indent=2)
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

    engine = _build_engine()
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


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        prog="mnemos-cli",
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

    # stats
    subparsers.add_parser("stats", help="Show system statistics")

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

    sp_doctor = subparsers.add_parser("doctor", help="Run profile readiness and dependency checks")
    sp_doctor.add_argument(
        "--qdrant-chunk-threshold",
        type=int,
        default=5000,
        help="Recommend qdrant upgrade only once SQLite chunk count reaches this threshold.",
    )
    sp_doctor.add_argument(
        "--latency-p95-threshold-ms",
        type=float,
        default=250.0,
        help="Recommend qdrant upgrade once observed p95 retrieval latency exceeds this threshold.",
    )
    sp_doctor.add_argument(
        "--observed-p95-ms",
        type=float,
        default=None,
        help="Optional observed p95 retrieval latency to evaluate against threshold.",
    )

    sp_profile = subparsers.add_parser(
        "profile",
        help="Generate ready-to-use env configuration for starter/local-performance/scale profiles.",
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
        help="SQLite path used by starter profile.",
    )
    sp_profile.add_argument(
        "--qdrant-path",
        default=".mnemos/qdrant",
        help="Embedded qdrant path used by local-performance profile.",
    )
    sp_profile.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Remote qdrant URL used by scale profile.",
    )
    sp_profile.add_argument(
        "--qdrant-collection",
        default="mnemos_memory",
        help="Qdrant collection name used by qdrant profiles.",
    )

    sp_antigravity = subparsers.add_parser(
        "antigravity",
        help="Generate host autopilot policy text for automatic Mnemos tool use.",
    )
    sp_antigravity.add_argument("host", choices=ANTIGRAVITY_HOST_CHOICES)
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

    args = parser.parse_args()
    if args.command == "retrieve":
        args.reconsolidate = not args.no_reconsolidate

    if args.command == "store":
        asyncio.run(_cmd_store(args))
    elif args.command == "retrieve":
        asyncio.run(_cmd_retrieve(args))
    elif args.command == "consolidate":
        asyncio.run(_cmd_consolidate(args))
    elif args.command == "stats":
        asyncio.run(_cmd_stats(args))
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
    elif args.command == "antigravity":
        asyncio.run(_cmd_antigravity(args))
    elif args.command == "autostore-hook":
        asyncio.run(_cmd_autostore_hook(args))


if __name__ == "__main__":
    main()
