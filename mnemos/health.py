"""
mnemos/health.py — Readiness checks for CLI/MCP production profiles.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _resolve_env_value(
    name: str,
    *,
    env: Mapping[str, str] | None,
    default: str | None = None,
    aliases: tuple[str, ...] = (),
) -> str | None:
    source = env if env is not None else os.environ
    primary = source.get(name)
    if primary not in (None, ""):
        return primary
    for alias in aliases:
        value = source.get(alias)
        if value not in (None, ""):
            return value
    return default


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _safe_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _safe_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _sqlite_chunk_count(db_path: Path) -> int | None:
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memory_chunks'"
            )
            table_exists = int(cursor.fetchone()[0]) > 0
            if not table_exists:
                return 0
            return int(conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0])
        finally:
            conn.close()
    except Exception:
        return None


def _sqlite_legacy_unscoped_chunk_count(db_path: Path) -> int | None:
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memory_chunks'"
            )
            table_exists = int(cursor.fetchone()[0]) > 0
            if not table_exists:
                return 0

            count = 0
            for (metadata_json,) in conn.execute("SELECT metadata FROM memory_chunks"):
                if metadata_json in (None, ""):
                    count += 1
                    continue
                try:
                    metadata = json.loads(metadata_json)
                except Exception:
                    count += 1
                    continue
                if not isinstance(metadata, dict) or "scope" not in metadata:
                    count += 1
            return count
        finally:
            conn.close()
    except Exception:
        return None


def _embedding_provider_from_env(env: Mapping[str, str] | None = None) -> str:
    explicit = _resolve_env_value("MNEMOS_EMBEDDING_PROVIDER", env=env, default=None)
    if explicit:
        return explicit.lower()

    llm_provider = (
        _resolve_env_value("MNEMOS_LLM_PROVIDER", env=env, default="mock") or "mock"
    ).lower()
    if llm_provider in {"ollama", "openai", "openclaw"}:
        return llm_provider
    return "simple"


def detect_profile(
    env: Mapping[str, str] | None = None,
    *,
    default_store_type: str = "sqlite",
) -> str:
    """
    Return onboarding profile inferred from runtime env.

    - starter: sqlite (plugin-first default)
    - local-performance: qdrant with local embedded path
    - scale: qdrant over URL
    """
    store_type = (
        _resolve_env_value(
            "MNEMOS_STORE_TYPE",
            env=env,
            default=default_store_type,
            aliases=("MNEMOS_STORAGE",),
        )
        or default_store_type
    ).lower()

    if store_type != "qdrant":
        return "starter"

    qdrant_path = _resolve_env_value("MNEMOS_QDRANT_PATH", env=env, default=None)
    if qdrant_path:
        return "local-performance"
    return "scale"


def run_health_checks(
    env: Mapping[str, str] | None = None,
    *,
    default_store_type: str = "sqlite",
) -> dict[str, Any]:
    """Evaluate readiness for current profile and provider/storage dependencies."""
    checks: list[dict[str, str]] = []

    def add_check(name: str, status: str, message: str) -> None:
        checks.append({"name": name, "status": status, "message": message})

    profile = detect_profile(env, default_store_type=default_store_type)
    store_type = (
        _resolve_env_value(
            "MNEMOS_STORE_TYPE",
            env=env,
            default=default_store_type,
            aliases=("MNEMOS_STORAGE",),
        )
        or default_store_type
    ).lower()
    llm_provider = (
        _resolve_env_value("MNEMOS_LLM_PROVIDER", env=env, default="mock") or "mock"
    ).lower()
    embedding_provider = _embedding_provider_from_env(env)
    sqlite_chunk_threshold = _safe_int(
        _resolve_env_value(
            "MNEMOS_DOCTOR_QDRANT_CHUNK_THRESHOLD",
            env=env,
            default="5000",
        ),
        default=5000,
    )
    latency_p95_threshold_ms = _safe_float(
        _resolve_env_value(
            "MNEMOS_DOCTOR_LATENCY_P95_THRESHOLD_MS",
            env=env,
            default="250",
        ),
        default=250.0,
    )
    observed_latency_p95_ms_text = _resolve_env_value(
        "MNEMOS_DOCTOR_OBSERVED_P95_MS",
        env=env,
        default=None,
    )
    observed_latency_p95_ms = (
        None
        if observed_latency_p95_ms_text in (None, "")
        else _safe_float(observed_latency_p95_ms_text, default=0.0)
    )
    sqlite_chunk_count: int | None = None
    legacy_unscoped_chunks: int | None = None

    if store_type == "memory":
        add_check(
            "store.memory",
            "warn",
            "InMemory store is non-persistent. Use sqlite or qdrant for production.",
        )
    elif store_type == "sqlite":
        sqlite_path = (
            _resolve_env_value(
                "MNEMOS_SQLITE_PATH",
                env=env,
                default="mnemos_memory.db",
                aliases=("MNEMOS_DB_PATH",),
            )
            or "mnemos_memory.db"
        )
        parent = Path(sqlite_path).expanduser().resolve().parent
        if parent.exists():
            add_check("store.sqlite", "pass", f"SQLite profile ready at {sqlite_path}.")
        else:
            add_check(
                "store.sqlite",
                "warn",
                f"SQLite parent directory does not exist yet: {parent}",
            )
        sqlite_db_path = Path(sqlite_path).expanduser().resolve()
        sqlite_chunk_count = _sqlite_chunk_count(sqlite_db_path)
        legacy_unscoped_chunks = _sqlite_legacy_unscoped_chunk_count(sqlite_db_path)
    elif store_type == "qdrant":
        if _module_available("qdrant_client"):
            add_check("dependency.qdrant_client", "pass", "qdrant-client dependency available.")
        else:
            add_check(
                "dependency.qdrant_client",
                "fail",
                "qdrant-client not installed. Install with `pip install 'mnemos-memory[qdrant]'`.",
            )

        qdrant_path = _resolve_env_value("MNEMOS_QDRANT_PATH", env=env, default=None)
        qdrant_url = _resolve_env_value(
            "MNEMOS_QDRANT_URL",
            env=env,
            default="http://localhost:6333",
        )
        if qdrant_path:
            add_check(
                "store.qdrant.embedded", "pass", f"Embedded Qdrant path configured: {qdrant_path}"
            )
        elif qdrant_url:
            add_check("store.qdrant.remote", "pass", f"Remote Qdrant URL configured: {qdrant_url}")
        else:
            add_check(
                "store.qdrant.remote", "fail", "MNEMOS_QDRANT_URL must be set for remote Qdrant."
            )
    else:
        add_check("store.type", "fail", f"Unsupported MNEMOS_STORE_TYPE: {store_type!r}")

    if llm_provider == "mock":
        add_check(
            "llm.mock",
            "warn",
            "Mock LLM provider is active. Cognitive modules run in degraded/mock mode.",
        )
    elif llm_provider == "ollama":
        ollama_url = _resolve_env_value(
            "MNEMOS_OLLAMA_URL", env=env, default="http://localhost:11434"
        )
        if ollama_url:
            add_check("llm.ollama", "pass", f"Ollama provider configured at {ollama_url}.")
        else:
            add_check("llm.ollama", "fail", "MNEMOS_OLLAMA_URL is required for ollama provider.")
    elif llm_provider == "openai":
        api_key = _resolve_env_value("MNEMOS_OPENAI_API_KEY", env=env, default="")
        if api_key:
            add_check("llm.openai", "pass", "OpenAI provider API key configured.")
        else:
            add_check(
                "llm.openai", "fail", "MNEMOS_OPENAI_API_KEY is required for openai provider."
            )
    elif llm_provider == "openclaw":
        api_key = _resolve_env_value(
            "MNEMOS_OPENCLAW_API_KEY", env=env, default=""
        ) or _resolve_env_value("MNEMOS_OPENAI_API_KEY", env=env, default="")
        if api_key:
            add_check("llm.openclaw", "pass", "OpenClaw provider API key configured.")
        else:
            add_check(
                "llm.openclaw",
                "fail",
                "MNEMOS_OPENCLAW_API_KEY (or MNEMOS_OPENAI_API_KEY) is required for openclaw provider.",
            )
    else:
        add_check("llm.provider", "fail", f"Unsupported MNEMOS_LLM_PROVIDER: {llm_provider!r}")

    if embedding_provider == "simple":
        add_check(
            "embedding.simple",
            "warn",
            "Simple embeddings are for development; use openclaw/openai/ollama for production retrieval quality.",
        )
    elif embedding_provider == "ollama":
        ollama_url = _resolve_env_value(
            "MNEMOS_OLLAMA_URL", env=env, default="http://localhost:11434"
        )
        if ollama_url:
            add_check(
                "embedding.ollama", "pass", f"Ollama embedding provider configured at {ollama_url}."
            )
        else:
            add_check(
                "embedding.ollama",
                "fail",
                "MNEMOS_OLLAMA_URL is required for ollama embedding provider.",
            )
    elif embedding_provider == "openai":
        api_key = _resolve_env_value("MNEMOS_OPENAI_API_KEY", env=env, default="")
        if api_key:
            add_check("embedding.openai", "pass", "OpenAI embedding provider API key configured.")
        else:
            add_check(
                "embedding.openai",
                "fail",
                "MNEMOS_OPENAI_API_KEY is required for openai embedding provider.",
            )
    elif embedding_provider == "openclaw":
        api_key = _resolve_env_value(
            "MNEMOS_OPENCLAW_API_KEY", env=env, default=""
        ) or _resolve_env_value("MNEMOS_OPENAI_API_KEY", env=env, default="")
        if api_key:
            add_check(
                "embedding.openclaw", "pass", "OpenClaw embedding provider API key configured."
            )
        else:
            add_check(
                "embedding.openclaw",
                "fail",
                "MNEMOS_OPENCLAW_API_KEY (or MNEMOS_OPENAI_API_KEY) is required for openclaw embeddings.",
            )
    else:
        add_check(
            "embedding.provider", "fail", f"Unsupported embedding provider: {embedding_provider!r}"
        )

    size_threshold_exceeded = (
        store_type == "sqlite"
        and sqlite_chunk_count is not None
        and sqlite_chunk_count >= sqlite_chunk_threshold
    )
    latency_threshold_exceeded = (
        observed_latency_p95_ms is not None and observed_latency_p95_ms >= latency_p95_threshold_ms
    )

    if store_type == "sqlite":
        if legacy_unscoped_chunks is None:
            add_check(
                "scope.isolation",
                "warn",
                "Unable to inspect SQLite store for legacy unscoped chunks.",
            )
        elif legacy_unscoped_chunks > 0:
            add_check(
                "scope.isolation",
                "warn",
                f"Found {legacy_unscoped_chunks} legacy unscoped chunks. Scope isolation is not fully ready.",
            )
        else:
            add_check(
                "scope.isolation",
                "pass",
                "All SQLite chunks include scope metadata for v1 isolation.",
            )

        if size_threshold_exceeded or latency_threshold_exceeded:
            add_check(
                "store.sqlite.scale_signal",
                "warn",
                "SQLite scale/latency threshold exceeded; consider local-performance profile.",
            )
        elif sqlite_chunk_count is not None:
            add_check(
                "store.sqlite.scale_signal",
                "pass",
                f"SQLite scale signal within threshold (chunks={sqlite_chunk_count}, threshold={sqlite_chunk_threshold}).",
            )
        else:
            add_check(
                "store.sqlite.scale_signal",
                "warn",
                "Unable to inspect SQLite scale signal from database.",
            )

    summary = {
        "pass": sum(1 for c in checks if c["status"] == "pass"),
        "warn": sum(1 for c in checks if c["status"] == "warn"),
        "fail": sum(1 for c in checks if c["status"] == "fail"),
    }

    if summary["fail"] > 0:
        status = "not_ready"
    elif summary["warn"] > 0:
        status = "degraded"
    else:
        status = "ready"

    recommendations: list[str] = []
    if store_type == "sqlite" and (size_threshold_exceeded or latency_threshold_exceeded):
        recommendations.append(
            "Upgrade path: set MNEMOS_STORE_TYPE=qdrant and MNEMOS_QDRANT_PATH=.mnemos_qdrant for local-performance profile."
        )
    if llm_provider == "mock":
        recommendations.append(
            "Set MNEMOS_LLM_PROVIDER=openclaw (or openai/ollama) to enable non-mock cognition."
        )
    if embedding_provider == "simple":
        recommendations.append(
            "Set MNEMOS_EMBEDDING_PROVIDER=openclaw/openai/ollama for production retrieval quality."
        )
    if legacy_unscoped_chunks is not None and legacy_unscoped_chunks > 0:
        recommendations.append(
            "Legacy unscoped chunks are still readable as global memory, but v1 scope isolation is degraded until you purge or rebuild them."
        )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "degraded_mode": status == "degraded",
        "profile": profile,
        "store_type": store_type,
        "llm_provider": llm_provider,
        "embedding_provider": embedding_provider,
        "upgrade_signals": {
            "sqlite_chunk_count": sqlite_chunk_count,
            "sqlite_chunk_threshold": sqlite_chunk_threshold,
            "observed_latency_p95_ms": observed_latency_p95_ms,
            "latency_p95_threshold_ms": latency_p95_threshold_ms,
            "threshold_exceeded": bool(size_threshold_exceeded or latency_threshold_exceeded),
        },
        "scope_isolation": {
            "legacy_unscoped_chunks": legacy_unscoped_chunks,
            "ready": not bool(legacy_unscoped_chunks),
        },
        "summary": summary,
        "checks": checks,
        "recommendations": recommendations,
    }
