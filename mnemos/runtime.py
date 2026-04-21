"""
mnemos/runtime.py — Shared runtime settings and provider/store factories.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Literal, Mapping

from pydantic import ValidationError

from .config import (
    MemoryGovernanceConfig,
    MemorySafetyConfig,
    MnemosConfig,
    SleepConfig,
    SurprisalConfig,
)
from .settings import AppSettings, ResolvedSettings, _persistent_env_value, load_settings
from .utils.embeddings import (
    EmbeddingProvider,
    FastembedProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SimpleEmbeddingProvider,
)
from .utils.llm import LLMProvider, MockLLMProvider, OllamaProvider, OpenAIProvider
from .utils.storage import InMemoryStore, MemoryStore, SQLiteStore


def _infer_sqlite_embedding_dim(sqlite_path: str | None) -> int | None:
    if not sqlite_path:
        return None

    path = Path(sqlite_path)
    if not path.exists():
        return None

    try:
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return None

    try:
        row = conn.execute("SELECT value FROM mnemos_meta WHERE key = 'sqlite_vec_dim'").fetchone()
        if row and row[0] not in (None, ""):
            dim = int(str(row[0]))
            return dim if dim > 0 else None

        sample = conn.execute(
            "SELECT embedding FROM memory_chunks WHERE embedding IS NOT NULL LIMIT 1"
        ).fetchone()
        if sample and sample[0]:
            embedding = json.loads(sample[0])
            if isinstance(embedding, list) and embedding:
                return len(embedding)
    except (json.JSONDecodeError, sqlite3.Error, TypeError, ValueError):
        return None
    finally:
        conn.close()
    return None


def resolve_env_value(
    name: str,
    default: str | None = None,
    aliases: tuple[str, ...] = (),
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """
    Resolve an env value with optional backward-compatible aliases.

    Canonical variable takes precedence; aliases are checked in order.
    Empty-string values are treated as unset.
    """
    source = os.environ if env is None else env
    primary = source.get(name)
    if primary not in (None, ""):
        return primary

    for alias in aliases:
        value = source.get(alias)
        if value not in (None, ""):
            return value

    return default


def load_runtime_settings(
    *,
    default_store_type: Literal["memory", "sqlite"] = "memory",
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> ResolvedSettings:
    return load_settings(
        env=env,
        cwd=cwd,
        default_store_type=default_store_type,
    )


def _persistent_provider_api_key(provider: str) -> str | None:
    if provider == "openai":
        return _persistent_env_value("MNEMOS_OPENAI_API_KEY")
    if provider == "openclaw":
        return _persistent_env_value("MNEMOS_OPENCLAW_API_KEY") or _persistent_env_value(
            "MNEMOS_OPENAI_API_KEY"
        )
    if provider == "openrouter":
        return _persistent_env_value("MNEMOS_OPENROUTER_API_KEY")
    return None


def build_store_from_settings(settings: AppSettings) -> MemoryStore:
    store_type = settings.storage.type

    if store_type == "memory":
        return InMemoryStore()

    if store_type == "sqlite":
        return SQLiteStore(db_path=settings.storage.sqlite_path)

    raise ValueError(f"Unknown store type: {store_type!r}. Use 'memory' or 'sqlite'.")


def build_embedder_from_settings(settings: AppSettings) -> EmbeddingProvider:
    provider = settings.embedding.provider or "fastembed"

    if provider == "fastembed":
        return FastembedProvider(
            model=settings.embedding.model or "BAAI/bge-small-en-v1.5",
        )

    if provider == "simple":
        dim = settings.embedding.dim
        if settings.storage.type == "sqlite":
            inferred_dim = _infer_sqlite_embedding_dim(settings.storage.sqlite_path)
            if inferred_dim is not None:
                dim = inferred_dim
        return SimpleEmbeddingProvider(dim=dim)

    if provider == "ollama":
        return OllamaEmbeddingProvider(
            model=settings.embedding.model or "nomic-embed-text",
            base_url=settings.base_url_for("ollama") or "http://localhost:11434",
        )

    if provider in {"openai", "openclaw", "openrouter"}:
        api_key = settings.api_key_for(provider)
        if not api_key:
            env_name = {
                "openai": "MNEMOS_OPENAI_API_KEY",
                "openclaw": "MNEMOS_OPENCLAW_API_KEY or MNEMOS_OPENAI_API_KEY",
                "openrouter": "MNEMOS_OPENROUTER_API_KEY",
            }[provider]
            raise ValueError(f"{env_name} must be set when using {provider} embedding provider")
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            api_key_fallback=_persistent_provider_api_key(provider),
            model=settings.embedding.model or "text-embedding-3-small",
            base_url=settings.base_url_for(provider) or "https://api.openai.com/v1",
        )

    raise ValueError(
        f"Unknown embedding provider: {provider!r}. "
        "Use 'fastembed', 'simple', 'ollama', 'openai', 'openclaw', or 'openrouter'."
    )


def build_llm_from_settings(settings: AppSettings) -> LLMProvider:
    provider = settings.llm.provider

    if provider == "mock":
        return MockLLMProvider()

    if provider == "ollama":
        return OllamaProvider(
            base_url=settings.base_url_for("ollama") or "http://localhost:11434",
            model=settings.llm.model or "llama3",
        )

    if provider in {"openai", "openclaw", "openrouter"}:
        api_key = settings.api_key_for(provider)
        if not api_key:
            env_name = {
                "openai": "MNEMOS_OPENAI_API_KEY",
                "openclaw": "MNEMOS_OPENCLAW_API_KEY or MNEMOS_OPENAI_API_KEY",
                "openrouter": "MNEMOS_OPENROUTER_API_KEY",
            }[provider]
            raise ValueError(f"{env_name} must be set when using {provider} provider")
        return OpenAIProvider(
            api_key=api_key,
            api_key_fallback=_persistent_provider_api_key(provider),
            base_url=settings.base_url_for(provider) or "https://api.openai.com/v1",
            model=settings.llm.model or "gpt-4o-mini",
        )

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        "Use 'mock', 'ollama', 'openai', 'openclaw', or 'openrouter'."
    )


def build_mnemos_config_from_settings(settings: AppSettings) -> MnemosConfig:
    return MnemosConfig(
        surprisal=SurprisalConfig(threshold=settings.runtime.surprisal_threshold),
        sleep=SleepConfig(
            recall_gated_plasticity_enabled=settings.sleep.recall_gated_plasticity_enabled,
            recall_min_supporting_episodes=settings.sleep.recall_min_supporting_episodes,
            recall_similarity_threshold=settings.sleep.recall_similarity_threshold,
        ),
        safety=MemorySafetyConfig.model_validate(settings.safety.model_dump(mode="python")),
        governance=MemoryGovernanceConfig.model_validate(
            settings.governance.model_dump(mode="python")
        ),
        debug=settings.runtime.debug,
    )


def build_store_from_env(
    default_store_type: Literal["memory", "sqlite"] = "memory",
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> MemoryStore:
    resolved = load_runtime_settings(
        default_store_type=default_store_type,
        env=env,
        cwd=cwd,
    )
    return build_store_from_settings(resolved.settings)


def build_embedder_from_env(
    default_provider: Literal[
        "fastembed", "simple", "ollama", "openai", "openclaw", "openrouter"
    ] = "fastembed",
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> EmbeddingProvider:
    try:
        resolved = load_runtime_settings(default_store_type="sqlite", env=env, cwd=cwd)
    except ValidationError as exc:
        message = str(exc)
        if "embedding.provider" in message:
            raise ValueError(
                "Unknown embedding provider. Use 'fastembed', 'simple', 'ollama', 'openai', "
                "'openclaw', or 'openrouter'."
            ) from exc
        raise
    settings = resolved.settings
    explicit_embedding_provider = resolve_env_value(
        "MNEMOS_EMBEDDING_PROVIDER",
        env=env,
    )
    if (
        explicit_embedding_provider is None
        and default_provider
        and settings.llm.provider == "mock"
        and not resolved.global_config_path.exists()
        and resolved.project_config_path is None
    ):
        settings.embedding.provider = default_provider
    return build_embedder_from_settings(settings)


def build_llm_from_env(
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> LLMProvider:
    resolved = load_runtime_settings(default_store_type="sqlite", env=env, cwd=cwd)
    return build_llm_from_settings(resolved.settings)


def build_mnemos_config_from_env(
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    default_store_type: Literal["memory", "sqlite"] = "sqlite",
) -> MnemosConfig:
    resolved = load_runtime_settings(
        default_store_type=default_store_type,
        env=env,
        cwd=cwd,
    )
    return build_mnemos_config_from_settings(resolved.settings)
