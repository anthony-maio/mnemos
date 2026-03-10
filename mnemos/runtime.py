"""
mnemos/runtime.py — Shared runtime settings and provider/store factories.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Mapping

from pydantic import ValidationError

from .config import MemoryGovernanceConfig, MemorySafetyConfig, MnemosConfig, SurprisalConfig
from .settings import AppSettings, ResolvedSettings, load_settings
from .utils.embeddings import (
    EmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SimpleEmbeddingProvider,
)
from .utils.llm import LLMProvider, MockLLMProvider, OllamaProvider, OpenAIProvider
from .utils.storage import InMemoryStore, MemoryStore, QdrantStore, SQLiteStore


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
    default_store_type: Literal["memory", "sqlite", "qdrant"] = "memory",
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> ResolvedSettings:
    return load_settings(
        env=env,
        cwd=cwd,
        default_store_type=default_store_type,
    )


def build_store_from_settings(settings: AppSettings) -> MemoryStore:
    store_type = settings.storage.type

    if store_type == "memory":
        return InMemoryStore()

    if store_type == "sqlite":
        return SQLiteStore(db_path=settings.storage.sqlite_path)

    if store_type == "qdrant":
        return QdrantStore(
            url=settings.storage.qdrant_url,
            api_key=settings.api_key_for("qdrant"),
            path=settings.storage.qdrant_path,
            collection_name=settings.storage.qdrant_collection,
            vector_size=settings.storage.qdrant_vector_size,
        )

    raise ValueError(f"Unknown store type: {store_type!r}. Use 'memory', 'sqlite', or 'qdrant'.")


def build_embedder_from_settings(settings: AppSettings) -> EmbeddingProvider:
    provider = settings.embedding.provider or "simple"

    if provider == "simple":
        return SimpleEmbeddingProvider(dim=settings.embedding.dim)

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
            model=settings.embedding.model or "text-embedding-3-small",
            base_url=settings.base_url_for(provider) or "https://api.openai.com/v1",
        )

    raise ValueError(
        f"Unknown embedding provider: {provider!r}. "
        "Use 'simple', 'ollama', 'openai', 'openclaw', or 'openrouter'."
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
        safety=MemorySafetyConfig.model_validate(settings.safety.model_dump(mode="python")),
        governance=MemoryGovernanceConfig.model_validate(
            settings.governance.model_dump(mode="python")
        ),
        debug=settings.runtime.debug,
    )


def build_store_from_env(
    default_store_type: Literal["memory", "sqlite", "qdrant"] = "memory",
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
    default_provider: Literal["simple", "ollama", "openai", "openclaw", "openrouter"] = "simple",
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
                "Unknown embedding provider. Use 'simple', 'ollama', 'openai', "
                "'openclaw', or 'openrouter'."
            ) from exc
        raise
    settings = resolved.settings
    if settings.embedding.provider is None and default_provider and settings.llm.provider == "mock":
        settings.embedding.provider = default_provider  # pragma: no cover
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
    default_store_type: Literal["memory", "sqlite", "qdrant"] = "sqlite",
) -> MnemosConfig:
    resolved = load_runtime_settings(
        default_store_type=default_store_type,
        env=env,
        cwd=cwd,
    )
    return build_mnemos_config_from_settings(resolved.settings)
