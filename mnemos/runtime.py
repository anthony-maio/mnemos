"""
mnemos/runtime.py — Shared runtime env parsing for CLI/MCP startup.
"""

from __future__ import annotations

import os

from .utils.embeddings import (
    EmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SimpleEmbeddingProvider,
)
from .utils.storage import InMemoryStore, MemoryStore, QdrantStore, SQLiteStore


def resolve_env_value(
    name: str,
    default: str | None = None,
    aliases: tuple[str, ...] = (),
) -> str | None:
    """
    Resolve an env value with optional backward-compatible aliases.

    Canonical variable takes precedence; aliases are checked in order.
    Empty-string values are treated as unset.
    """
    primary = os.getenv(name)
    if primary not in (None, ""):
        return primary

    for alias in aliases:
        value = os.getenv(alias)
        if value not in (None, ""):
            return value

    return default


def build_store_from_env(default_store_type: str = "memory") -> MemoryStore:
    """
    Build storage backend from env vars.

    Supported:
    - `MNEMOS_STORE_TYPE` (alias: `MNEMOS_STORAGE`)
    - `MNEMOS_SQLITE_PATH` (alias: `MNEMOS_DB_PATH`)
    - `MNEMOS_QDRANT_*` (for `qdrant` store type)
    """
    store_type = (
        resolve_env_value(
            "MNEMOS_STORE_TYPE",
            default=default_store_type,
            aliases=("MNEMOS_STORAGE",),
        )
        or default_store_type
    ).lower()

    if store_type == "memory":
        return InMemoryStore()

    if store_type == "sqlite":
        db_path = resolve_env_value(
            "MNEMOS_SQLITE_PATH",
            default="mnemos_memory.db",
            aliases=("MNEMOS_DB_PATH",),
        )
        return SQLiteStore(db_path=db_path or "mnemos_memory.db")

    if store_type == "qdrant":
        url = resolve_env_value("MNEMOS_QDRANT_URL", default="http://localhost:6333")
        api_key = resolve_env_value("MNEMOS_QDRANT_API_KEY", default=None)
        path = resolve_env_value("MNEMOS_QDRANT_PATH", default=None)
        collection_name = (
            resolve_env_value("MNEMOS_QDRANT_COLLECTION", default="mnemos_memory")
            or "mnemos_memory"
        )
        vector_size_text = resolve_env_value(
            "MNEMOS_QDRANT_VECTOR_SIZE",
            default=None,
            aliases=("MNEMOS_EMBEDDING_DIM",),
        )
        vector_size: int | None = None
        if vector_size_text is not None and vector_size_text != "":
            vector_size = int(vector_size_text)

        return QdrantStore(
            url=url,
            api_key=api_key,
            path=path,
            collection_name=collection_name,
            vector_size=vector_size,
        )

    raise ValueError(f"Unknown store type: {store_type!r}. Use 'memory', 'sqlite', or 'qdrant'.")


def build_embedder_from_env(default_provider: str = "simple") -> EmbeddingProvider:
    """
    Build embedding provider from env vars.

    Supported:
    - `MNEMOS_EMBEDDING_PROVIDER`: `simple`, `ollama`, `openai`, `openclaw`
    - `MNEMOS_EMBEDDING_MODEL` (provider model name)
    - provider-specific auth/url vars
    """
    explicit_provider = resolve_env_value("MNEMOS_EMBEDDING_PROVIDER", default=None)
    if explicit_provider is not None:
        provider = explicit_provider.lower()
    else:
        llm_provider = (resolve_env_value("MNEMOS_LLM_PROVIDER", default="") or "").lower()
        if llm_provider in {"ollama", "openai", "openclaw"}:
            provider = llm_provider
        else:
            provider = default_provider.lower()

    if provider == "simple":
        dim_text = resolve_env_value("MNEMOS_EMBEDDING_DIM", default="384") or "384"
        return SimpleEmbeddingProvider(dim=int(dim_text))

    if provider == "ollama":
        model = (
            resolve_env_value(
                "MNEMOS_EMBEDDING_MODEL",
                default="nomic-embed-text",
            )
            or "nomic-embed-text"
        )
        base_url = (
            resolve_env_value("MNEMOS_OLLAMA_URL", default="http://localhost:11434")
            or "http://localhost:11434"
        )
        return OllamaEmbeddingProvider(
            model=model,
            base_url=base_url,
        )

    if provider == "openai":
        api_key = resolve_env_value("MNEMOS_OPENAI_API_KEY", default="")
        if not api_key:
            raise ValueError(
                "MNEMOS_OPENAI_API_KEY must be set when using openai embedding provider"
            )
        model = (
            resolve_env_value(
                "MNEMOS_EMBEDDING_MODEL",
                default="text-embedding-3-small",
            )
            or "text-embedding-3-small"
        )
        base_url = (
            resolve_env_value("MNEMOS_OPENAI_URL", default="https://api.openai.com/v1")
            or "https://api.openai.com/v1"
        )
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
        )

    if provider == "openclaw":
        api_key = resolve_env_value("MNEMOS_OPENCLAW_API_KEY", default="") or resolve_env_value(
            "MNEMOS_OPENAI_API_KEY", default=""
        )
        if not api_key:
            raise ValueError(
                "MNEMOS_OPENCLAW_API_KEY or MNEMOS_OPENAI_API_KEY must be set "
                "when using openclaw embedding provider"
            )
        model = (
            resolve_env_value(
                "MNEMOS_EMBEDDING_MODEL",
                default="text-embedding-3-small",
            )
            or "text-embedding-3-small"
        )
        base_url = (
            resolve_env_value("MNEMOS_OPENCLAW_URL", default="")
            or resolve_env_value("MNEMOS_OPENAI_URL", default="https://api.openai.com/v1")
            or "https://api.openai.com/v1"
        )
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
        )

    raise ValueError(
        f"Unknown embedding provider: {provider!r}. "
        "Use 'simple', 'ollama', 'openai', or 'openclaw'."
    )
