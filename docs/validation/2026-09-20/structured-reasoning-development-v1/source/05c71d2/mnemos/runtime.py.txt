"""
mnemos/runtime.py — Shared runtime settings and provider/store factories.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Literal, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import MnemosEngine

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
    PrefixEmbeddingProvider,
    SimpleEmbeddingProvider,
)
from .utils.llm import (
    LLMProvider,
    MockLLMProvider,
    OllamaProvider,
    OpenAIProvider,
    UnavailableLLMProvider,
)
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
        Path(settings.storage.sqlite_path).expanduser().resolve().parent.mkdir(
            parents=True, exist_ok=True
        )
        return SQLiteStore(db_path=settings.storage.sqlite_path)

    raise ValueError(f"Unknown store type: {store_type!r}. Use 'memory' or 'sqlite'.")


def build_embedder_from_settings(settings: AppSettings) -> EmbeddingProvider:
    embedder = _build_unprefixed_embedder_from_settings(settings)
    prefixes = _embedding_prefixes(settings)
    if any(prefixes.values()):
        return PrefixEmbeddingProvider(embedder, **prefixes)
    return embedder


def _embedding_prefixes(settings: AppSettings) -> dict[str, str]:
    return {
        "query_prefix": settings.embedding.query_prefix,
        "document_prefix": settings.embedding.document_prefix,
        "symmetric_prefix": settings.embedding.symmetric_prefix,
    }


def embedding_task_contract(
    *, query_prefix: str = "", document_prefix: str = "", symmetric_prefix: str = ""
) -> dict[str, str] | None:
    """Return the explicit task contract, omitting unchanged legacy behavior."""
    prefixes = {
        "query_prefix": query_prefix,
        "document_prefix": document_prefix,
        "symmetric_prefix": symmetric_prefix,
    }
    if not any(prefixes.values()):
        return None
    return {"version": "explicit-prefix-v1", **prefixes}


def _build_unprefixed_embedder_from_settings(settings: AppSettings) -> EmbeddingProvider:
    provider = settings.embedding.provider or "fastembed"

    if provider == "fastembed":
        return FastembedProvider(
            model=settings.embedding.model or "BAAI/bge-small-en-v1.5",
        )

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
            think=settings.providers.ollama.think,
            max_tokens=settings.providers.ollama.max_tokens,
            timeout=settings.providers.ollama.timeout,
            temperature=settings.providers.ollama.temperature,
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
        modules=settings.modules.model_copy(deep=True),
        retrieval=settings.retrieval.model_copy(deep=True),
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


def _mock_provider_is_explicit(
    settings: AppSettings,
    resolved: ResolvedSettings,
    env: Mapping[str, str] | None,
) -> bool:
    """Return whether a mock provider was selected by user configuration."""
    requested = resolve_env_value("MNEMOS_LLM_PROVIDER", env=env)
    if requested is not None:
        return requested.strip().lower() == "mock"
    return settings.llm.provider == "mock" and resolved.llm_provider_explicit


def embedding_identity(settings: AppSettings) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "provider": settings.embedding.provider,
        "model": settings.embedding.model,
        "revision": settings.embedding.revision,
        "dimension": settings.embedding.dim if settings.embedding.provider == "simple" else None,
        "normalization": "provider_default",
    }
    contract = embedding_task_contract(**_embedding_prefixes(settings))
    if contract is not None:
        identity["task_contract"] = contract
    return identity


def _configured_store(settings: AppSettings) -> MemoryStore:
    store = build_store_from_settings(settings)
    if isinstance(store, SQLiteStore):
        identity = embedding_identity(settings)
        try:
            store.bind_embedding_identity(identity)
        except Exception:
            store.close()
            raise
    return store


def build_configured_engine_from_settings(
    settings: AppSettings,
    *,
    allow_mock: bool = False,
) -> "MnemosEngine":
    """Build an engine from validated settings without implicit mock cognition.

    ``allow_mock`` is reserved for deterministic tests and explicitly labelled
    development profiles.  A missing provider receives an unavailable adapter,
    which preserves capture while making enrichment visibly pending.
    """
    from .engine import MnemosEngine

    if settings.llm.provider == "mock":
        if not allow_mock:
            raise ValueError(
                "Mock LLM cognition is disabled for configured runtimes; "
                "pass allow_mock=True only for tests or development."
            )
        llm: LLMProvider = MockLLMProvider()
    else:
        llm = build_llm_from_settings(settings)

    return MnemosEngine(
        config=build_mnemos_config_from_settings(settings),
        llm=llm,
        embedder=build_embedder_from_settings(settings),
        store=_configured_store(settings),
        owns_providers=True,
    )


def build_configured_engine(
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    allow_mock: bool = False,
    default_store_type: Literal["memory", "sqlite"] = "sqlite",
) -> "MnemosEngine":
    """Build the application engine using local-first, validated configuration.

    With no configured LLM, the engine receives :class:`UnavailableLLMProvider`
    instead of the legacy mock.  This keeps ingestion and persistence usable
    without allowing fabricated cognitive output.
    """
    resolved = load_runtime_settings(
        default_store_type=default_store_type,
        env=env,
        cwd=cwd,
    )
    settings = resolved.settings
    if settings.llm.provider == "mock" and not allow_mock:
        if _mock_provider_is_explicit(settings, resolved, env):
            raise ValueError(
                "Mock LLM cognition is disabled for configured runtimes; "
                "pass allow_mock=True only for tests or development."
            )
        from .engine import MnemosEngine

        return MnemosEngine(
            config=build_mnemos_config_from_settings(settings),
            llm=UnavailableLLMProvider(),
            embedder=build_embedder_from_settings(settings),
            store=_configured_store(settings),
            owns_providers=True,
        )
    return build_configured_engine_from_settings(settings, allow_mock=allow_mock)


# Name used by application entrypoints during the runtime migration.
build_engine_from_env = build_configured_engine


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
