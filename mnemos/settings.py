"""
mnemos/settings.py — Canonical config loading, precedence, and persistence.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import MemoryGovernanceConfig, MemorySafetyConfig

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib


def _fallback_user_config_dir(app_name: str) -> Path:
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / app_name
        return Path.home() / "AppData" / "Roaming" / app_name
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app_name
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / app_name
    return Path.home() / ".config" / app_name


def _user_config_dir(app_name: str) -> Path:
    try:
        from platformdirs import user_config_path
    except ImportError:  # pragma: no cover
        return _fallback_user_config_dir(app_name)
    return Path(user_config_path(app_name, appauthor=False, roaming=True))


def default_global_config_path() -> Path:
    return _user_config_dir("Mnemos") / "mnemos.toml"


def find_project_config_path(cwd: str | Path | None = None) -> Path | None:
    current = Path.cwd() if cwd is None else Path(cwd)
    current = current.resolve()
    for candidate_dir in (current, *current.parents):
        candidate = candidate_dir / ".mnemos" / "mnemos.toml"
        if candidate.exists():
            return candidate
    return None


def _read_toml_file(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        return {}
    return dict(data)


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _set_nested(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = target
    for key in path[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[path[-1]] = value


def _env_value(
    env: Mapping[str, str],
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> str | None:
    primary = env.get(name)
    if primary not in (None, ""):
        return primary
    for alias in aliases:
        value = env.get(alias)
        if value not in (None, ""):
            return value
    return None


def _env_bool(env: Mapping[str, str], name: str) -> bool | None:
    raw = env.get(name)
    if raw in (None, ""):
        return None
    assert raw is not None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(env: Mapping[str, str], name: str, *, aliases: tuple[str, ...] = ()) -> int | None:
    raw = _env_value(env, name, aliases=aliases)
    if raw in (None, ""):
        return None
    assert raw is not None
    return int(raw)


def _env_float(env: Mapping[str, str], name: str) -> float | None:
    raw = env.get(name)
    if raw in (None, ""):
        return None
    assert raw is not None
    return float(raw)


def _strip_global_only_sections(
    raw: dict[str, Any],
    *,
    warnings: list[str],
) -> dict[str, Any]:
    sanitized = dict(raw)
    if "providers" in sanitized:
        sanitized.pop("providers", None)
        warnings.append(
            "Project config attempted to define provider credentials/endpoints; "
            "the [providers] section is global-only and was ignored."
        )
    return sanitized


def _env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    llm_provider = _env_value(env, "MNEMOS_LLM_PROVIDER")
    if llm_provider is not None:
        _set_nested(overrides, ("llm", "provider"), llm_provider.lower())

    llm_model = _env_value(env, "MNEMOS_LLM_MODEL")
    if llm_model is not None:
        _set_nested(overrides, ("llm", "model"), llm_model)

    embedding_provider = _env_value(env, "MNEMOS_EMBEDDING_PROVIDER")
    if embedding_provider is not None:
        _set_nested(overrides, ("embedding", "provider"), embedding_provider.lower())

    embedding_model = _env_value(env, "MNEMOS_EMBEDDING_MODEL")
    if embedding_model is not None:
        _set_nested(overrides, ("embedding", "model"), embedding_model)

    embedding_dim = _env_int(env, "MNEMOS_EMBEDDING_DIM")
    if embedding_dim is not None:
        _set_nested(overrides, ("embedding", "dim"), embedding_dim)

    store_type = _env_value(env, "MNEMOS_STORE_TYPE", aliases=("MNEMOS_STORAGE",))
    if store_type is not None:
        _set_nested(overrides, ("storage", "type"), store_type.lower())

    sqlite_path = _env_value(env, "MNEMOS_SQLITE_PATH", aliases=("MNEMOS_DB_PATH",))
    if sqlite_path is not None:
        _set_nested(overrides, ("storage", "sqlite_path"), sqlite_path)

    qdrant_url = _env_value(env, "MNEMOS_QDRANT_URL")
    if qdrant_url is not None:
        _set_nested(overrides, ("storage", "qdrant_url"), qdrant_url)

    qdrant_path = _env_value(env, "MNEMOS_QDRANT_PATH")
    if qdrant_path is not None:
        _set_nested(overrides, ("storage", "qdrant_path"), qdrant_path)

    qdrant_collection = _env_value(env, "MNEMOS_QDRANT_COLLECTION")
    if qdrant_collection is not None:
        _set_nested(overrides, ("storage", "qdrant_collection"), qdrant_collection)

    qdrant_vector_size = _env_int(
        env,
        "MNEMOS_QDRANT_VECTOR_SIZE",
        aliases=("MNEMOS_EMBEDDING_DIM",),
    )
    if qdrant_vector_size is not None:
        _set_nested(overrides, ("storage", "qdrant_vector_size"), qdrant_vector_size)

    qdrant_api_key = _env_value(env, "MNEMOS_QDRANT_API_KEY")
    if qdrant_api_key is not None:
        _set_nested(overrides, ("providers", "qdrant", "api_key"), qdrant_api_key)

    ollama_url = _env_value(env, "MNEMOS_OLLAMA_URL")
    if ollama_url is not None:
        _set_nested(overrides, ("providers", "ollama", "base_url"), ollama_url)

    openai_api_key = _env_value(env, "MNEMOS_OPENAI_API_KEY")
    if openai_api_key is not None:
        _set_nested(overrides, ("providers", "openai", "api_key"), openai_api_key)

    openai_url = _env_value(env, "MNEMOS_OPENAI_URL")
    if openai_url is not None:
        _set_nested(overrides, ("providers", "openai", "base_url"), openai_url)

    openclaw_api_key = _env_value(env, "MNEMOS_OPENCLAW_API_KEY")
    if openclaw_api_key is not None:
        _set_nested(overrides, ("providers", "openclaw", "api_key"), openclaw_api_key)

    openclaw_url = _env_value(env, "MNEMOS_OPENCLAW_URL")
    if openclaw_url is not None:
        _set_nested(overrides, ("providers", "openclaw", "base_url"), openclaw_url)

    openrouter_api_key = _env_value(env, "MNEMOS_OPENROUTER_API_KEY")
    if openrouter_api_key is not None:
        _set_nested(overrides, ("providers", "openrouter", "api_key"), openrouter_api_key)

    openrouter_url = _env_value(env, "MNEMOS_OPENROUTER_URL")
    if openrouter_url is not None:
        _set_nested(overrides, ("providers", "openrouter", "base_url"), openrouter_url)

    surprisal_threshold = _env_float(env, "MNEMOS_SURPRISAL_THRESHOLD")
    if surprisal_threshold is not None:
        _set_nested(overrides, ("runtime", "surprisal_threshold"), surprisal_threshold)

    debug = _env_bool(env, "MNEMOS_DEBUG")
    if debug is not None:
        _set_nested(overrides, ("runtime", "debug"), debug)

    memory_safety_enabled = _env_bool(env, "MNEMOS_MEMORY_SAFETY_ENABLED")
    if memory_safety_enabled is not None:
        _set_nested(overrides, ("safety", "enabled"), memory_safety_enabled)

    secret_action = _env_value(env, "MNEMOS_MEMORY_SECRET_ACTION")
    if secret_action is not None:
        _set_nested(overrides, ("safety", "secret_action"), secret_action)

    pii_action = _env_value(env, "MNEMOS_MEMORY_PII_ACTION")
    if pii_action is not None:
        _set_nested(overrides, ("safety", "pii_action"), pii_action)

    capture_mode = _env_value(env, "MNEMOS_MEMORY_CAPTURE_MODE")
    if capture_mode is not None:
        _set_nested(overrides, ("governance", "capture_mode"), capture_mode)

    retention_ttl_days = _env_int(env, "MNEMOS_MEMORY_RETENTION_TTL_DAYS")
    if retention_ttl_days is not None:
        _set_nested(overrides, ("governance", "retention_ttl_days"), retention_ttl_days)

    max_chunks_per_scope = _env_int(env, "MNEMOS_MEMORY_MAX_CHUNKS_PER_SCOPE")
    if max_chunks_per_scope is not None:
        _set_nested(overrides, ("governance", "max_chunks_per_scope"), max_chunks_per_scope)

    return overrides


def _default_llm_model(provider: str) -> str | None:
    if provider == "ollama":
        return "llama3"
    if provider in {"openai", "openclaw", "openrouter"}:
        return "gpt-4o-mini"
    return None


def _default_embedding_model(provider: str) -> str | None:
    if provider == "ollama":
        return "nomic-embed-text"
    if provider in {"openai", "openclaw", "openrouter"}:
        return "text-embedding-3-small"
    return None


class OpenAICompatibleProviderSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api_key: str | None = None
    base_url: str


class OllamaProviderSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    base_url: str = "http://localhost:11434"


class QdrantProviderSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api_key: str | None = None


class ProvidersSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    openai: OpenAICompatibleProviderSettings = Field(
        default_factory=lambda: OpenAICompatibleProviderSettings(
            base_url="https://api.openai.com/v1"
        )
    )
    openclaw: OpenAICompatibleProviderSettings = Field(
        default_factory=lambda: OpenAICompatibleProviderSettings(
            base_url="https://api.openai.com/v1"
        )
    )
    openrouter: OpenAICompatibleProviderSettings = Field(
        default_factory=lambda: OpenAICompatibleProviderSettings(
            base_url="https://openrouter.ai/api/v1"
        )
    )
    ollama: OllamaProviderSettings = Field(default_factory=OllamaProviderSettings)
    qdrant: QdrantProviderSettings = Field(default_factory=QdrantProviderSettings)


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: Literal["mock", "ollama", "openai", "openclaw", "openrouter"] = "mock"
    model: str | None = None


class EmbeddingSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: Literal["simple", "ollama", "openai", "openclaw", "openrouter"] | None = None
    model: str | None = None
    dim: int = 384


class StorageSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["memory", "sqlite", "qdrant"] = "sqlite"
    sqlite_path: str = "mnemos_memory.db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_path: str | None = None
    qdrant_collection: str = "mnemos_memory"
    qdrant_vector_size: int | None = None


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    surprisal_threshold: float = 0.3
    debug: bool = False


class OnboardingSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: Literal["dev", "pro"] = "dev"
    preferred_host: Literal["claude-code", "cursor", "codex"] = "claude-code"


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    providers: ProvidersSettings = Field(default_factory=ProvidersSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    safety: MemorySafetyConfig = Field(default_factory=MemorySafetyConfig)
    governance: MemoryGovernanceConfig = Field(default_factory=MemoryGovernanceConfig)
    onboarding: OnboardingSettings = Field(default_factory=OnboardingSettings)

    @model_validator(mode="after")
    def apply_inferred_defaults(self) -> "AppSettings":
        if self.embedding.provider is None:
            if self.llm.provider in {"ollama", "openai", "openclaw", "openrouter"}:
                self.embedding.provider = cast(
                    Literal["ollama", "openai", "openclaw", "openrouter"],
                    self.llm.provider,
                )
            else:
                self.embedding.provider = "simple"

        if not self.llm.model:
            self.llm.model = _default_llm_model(self.llm.provider)

        if not self.embedding.model:
            provider = self.embedding.provider or "simple"
            self.embedding.model = _default_embedding_model(provider)

        return self

    def api_key_for(self, provider: str) -> str | None:
        if provider == "openai":
            return self.providers.openai.api_key
        if provider == "openclaw":
            return self.providers.openclaw.api_key or self.providers.openai.api_key
        if provider == "openrouter":
            return self.providers.openrouter.api_key
        if provider == "qdrant":
            return self.providers.qdrant.api_key
        return None

    def base_url_for(self, provider: str) -> str | None:
        if provider == "openai":
            return self.providers.openai.base_url
        if provider == "openclaw":
            return self.providers.openclaw.base_url or self.providers.openai.base_url
        if provider == "openrouter":
            return self.providers.openrouter.base_url
        if provider == "ollama":
            return self.providers.ollama.base_url
        return None


@dataclass(slots=True)
class ResolvedSettings:
    settings: AppSettings
    global_config_path: Path
    project_config_path: Path | None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ImportedSetup:
    settings: AppSettings
    sources: list[str] = field(default_factory=list)
    merged_env: dict[str, str] = field(default_factory=dict)


def load_settings(
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    global_config_path: str | Path | None = None,
    default_store_type: Literal["memory", "sqlite", "qdrant"] = "sqlite",
) -> ResolvedSettings:
    source_env = dict(os.environ if env is None else env)
    warnings: list[str] = []

    explicit_config = _env_value(source_env, "MNEMOS_CONFIG_PATH")
    resolved_global_path = Path(
        explicit_config or global_config_path or default_global_config_path()
    ).expanduser()

    merged: dict[str, Any] = AppSettings(
        storage=StorageSettings(type=default_store_type)
    ).model_dump(mode="python")
    embedding_defaults = merged.get("embedding")
    if isinstance(embedding_defaults, dict):
        embedding_defaults["provider"] = None
    if resolved_global_path.exists():
        merged = _deep_merge(merged, _read_toml_file(resolved_global_path))

    project_config_path = find_project_config_path(cwd)
    if project_config_path is not None:
        project_raw = _read_toml_file(project_config_path)
        project_sanitized = _strip_global_only_sections(project_raw, warnings=warnings)
        merged = _deep_merge(merged, project_sanitized)

    merged = _deep_merge(merged, _env_overrides(source_env))
    settings = AppSettings.model_validate(merged)
    return ResolvedSettings(
        settings=settings,
        global_config_path=resolved_global_path,
        project_config_path=project_config_path,
        warnings=warnings,
    )


def _extract_mnemos_env_from_json(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        payload = tomllib.loads(raw) if path.suffix == ".toml" else None
    except Exception:
        payload = None
    if path.suffix == ".json":
        import json

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    if not isinstance(payload, dict):
        return {}
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        return {}
    mnemos = servers.get("mnemos")
    if not isinstance(mnemos, dict):
        return {}
    env = mnemos.get("env")
    if not isinstance(env, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in env.items()
        if str(key).startswith("MNEMOS_") and value not in (None, "")
    }


def _extract_mnemos_env_from_codex(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    servers = payload.get("mcp_servers")
    if not isinstance(servers, dict):
        return {}
    mnemos = servers.get("mnemos")
    if not isinstance(mnemos, dict):
        return {}
    env = mnemos.get("env")
    if not isinstance(env, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in env.items()
        if str(key).startswith("MNEMOS_") and value not in (None, "")
    }


def import_existing_setup(
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    home: str | Path | None = None,
    global_config_path: str | Path | None = None,
    default_store_type: Literal["memory", "sqlite", "qdrant"] = "sqlite",
) -> ImportedSetup:
    merged_env = dict(os.environ if env is None else env)
    sources: list[str] = []
    home_path = Path.home() if home is None else Path(home)
    cwd_path = Path.cwd() if cwd is None else Path(cwd)

    if any(key.startswith("MNEMOS_") for key in merged_env):
        sources.append("environment")

    host_sources = [
        ("cursor", cwd_path / ".cursor" / "mcp.json", _extract_mnemos_env_from_json),
        (
            "claude-code",
            home_path / ".claude" / "claude_desktop_config.json",
            _extract_mnemos_env_from_json,
        ),
        ("codex", home_path / ".codex" / "config.toml", _extract_mnemos_env_from_codex),
    ]
    for source_name, path, extractor in host_sources:
        extracted = extractor(path)
        if not extracted:
            continue
        for key, value in extracted.items():
            merged_env.setdefault(key, value)
        sources.append(source_name)

    resolved = load_settings(
        env=merged_env,
        cwd=cwd_path,
        global_config_path=global_config_path,
        default_store_type=default_store_type,
    )
    return ImportedSetup(
        settings=resolved.settings,
        sources=sources,
        merged_env=merged_env,
    )


def _to_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_to_toml_value(item) for item in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _emit_toml_sections(
    data: Mapping[str, Any],
    *,
    prefix: tuple[str, ...] = (),
) -> list[str]:
    lines: list[str] = []
    scalar_items: list[tuple[str, Any]] = []
    table_items: list[tuple[str, Mapping[str, Any]]] = []

    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            table_items.append((key, value))
        else:
            scalar_items.append((key, value))

    if prefix:
        lines.append(f"[{'.'.join(prefix)}]")
    for key, value in scalar_items:
        lines.append(f"{key} = {_to_toml_value(value)}")
    if prefix and table_items:
        lines.append("")

    for index, (key, value) in enumerate(table_items):
        nested_lines = _emit_toml_sections(value, prefix=(*prefix, key))
        if nested_lines:
            lines.extend(nested_lines)
            if index != len(table_items) - 1:
                lines.append("")

    return lines


def save_settings(
    settings: AppSettings, path: str | Path, *, scope: Literal["global", "project"]
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = settings.model_dump(mode="python", exclude_none=True)
    if scope == "project":
        payload.pop("providers", None)
    text = "\n".join(_emit_toml_sections(payload)).strip() + "\n"
    target.write_text(text, encoding="utf-8")
