"""
tests/test_settings.py — Canonical config loading and persistence tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from mnemos.settings import import_existing_setup, load_settings, save_settings


def test_load_settings_merges_global_and_project_configs(tmp_path: Path) -> None:
    global_config = tmp_path / "global.toml"
    global_config.write_text(
        """
[llm]
provider = "openai"
model = "gpt-4o-mini"

[storage]
type = "sqlite"
sqlite_path = "global.db"

[providers.openai]
api_key = "global-key"
base_url = "https://api.openai.com/v1"
""".strip(),
        encoding="utf-8",
    )

    project_dir = tmp_path / "repo"
    project_config = project_dir / ".mnemos" / "mnemos.toml"
    project_config.parent.mkdir(parents=True)
    project_config.write_text(
        """
[embedding]
provider = "openai"
model = "text-embedding-3-small"

[storage]
sqlite_path = ".mnemos/project.db"
""".strip(),
        encoding="utf-8",
    )

    resolved = load_settings(
        env={},
        cwd=project_dir,
        global_config_path=global_config,
    )

    assert resolved.settings.llm.provider == "openai"
    assert resolved.settings.llm.model == "gpt-4o-mini"
    assert resolved.settings.embedding.provider == "openai"
    assert resolved.settings.embedding.model == "text-embedding-3-small"
    assert resolved.settings.storage.type == "sqlite"
    assert resolved.settings.storage.sqlite_path == ".mnemos/project.db"
    assert resolved.settings.providers.openai.api_key == "global-key"
    assert resolved.project_config_path == project_config


def test_load_settings_env_overrides_project_and_global(tmp_path: Path) -> None:
    global_config = tmp_path / "global.toml"
    global_config.write_text(
        """
[llm]
provider = "openai"

[storage]
type = "sqlite"
sqlite_path = "global.db"
""".strip(),
        encoding="utf-8",
    )

    project_dir = tmp_path / "repo"
    project_config = project_dir / ".mnemos" / "mnemos.toml"
    project_config.parent.mkdir(parents=True)
    project_config.write_text(
        """
[storage]
type = "sqlite"
sqlite_path = ".mnemos/project.db"
""".strip(),
        encoding="utf-8",
    )

    resolved = load_settings(
        env={
            "MNEMOS_LLM_PROVIDER": "ollama",
            "MNEMOS_OLLAMA_URL": "http://192.168.86.35:11434",
            "MNEMOS_STORE_TYPE": "sqlite",
            "MNEMOS_SQLITE_PATH": ".mnemos/runtime.db",
        },
        cwd=project_dir,
        global_config_path=global_config,
    )

    assert resolved.settings.llm.provider == "ollama"
    assert resolved.settings.providers.ollama.base_url == "http://192.168.86.35:11434"
    assert resolved.settings.storage.type == "sqlite"
    assert resolved.settings.storage.sqlite_path == ".mnemos/runtime.db"


def test_load_settings_rejects_legacy_store_types(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_settings(
            env={
                "MNEMOS_STORE_TYPE": "neo4j",
                "MNEMOS_NEO4J_URI": "bolt://localhost:7687",
                "MNEMOS_NEO4J_USERNAME": "neo4j",
                "MNEMOS_NEO4J_PASSWORD": "secret",
            },
            cwd=tmp_path,
        )

    with pytest.raises(ValidationError):
        load_settings(
            env={
                "MNEMOS_STORE_TYPE": "qdrant",
                "MNEMOS_QDRANT_URL": "http://localhost:6333",
            },
            cwd=tmp_path,
        )


def test_load_settings_exposes_only_live_storage_surfaces(tmp_path: Path) -> None:
    resolved = load_settings(env={}, cwd=tmp_path)

    assert not hasattr(resolved.settings.storage, "qdrant_url")
    assert not hasattr(resolved.settings.storage, "qdrant_path")
    assert not hasattr(resolved.settings.storage, "qdrant_collection")
    assert not hasattr(resolved.settings.storage, "qdrant_vector_size")
    assert not hasattr(resolved.settings.storage, "neo4j_uri")
    assert not hasattr(resolved.settings.storage, "neo4j_database")
    assert not hasattr(resolved.settings.storage, "neo4j_label")
    assert not hasattr(resolved.settings.providers, "qdrant")
    assert not hasattr(resolved.settings.providers, "neo4j")


def test_project_config_secrets_are_ignored(tmp_path: Path) -> None:
    global_config = tmp_path / "global.toml"
    global_config.write_text(
        """
[llm]
provider = "openai"

[providers.openai]
api_key = "global-key"
""".strip(),
        encoding="utf-8",
    )

    project_dir = tmp_path / "repo"
    project_config = project_dir / ".mnemos" / "mnemos.toml"
    project_config.parent.mkdir(parents=True)
    project_config.write_text(
        """
[providers.openai]
api_key = "project-key"
base_url = "https://example.invalid/v1"
""".strip(),
        encoding="utf-8",
    )

    resolved = load_settings(
        env={},
        cwd=project_dir,
        global_config_path=global_config,
    )

    assert resolved.settings.providers.openai.api_key == "global-key"
    assert resolved.settings.providers.openai.base_url == "https://api.openai.com/v1"
    assert any("project config" in warning.lower() for warning in resolved.warnings)


def test_mnemos_config_path_env_selects_global_config(tmp_path: Path) -> None:
    explicit_config = tmp_path / "explicit.toml"
    explicit_config.write_text(
        """
[llm]
provider = "openrouter"
model = "openrouter/auto"

[providers.openrouter]
api_key = "router-key"
base_url = "https://openrouter.ai/api/v1"
""".strip(),
        encoding="utf-8",
    )

    resolved = load_settings(
        env={"MNEMOS_CONFIG_PATH": str(explicit_config)},
        cwd=tmp_path,
    )

    assert resolved.global_config_path == explicit_config
    assert resolved.settings.llm.provider == "openrouter"
    assert resolved.settings.providers.openrouter.api_key == "router-key"


def test_load_settings_uses_windows_persistent_env_fallback_for_provider_keys(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "mnemos.toml"
    config_path.write_text(
        """
[llm]
provider = "openrouter"

[embedding]
provider = "openrouter"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "mnemos.settings._persistent_env_value",
        lambda name: "router-key" if name == "MNEMOS_OPENROUTER_API_KEY" else None,
        raising=False,
    )

    resolved = load_settings(
        env={"MNEMOS_CONFIG_PATH": str(config_path)},
        cwd=tmp_path,
    )

    assert resolved.settings.providers.openrouter.api_key == "router-key"


def test_persistent_env_fallback_resolves_registry_references_without_process_env(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "mnemos.toml"
    config_path.write_text(
        """
[llm]
provider = "openrouter"

[embedding]
provider = "openrouter"
""".strip(),
        encoding="utf-8",
    )

    class _FakeKey:
        def __init__(self, hive: str, subkey: str) -> None:
            self.hive = hive
            self.subkey = subkey

        def __enter__(self) -> "_FakeKey":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            _ = exc_type, exc, tb
            return None

    class _FakeWinreg:
        HKEY_CURRENT_USER = "hkcu"
        HKEY_LOCAL_MACHINE = "hklm"

        values = {
            ("hkcu", "Environment", "MNEMOS_OPENROUTER_API_KEY"): "%OPENROUTER_API_KEY%",
            ("hkcu", "Environment", "OPENROUTER_API_KEY"): "router-key",
        }

        @staticmethod
        def OpenKey(hive: str, subkey: str) -> _FakeKey:
            return _FakeKey(hive, subkey)

        @staticmethod
        def QueryValueEx(key: _FakeKey, name: str) -> tuple[str, int]:
            try:
                value = _FakeWinreg.values[(key.hive, key.subkey, name)]
            except KeyError as exc:
                raise OSError(name) from exc
            return value, 1

    monkeypatch.setattr("mnemos.settings._is_windows", lambda: True)
    monkeypatch.setitem(sys.modules, "winreg", _FakeWinreg)
    monkeypatch.setenv("OPENROUTER_API_KEY", "stale-process-key")

    resolved = load_settings(
        env={"MNEMOS_CONFIG_PATH": str(config_path)},
        cwd=tmp_path,
    )

    assert resolved.settings.providers.openrouter.api_key == "router-key"


def test_save_settings_omits_secrets_for_project_scope(tmp_path: Path) -> None:
    global_config = tmp_path / "global.toml"
    global_config.write_text(
        """
[llm]
provider = "openrouter"

[embedding]
provider = "openrouter"

[providers.openrouter]
api_key = "router-key"
base_url = "https://openrouter.ai/api/v1"
""".strip(),
        encoding="utf-8",
    )

    resolved = load_settings(env={}, cwd=tmp_path, global_config_path=global_config)
    project_config = tmp_path / ".mnemos" / "mnemos.toml"
    project_config.parent.mkdir(parents=True)

    save_settings(resolved.settings, project_config, scope="project")

    text = project_config.read_text(encoding="utf-8")
    assert "router-key" not in text
    assert "api_key" not in text
    assert 'provider = "openrouter"' in text


def test_import_existing_setup_reads_codex_host_config(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    codex_config = home_dir / ".codex" / "config.toml"
    codex_config.parent.mkdir(parents=True)
    codex_config.write_text(
        """
[mcp_servers.mnemos]
command = "mnemos-mcp"

[mcp_servers.mnemos.env]
MNEMOS_LLM_PROVIDER = "openrouter"
MNEMOS_EMBEDDING_PROVIDER = "openrouter"
MNEMOS_OPENROUTER_API_KEY = "router-key"
MNEMOS_STORE_TYPE = "sqlite"
MNEMOS_SQLITE_PATH = ".mnemos/memory.db"
""".strip(),
        encoding="utf-8",
    )

    imported = import_existing_setup(
        env={},
        cwd=tmp_path / "repo",
        home=home_dir,
    )

    assert imported.settings.llm.provider == "openrouter"
    assert imported.settings.embedding.provider == "openrouter"
    assert imported.settings.providers.openrouter.api_key == "router-key"
    assert imported.settings.storage.sqlite_path == ".mnemos/memory.db"
    assert "codex" in imported.sources


def test_import_existing_setup_reads_codex_config_path_for_sqlite(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    config_path = tmp_path / "Mnemos" / "mnemos.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
[llm]
provider = "ollama"

[embedding]
provider = "ollama"

[storage]
type = "sqlite"
sqlite_path = ".mnemos/memory.db"

[providers.ollama]
base_url = "http://ollama:11434"
""".strip(),
        encoding="utf-8",
    )

    codex_config = home_dir / ".codex" / "config.toml"
    codex_config.parent.mkdir(parents=True)
    codex_config.write_text(
        f"""
[mcp_servers.mnemos]
command = "mnemos-mcp"

[mcp_servers.mnemos.env]
MNEMOS_CONFIG_PATH = "{config_path.as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    imported = import_existing_setup(
        env={},
        cwd=tmp_path / "repo",
        home=home_dir,
    )

    assert imported.settings.storage.type == "sqlite"
    assert imported.settings.storage.sqlite_path == ".mnemos/memory.db"
    assert imported.settings.providers.ollama.base_url == "http://ollama:11434"
    assert "codex" in imported.sources


def test_import_existing_setup_reads_cursor_config_path_for_sqlite(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True)
    config_path = tmp_path / "Mnemos" / "mnemos.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
[llm]
provider = "ollama"

[embedding]
provider = "ollama"

[storage]
type = "sqlite"
sqlite_path = ".mnemos/cursor-memory.db"

[providers.ollama]
base_url = "http://ollama:11434"
""".strip(),
        encoding="utf-8",
    )

    cursor_config = repo_dir / ".cursor" / "mcp.json"
    cursor_config.parent.mkdir(parents=True)
    cursor_config.write_text(
        f"""
{{
  "mcpServers": {{
    "mnemos": {{
      "command": "mnemos-mcp",
      "env": {{
        "MNEMOS_CONFIG_PATH": "{config_path.as_posix()}"
      }}
    }}
  }}
}}
""".strip(),
        encoding="utf-8",
    )

    imported = import_existing_setup(
        env={},
        cwd=repo_dir,
        home=tmp_path / "home",
    )

    assert imported.settings.storage.type == "sqlite"
    assert imported.settings.storage.sqlite_path == ".mnemos/cursor-memory.db"
    assert imported.settings.providers.ollama.base_url == "http://ollama:11434"
    assert "cursor" in imported.sources
