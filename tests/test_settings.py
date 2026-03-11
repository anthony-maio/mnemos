"""
tests/test_settings.py — Canonical config loading and persistence tests.
"""

from __future__ import annotations

from pathlib import Path

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
            "MNEMOS_STORE_TYPE": "qdrant",
            "MNEMOS_QDRANT_PATH": ".mnemos/qdrant",
        },
        cwd=project_dir,
        global_config_path=global_config,
    )

    assert resolved.settings.llm.provider == "ollama"
    assert resolved.settings.providers.ollama.base_url == "http://192.168.86.35:11434"
    assert resolved.settings.storage.type == "qdrant"
    assert resolved.settings.storage.qdrant_path == ".mnemos/qdrant"


def test_load_settings_env_overrides_support_neo4j(tmp_path: Path) -> None:
    resolved = load_settings(
        env={
            "MNEMOS_STORE_TYPE": "neo4j",
            "MNEMOS_NEO4J_URI": "bolt://localhost:7687",
            "MNEMOS_NEO4J_USERNAME": "neo4j",
            "MNEMOS_NEO4J_PASSWORD": "secret",
            "MNEMOS_NEO4J_DATABASE": "mnemos",
            "MNEMOS_NEO4J_LABEL": "MemoryChunk",
        },
        cwd=tmp_path,
    )

    assert resolved.settings.storage.type == "neo4j"
    assert resolved.settings.storage.neo4j_uri == "bolt://localhost:7687"
    assert resolved.settings.storage.neo4j_database == "mnemos"
    assert resolved.settings.storage.neo4j_label == "MemoryChunk"
    assert resolved.settings.providers.neo4j.username == "neo4j"
    assert resolved.settings.providers.neo4j.password == "secret"


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
