"""
tests/test_cli.py — Tests for CLI runtime wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from argparse import Namespace

import mnemos.cli as cli_module
from mnemos.cli import (
    _build_antigravity_policy,
    _build_engine,
    _build_profile_env,
    _cmd_antigravity,
    _cmd_autostore_hook,
    _cmd_doctor,
    _cmd_export,
    _cmd_inspect,
    _cmd_list,
    _cmd_migrate_store,
    _cmd_purge,
    _cmd_profile,
    _cmd_retrieve,
    _cmd_search,
    _cmd_store,
)
from mnemos.types import ActivationNode, MemoryChunk, ProcessResult
from mnemos.utils import (
    OpenAIEmbeddingProvider,
    OpenAIProvider,
    SimpleEmbeddingProvider,
    SQLiteStore,
)


def test_build_engine_supports_storage_alias_vars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "mnemos_cli_alias.db"
    monkeypatch.delenv("MNEMOS_STORE_TYPE", raising=False)
    monkeypatch.delenv("MNEMOS_SQLITE_PATH", raising=False)
    monkeypatch.setenv("MNEMOS_STORAGE", "sqlite")
    monkeypatch.setenv("MNEMOS_DB_PATH", str(db_path))
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "mock")
    monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "simple")

    engine = _build_engine()
    assert isinstance(engine.store, SQLiteStore)
    assert engine.store.db_path == str(db_path)
    assert isinstance(engine.embedder, SimpleEmbeddingProvider)
    engine.store.close()


def test_build_engine_supports_openclaw_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "mnemos_cli_openclaw.db"
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "openclaw")
    monkeypatch.setenv("MNEMOS_OPENCLAW_API_KEY", "claw-key")
    monkeypatch.setenv("MNEMOS_OPENCLAW_URL", "https://api.openclaw.example/v1")
    monkeypatch.setenv("MNEMOS_LLM_MODEL", "openclaw/claude")
    monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "simple")
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "sqlite")
    monkeypatch.setenv("MNEMOS_SQLITE_PATH", str(db_path))

    engine = _build_engine()

    assert isinstance(engine.llm, OpenAIProvider)
    assert engine.llm.api_key == "claw-key"
    assert engine.llm.base_url == "https://api.openclaw.example/v1"
    engine.store.close()


def test_build_engine_infers_openclaw_embedder_from_llm_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "mnemos_cli_openclaw_inferred.db"
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "openclaw")
    monkeypatch.setenv("MNEMOS_OPENCLAW_API_KEY", "claw-key")
    monkeypatch.setenv("MNEMOS_OPENCLAW_URL", "https://api.openclaw.example/v1")
    monkeypatch.delenv("MNEMOS_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "sqlite")
    monkeypatch.setenv("MNEMOS_SQLITE_PATH", str(db_path))

    engine = _build_engine()

    assert isinstance(engine.embedder, OpenAIEmbeddingProvider)
    assert engine.embedder.api_key == "claw-key"
    assert engine.embedder.base_url == "https://api.openclaw.example/v1"
    engine.store.close()


def test_build_engine_reads_memory_governance_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "mnemos_cli_governance.db"
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "mock")
    monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "simple")
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "sqlite")
    monkeypatch.setenv("MNEMOS_SQLITE_PATH", str(db_path))
    monkeypatch.setenv("MNEMOS_MEMORY_CAPTURE_MODE", "hooks_only")
    monkeypatch.setenv("MNEMOS_MEMORY_RETENTION_TTL_DAYS", "7")
    monkeypatch.setenv("MNEMOS_MEMORY_MAX_CHUNKS_PER_SCOPE", "50")

    engine = _build_engine()
    assert engine.config.governance.capture_mode == "hooks_only"
    assert engine.config.governance.retention_ttl_days == 7
    assert engine.config.governance.max_chunks_per_scope == 50
    engine.store.close()


def test_build_engine_uses_mnemos_config_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "mnemos.toml"
    db_path = tmp_path / "configured.db"
    config_path.write_text(
        f"""
[llm]
provider = "openrouter"
model = "openrouter/auto"

[embedding]
provider = "openrouter"
model = "text-embedding-3-small"

[storage]
type = "sqlite"
sqlite_path = "{db_path.as_posix()}"

[providers.openrouter]
api_key = "router-key"
base_url = "https://openrouter.ai/api/v1"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("MNEMOS_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("MNEMOS_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("MNEMOS_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("MNEMOS_STORE_TYPE", raising=False)

    engine = _build_engine()

    assert isinstance(engine.llm, OpenAIProvider)
    assert engine.llm.base_url == "https://openrouter.ai/api/v1"
    assert isinstance(engine.embedder, OpenAIEmbeddingProvider)
    assert isinstance(engine.store, SQLiteStore)
    assert Path(engine.store.db_path).resolve() == db_path.resolve()
    engine.store.close()


@pytest.mark.asyncio
async def test_cli_doctor_prints_report(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "sqlite")
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "mock")
    await _cmd_doctor(
        Namespace(
            qdrant_chunk_threshold=5000,
            latency_p95_threshold_ms=250.0,
            observed_p95_ms=None,
        )
    )
    captured = capsys.readouterr().out
    assert '"profile": "starter"' in captured
    assert '"status": "degraded"' in captured


def test_build_profile_env_starter_defaults() -> None:
    env = _build_profile_env(
        profile="starter",
        llm_provider="openclaw",
        embedding_provider=None,
        model="openclaw/claude-sonnet",
        sqlite_path=".mnemos/memory.db",
        qdrant_path=".mnemos/qdrant",
        qdrant_url="http://localhost:6333",
        qdrant_collection="mnemos_memory",
    )
    assert env["MNEMOS_STORE_TYPE"] == "sqlite"
    assert env["MNEMOS_SQLITE_PATH"] == ".mnemos/memory.db"
    assert env["MNEMOS_LLM_PROVIDER"] == "openclaw"
    assert env["MNEMOS_EMBEDDING_PROVIDER"] == "openclaw"


def test_build_profile_env_local_performance() -> None:
    env = _build_profile_env(
        profile="local-performance",
        llm_provider="openai",
        embedding_provider="openai",
        model="gpt-4o-mini",
        sqlite_path=".mnemos/memory.db",
        qdrant_path=".mnemos/qdrant",
        qdrant_url="http://localhost:6333",
        qdrant_collection="mnemos_memory",
    )
    assert env["MNEMOS_STORE_TYPE"] == "qdrant"
    assert env["MNEMOS_QDRANT_PATH"] == ".mnemos/qdrant"
    assert env["MNEMOS_QDRANT_COLLECTION"] == "mnemos_memory"
    assert "MNEMOS_QDRANT_URL" not in env


@pytest.mark.asyncio
async def test_cli_profile_writes_dotenv(tmp_path: Path, capsys: Any) -> None:
    output_path = tmp_path / "mnemos.profile.env"
    await _cmd_profile(
        Namespace(
            profile="starter",
            format="dotenv",
            write=str(output_path),
            llm_provider="openclaw",
            embedding_provider="",
            model="",
            sqlite_path=".mnemos/memory.db",
            qdrant_path=".mnemos/qdrant",
            qdrant_url="http://localhost:6333",
            qdrant_collection="mnemos_memory",
        )
    )
    text = output_path.read_text(encoding="utf-8")
    assert "MNEMOS_STORE_TYPE=sqlite" in text
    assert "MNEMOS_SQLITE_PATH=.mnemos/memory.db" in text
    assert "MNEMOS_LLM_PROVIDER=openclaw" in text
    captured = capsys.readouterr().out
    assert "MNEMOS_STORE_TYPE=sqlite" in captured


@pytest.mark.asyncio
async def test_cli_store_forwards_scope_args(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    class DummyEngine:
        def __init__(self) -> None:
            self.scope: str | None = None
            self.scope_id: str | None = None

        async def process(
            self,
            interaction: Any,
            scope: str = "project",
            scope_id: str | None = None,
        ) -> ProcessResult:
            self.scope = scope
            self.scope_id = scope_id
            return ProcessResult(
                stored=True,
                salience=0.9,
                reason="ok",
                chunk=MemoryChunk(
                    content=interaction.content,
                    metadata={"scope": scope, "scope_id": scope_id},
                ),
            )

    engine = DummyEngine()
    monkeypatch.setattr("mnemos.cli._build_engine", lambda: engine)

    await _cmd_store(
        Namespace(
            content="remember this fact",
            role="user",
            scope="workspace",
            scope_id="ws-1",
        )
    )
    assert engine.scope == "workspace"
    assert engine.scope_id == "ws-1"
    assert '"scope": "workspace"' in capsys.readouterr().out


@pytest.mark.asyncio
async def test_cli_retrieve_forwards_scope_args(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    class DummyEngine:
        def __init__(self) -> None:
            self.current_scope: str | None = None
            self.scope_id: str | None = None
            self.allowed_scopes: tuple[str, ...] = ()

        async def retrieve(
            self,
            query: str,
            top_k: int = 5,
            reconsolidate: bool = True,
            current_scope: str = "project",
            scope_id: str | None = None,
            allowed_scopes: tuple[str, ...] = ("project", "workspace", "global"),
        ) -> list[MemoryChunk]:
            _ = query, top_k, reconsolidate
            self.current_scope = current_scope
            self.scope_id = scope_id
            self.allowed_scopes = allowed_scopes
            return [
                MemoryChunk(
                    content="result",
                    metadata={"scope": "project", "scope_id": "alpha"},
                )
            ]

    engine = DummyEngine()
    monkeypatch.setattr("mnemos.cli._build_engine", lambda: engine)

    await _cmd_retrieve(
        Namespace(
            query="deployment",
            top_k=3,
            current_scope="project",
            scope_id="alpha",
            allowed_scopes="project,global",
            reconsolidate=False,
        )
    )
    assert engine.current_scope == "project"
    assert engine.scope_id == "alpha"
    assert engine.allowed_scopes == ("project", "global")
    assert '"scope": "project"' in capsys.readouterr().out


@pytest.mark.asyncio
async def test_cli_inspect_outputs_chunk_details(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    chunk = MemoryChunk(
        id="chunk-123",
        content="We use FastAPI for internal APIs.",
        metadata={
            "scope": "project",
            "scope_id": "repo-alpha",
            "source": "surprisal_gate",
            "ingest_channel": "manual",
            "encoding_reason": "High surprisal (0.500 > threshold 0.000). Encoded with salience 0.500.",
        },
    )

    class DummyStore:
        def get(self, chunk_id: str) -> MemoryChunk | None:
            return chunk if chunk_id == chunk.id else None

    class DummySpreading:
        def get_node(self, node_id: str) -> ActivationNode | None:
            if node_id != chunk.id:
                return None
            return ActivationNode(
                id=node_id,
                content=chunk.content,
                neighbors={"neighbor-1": 0.91},
            )

    class DummyEngine:
        store = DummyStore()
        spreading_activation = DummySpreading()

    monkeypatch.setattr("mnemos.cli._build_engine", lambda: DummyEngine())

    await _cmd_inspect(Namespace(chunk_id="chunk-123"))

    payload = capsys.readouterr().out
    assert '"id": "chunk-123"' in payload
    assert '"stored_by": "surprisal_gate"' in payload
    assert '"neighbor_count": 1' in payload


def test_build_antigravity_policy_mentions_required_tools() -> None:
    policy = _build_antigravity_policy("cursor")
    assert "mnemos_retrieve" in policy
    assert "mnemos_store" in policy
    assert "mnemos_consolidate" in policy


def test_build_antigravity_policy_codex_mentions_agents() -> None:
    policy = _build_antigravity_policy("codex")
    assert "AGENTS.md" in policy
    assert "mnemos_retrieve" in policy
    assert "mnemos_consolidate" in policy


@pytest.mark.asyncio
async def test_cli_antigravity_writes_policy(tmp_path: Path, capsys: Any) -> None:
    output_path = tmp_path / "mnemos-antigravity.txt"
    await _cmd_antigravity(
        Namespace(
            host="cursor",
            format="text",
            write=str(output_path),
        )
    )
    text = output_path.read_text(encoding="utf-8")
    assert "Mnemos Antigravity Autopilot Policy" in text
    assert "mnemos_retrieve" in text
    captured = capsys.readouterr().out
    assert "Mnemos Antigravity Autopilot Policy" in captured


@pytest.mark.asyncio
async def test_cli_autostore_hook_dry_run_prints_decision(capsys: Any) -> None:
    payload = '{"prompt":"Use uv and mypy in this repo","cwd":"/tmp/repo-alpha"}'
    await _cmd_autostore_hook(
        Namespace(
            event="UserPromptSubmit",
            payload=payload,
            scope="project",
            scope_id="",
            max_chars=1200,
            dry_run=True,
        )
    )
    captured = capsys.readouterr().out
    assert '"stored": false' in captured.lower()
    assert "Dry run" in captured


@pytest.mark.asyncio
async def test_cli_autostore_hook_stores_when_decision_allows(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    class DummyEngine:
        def __init__(self) -> None:
            self.scope: str | None = None
            self.scope_id: str | None = None

        async def process(
            self,
            interaction: Any,
            scope: str = "project",
            scope_id: str | None = None,
        ) -> ProcessResult:
            self.scope = scope
            self.scope_id = scope_id
            return ProcessResult(
                stored=True,
                salience=0.8,
                reason="stored",
                chunk=MemoryChunk(
                    content=interaction.content,
                    metadata={"scope": scope, "scope_id": scope_id},
                ),
            )

    engine = DummyEngine()
    monkeypatch.setattr("mnemos.cli._build_engine", lambda: engine)

    payload = '{"prompt":"Set deployment target to ECS in this repository","cwd":"/tmp/repo-alpha"}'
    await _cmd_autostore_hook(
        Namespace(
            event="UserPromptSubmit",
            payload=payload,
            scope="project",
            scope_id="",
            max_chars=1200,
            dry_run=False,
        )
    )
    assert engine.scope == "project"
    assert engine.scope_id == "repo-alpha"
    captured = capsys.readouterr().out
    assert '"stored": true' in captured.lower()


def test_migrate_chunks_dry_run_reports_counts_without_writing() -> None:
    class DummyStore:
        def __init__(self, chunks: list[MemoryChunk]) -> None:
            self._chunks = chunks
            self.stored: list[MemoryChunk] = []

        def get_all(self) -> list[MemoryChunk]:
            return list(self._chunks)

        def store(self, chunk: MemoryChunk) -> None:
            self.stored.append(chunk)

    source = DummyStore(
        [
            MemoryChunk(content="chunk-a"),
            MemoryChunk(content="chunk-b"),
        ]
    )
    target = DummyStore([])

    summary = cli_module._migrate_chunks(source_store=source, target_store=target, dry_run=True)

    assert summary["scanned"] == 2
    assert summary["migrated"] == 0
    assert summary["dry_run"] is True
    assert target.stored == []


def test_migrate_chunks_copies_chunks_to_target() -> None:
    class DummyStore:
        def __init__(self, chunks: list[MemoryChunk]) -> None:
            self._chunks = chunks
            self.stored: list[MemoryChunk] = []

        def get_all(self) -> list[MemoryChunk]:
            return list(self._chunks)

        def store(self, chunk: MemoryChunk) -> None:
            self.stored.append(chunk)

    source = DummyStore([MemoryChunk(content="chunk-a"), MemoryChunk(content="chunk-b")])
    target = DummyStore([])

    summary = cli_module._migrate_chunks(source_store=source, target_store=target, dry_run=False)

    assert summary["scanned"] == 2
    assert summary["migrated"] == 2
    assert [chunk.content for chunk in target.stored] == ["chunk-a", "chunk-b"]


@pytest.mark.asyncio
async def test_cli_migrate_store_prints_json_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    class DummyStore:
        def __init__(self, chunks: list[MemoryChunk]) -> None:
            self._chunks = chunks
            self.stored: list[MemoryChunk] = []
            self.closed = False

        def get_all(self) -> list[MemoryChunk]:
            return list(self._chunks)

        def store(self, chunk: MemoryChunk) -> None:
            self.stored.append(chunk)

        def close(self) -> None:
            self.closed = True

    source_store = DummyStore([MemoryChunk(content="chunk-a")])
    target_store = DummyStore([])

    def fake_build_store_for_migration(**kwargs: Any) -> DummyStore:
        store_type = kwargs["store_type"]
        if store_type == "qdrant":
            return source_store
        if store_type == "neo4j":
            return target_store
        raise AssertionError(f"Unexpected store type: {store_type}")

    monkeypatch.setattr(cli_module, "_build_store_for_migration", fake_build_store_for_migration)

    await _cmd_migrate_store(
        Namespace(
            source_store="qdrant",
            target_store="neo4j",
            source_sqlite_path="",
            source_qdrant_path="",
            source_qdrant_url="",
            source_qdrant_collection="",
            source_neo4j_uri="",
            source_neo4j_database="",
            source_neo4j_label="",
            source_neo4j_username="",
            source_neo4j_password="",
            target_sqlite_path="",
            target_qdrant_path="",
            target_qdrant_url="",
            target_qdrant_collection="",
            target_neo4j_uri="",
            target_neo4j_database="",
            target_neo4j_label="",
            target_neo4j_username="",
            target_neo4j_password="",
            dry_run=False,
        )
    )

    captured = capsys.readouterr().out
    assert '"source_store": "qdrant"' in captured
    assert '"target_store": "neo4j"' in captured
    assert '"migrated": 1' in captured
    assert source_store.closed is True
    assert target_store.closed is True


@pytest.mark.asyncio
async def test_cli_list_filters_by_scope(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    class DummyStore:
        def __init__(self, chunks: list[MemoryChunk]) -> None:
            self._chunks = chunks

        def get_all(self) -> list[MemoryChunk]:
            return list(self._chunks)

    class DummyEngine:
        def __init__(self) -> None:
            self.store = DummyStore(
                [
                    MemoryChunk(
                        content="alpha fact", metadata={"scope": "project", "scope_id": "alpha"}
                    ),
                    MemoryChunk(
                        content="beta fact", metadata={"scope": "project", "scope_id": "beta"}
                    ),
                    MemoryChunk(content="global fact", metadata={"scope": "global"}),
                ]
            )

    monkeypatch.setattr("mnemos.cli._build_engine", lambda: DummyEngine())
    await _cmd_list(
        Namespace(
            scope="project",
            scope_id="alpha",
            query="",
            sort_by="created_at",
            limit=50,
        )
    )
    output = capsys.readouterr().out
    assert '"total": 1' in output
    assert "alpha fact" in output
    assert "beta fact" not in output


@pytest.mark.asyncio
async def test_cli_search_filters_by_query(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    class DummyStore:
        def __init__(self, chunks: list[MemoryChunk]) -> None:
            self._chunks = chunks

        def get_all(self) -> list[MemoryChunk]:
            return list(self._chunks)

    class DummyEngine:
        def __init__(self) -> None:
            self.store = DummyStore(
                [
                    MemoryChunk(
                        content="uses terraform modules",
                        metadata={"scope": "project", "scope_id": "alpha"},
                    ),
                    MemoryChunk(
                        content="uses ansible", metadata={"scope": "project", "scope_id": "alpha"}
                    ),
                ]
            )

    monkeypatch.setattr("mnemos.cli._build_engine", lambda: DummyEngine())
    await _cmd_search(
        Namespace(
            query="terraform",
            scope="project",
            scope_id="alpha",
            sort_by="created_at",
            limit=50,
        )
    )
    output = capsys.readouterr().out
    assert "terraform" in output
    assert "ansible" not in output


@pytest.mark.asyncio
async def test_cli_export_writes_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyStore:
        def __init__(self, chunks: list[MemoryChunk]) -> None:
            self._chunks = chunks

        def get_all(self) -> list[MemoryChunk]:
            return list(self._chunks)

    class DummyEngine:
        def __init__(self) -> None:
            self.store = DummyStore(
                [
                    MemoryChunk(content="global preference", metadata={"scope": "global"}),
                ]
            )

    monkeypatch.setattr("mnemos.cli._build_engine", lambda: DummyEngine())
    out_path = tmp_path / "export.jsonl"
    await _cmd_export(
        Namespace(
            scope="all",
            scope_id="default",
            query="",
            sort_by="created_at",
            limit=0,
            format="jsonl",
            output=str(out_path),
        )
    )
    text = out_path.read_text(encoding="utf-8")
    assert "global preference" in text
    assert text.strip().startswith("{")


@pytest.mark.asyncio
async def test_cli_purge_requires_yes(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    class DummyStore:
        def __init__(self, chunks: list[MemoryChunk]) -> None:
            self._chunks = {chunk.id: chunk for chunk in chunks}

        def get_all(self) -> list[MemoryChunk]:
            return list(self._chunks.values())

        def delete(self, chunk_id: str) -> bool:
            if chunk_id not in self._chunks:
                return False
            del self._chunks[chunk_id]
            return True

    class DummySpreading:
        def get_node(self, chunk_id: str) -> None:
            _ = chunk_id
            return None

        def remove_node(self, chunk_id: str) -> None:
            _ = chunk_id

    class DummyEngine:
        def __init__(self) -> None:
            self.store = DummyStore(
                [
                    MemoryChunk(
                        content="alpha fact", metadata={"scope": "project", "scope_id": "alpha"}
                    )
                ]
            )
            self.spreading_activation = DummySpreading()

    engine = DummyEngine()
    monkeypatch.setattr("mnemos.cli._build_engine", lambda: engine)
    await _cmd_purge(
        Namespace(
            scope="project",
            scope_id="alpha",
            query="",
            older_than_days=0,
            limit=0,
            dry_run=False,
            yes=False,
        )
    )
    output = capsys.readouterr().out
    assert "Refusing purge without --yes" in output
    assert len(engine.store.get_all()) == 1


@pytest.mark.asyncio
async def test_cli_purge_deletes_when_confirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyStore:
        def __init__(self, chunks: list[MemoryChunk]) -> None:
            self._chunks = {chunk.id: chunk for chunk in chunks}

        def get_all(self) -> list[MemoryChunk]:
            return list(self._chunks.values())

        def delete(self, chunk_id: str) -> bool:
            if chunk_id not in self._chunks:
                return False
            del self._chunks[chunk_id]
            return True

    class DummySpreading:
        def get_node(self, chunk_id: str) -> None:
            _ = chunk_id
            return None

        def remove_node(self, chunk_id: str) -> None:
            _ = chunk_id

    class DummyEngine:
        def __init__(self) -> None:
            self.store = DummyStore(
                [
                    MemoryChunk(
                        content="alpha fact", metadata={"scope": "project", "scope_id": "alpha"}
                    )
                ]
            )
            self.spreading_activation = DummySpreading()

    engine = DummyEngine()
    monkeypatch.setattr("mnemos.cli._build_engine", lambda: engine)
    await _cmd_purge(
        Namespace(
            scope="project",
            scope_id="alpha",
            query="",
            older_than_days=0,
            limit=0,
            dry_run=False,
            yes=True,
        )
    )
    assert len(engine.store.get_all()) == 0
