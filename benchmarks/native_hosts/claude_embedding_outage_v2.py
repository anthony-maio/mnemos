"""Bounded native Claude retrieval while the local embedding endpoint is down.

The runner is intentionally strict about its disposable environment.  It keeps
an owned loopback socket bound without listening, so no other process can take
the port while the Ollama provider receives connection failures.  The seeded
vectors are synthetic schema fixtures for the lexical fallback path; this is
not an embedding quality experiment.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
from typing import Any, Mapping, Sequence
import zipfile

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "native_hosts"

from .claude_service_outage_v1 import rows
from .native_processes import (
    CLAUDE_PREFIX,
    PhaseEnvironment,
    _selectors,
    clean_environment,
    run_claude_phase,
)
from .outage_acceptance import require_health_gate

PRODUCT_COMMIT = "4bbab4f01fd6b5a8e1e4d76504eaa8d59d349da8"
CHECKS = Path(
    "C:/Users/Ant/AppData/Local/Temp/mnemos-package-checks-retrieval-outage-v1/verification.json"
)
INSTALLED = Path(
    "C:/Users/Ant/AppData/Local/Temp/mnemos-installed-checks-retrieval-outage-v1/installation.json"
)
OUTPUT_ROOT = Path("C:/Users/Ant/AppData/Local/Temp/mnemos-native-claude-embedding-outage-v2")
PROTOCOL = "docs/plans/2026-09-20-native-claude-embedding-outage-v2.md"
TARGET_CONTENT = "The deployment marker is maple-heron-719."
TARGET_QUERY = "What is the deployment marker for this project?"
PROMPT = (
    TARGET_QUERY
    + " Call mnemos_retrieve exactly once with query 'deployment marker', then answer with the stored deployment marker."
)
TARGET_VECTOR_DIMENSION = 768


def save(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode())


def synthetic_vector(seed: float = 1.0, dimension: int = TARGET_VECTOR_DIMENSION) -> list[float]:
    """Return a nonzero schema fixture vector, never a provider result."""
    if dimension <= 0 or not seed:
        raise ValueError("synthetic vectors need a positive dimension and nonzero seed")
    return [seed if index == 0 else 0.0 for index in range(dimension)]


def owned_non_listening_socket() -> socket.socket:
    """Bind an owned port and deliberately leave it in the non-listening state."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    sock.bind(("127.0.0.1", 0))
    return sock


def require_retrieval_response(
    response: Mapping[str, Any],
    *,
    target_id: str,
    target_content: str,
    project_id: str,
    foreign_id: str,
    foreign_content: str,
) -> None:
    """Validate the real service response, including degraded lexical fallback."""
    if response.get("status") != "degraded" or response.get("degraded") is not True:
        raise ValueError("embedding outage did not produce an honest degraded response")
    retrieval = response.get("retrieval")
    if not isinstance(retrieval, Mapping) or retrieval.get("mode") != "lexical_only":
        raise ValueError("retrieval response did not identify lexical-only fallback")
    if retrieval.get("reason") != "embedding_unavailable":
        raise ValueError("degraded retrieval did not identify the embedding outage")
    facts = response.get("facts")
    if not isinstance(facts, list) or len(facts) != 1:
        raise ValueError("degraded retrieval must return only the seeded target")
    target = [fact for fact in facts if isinstance(fact, Mapping) and fact.get("id") == target_id]
    if len(target) != 1:
        raise ValueError("seeded target fact was not returned exactly once")
    fact = target[0]
    if fact.get("content") != target_content or fact.get("scope") != "project":
        raise ValueError("target content or scope changed")
    if fact.get("scope_id") != project_id:
        raise ValueError("target project scope changed")
    scores = fact.get("scores")
    if not isinstance(scores, Mapping) or scores.get("semantic") is not None:
        raise ValueError("lexical fallback exposed a semantic score")
    if scores.get("retrieval_mode") != "lexical_only":
        raise ValueError("target score did not preserve lexical-only mode")
    if scores.get("state_status") != "not_measured" or scores.get("graph_status") != "not_used":
        raise ValueError("degraded retrieval used state or graph scoring")
    if scores.get("total") is not None:
        raise ValueError("lexical fallback exposed a total semantic score")
    if any(
        isinstance(item, Mapping)
        and (item.get("id") == foreign_id or item.get("content") == foreign_content)
        for item in facts
    ):
        raise ValueError("foreign scope fact leaked into retrieval")
    context = response.get("context")
    if (
        not isinstance(context, str)
        or target_content not in context
        or foreign_content in context
        or foreign_id in context
    ):
        raise ValueError("rendered degraded context omitted the target fact")


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(b.get("text", ""))
            for b in content
            if isinstance(b, Mapping) and b.get("type") == "text"
        )
    return ""


def decode_tool_response(content: Any) -> dict[str, Any]:
    """Decode the bounded JSON envelope emitted by the native MCP client."""
    value = content
    for _ in range(6):
        if isinstance(value, str):
            value = json.loads(value)
        elif isinstance(value, Mapping) and "facts" in value:
            return dict(value)
        elif isinstance(value, Mapping) and set(value) == {"result"}:
            value = value["result"]
        elif (
            isinstance(value, list)
            and len(value) == 1
            and isinstance(value[0], Mapping)
            and value[0].get("type") == "text"
        ):
            value = value[0].get("text")
        else:
            break
    raise ValueError("native tool result has no recognized retrieval response")


def require_native_retrieval(
    records: Sequence[Mapping[str, Any]],
    *,
    submitted_prompt: str,
    target_content: str,
    expected_model: str,
    expected_query: str = "deployment marker",
    expected_session_id: str | None = None,
) -> dict[str, Any]:
    """Require one correlated successful retrieval and a later final answer."""
    messages = [r for r in records if r.get("type") in {"user", "assistant"}]
    if not messages or messages[0].get("type") != "user":
        raise ValueError("native conversation must start with its submitted prompt")
    session_id = messages[0].get("sessionId")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("native session identity is absent")
    if expected_session_id is not None and session_id != expected_session_id:
        raise ValueError("native session differs from launched process")
    tool_use: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    final_answer = ""
    for index, row in enumerate(messages):
        if row.get("sessionId") != session_id:
            raise ValueError("native conversation contains a foreign session")
        message = row.get("message")
        if not isinstance(message, Mapping):
            raise ValueError("native message is missing")
        content = message.get("content")
        blocks = content if isinstance(content, list) else []
        if index == 0:
            if _message_text(content) != submitted_prompt:
                raise ValueError("native prompt differs from frozen request")
            continue
        if row["type"] == "assistant":
            if message.get("model") != expected_model:
                raise ValueError("native model is absent or differs from frozen model")
            for block in blocks:
                if isinstance(block, Mapping) and block.get("type") == "tool_use":
                    if (
                        tool_use is not None
                        or block.get("name") != CLAUDE_PREFIX + "mnemos_retrieve"
                    ):
                        raise ValueError("native tool differs or more than one tool call occurred")
                    if not isinstance(block.get("id"), str) or not block["id"]:
                        raise ValueError("native tool identity is absent")
                    args = block.get("input")
                    if not isinstance(args, Mapping) or args.get("query") != expected_query:
                        raise ValueError("native retrieval query differs")
                    tool_use = dict(block)
                elif isinstance(block, Mapping) and block.get("type") == "tool_result":
                    raise ValueError("tool result appeared in an assistant message")
            if result is not None:
                final_answer = _message_text(content)
        else:
            if not blocks or any(
                not isinstance(b, Mapping) or b.get("type") != "tool_result" for b in blocks
            ):
                raise ValueError("unexpected second native prompt")
            for block in blocks:
                if (
                    tool_use is None
                    or result is not None
                    or block.get("tool_use_id") != tool_use["id"]
                    or block.get("is_error", False) is not False
                ):
                    raise ValueError("native tool result is failed, duplicate, or uncorrelated")
                result = decode_tool_response(block.get("content"))
    marker = target_content.rstrip(".").split()[-1]
    if (
        tool_use is None
        or result is None
        or messages[-1].get("type") != "assistant"
        or marker not in final_answer
    ):
        raise ValueError("native result and final answer are not established")
    if target_content not in json.dumps(result, ensure_ascii=True):
        raise ValueError("native tool result omitted the target")
    return {"session_id": session_id, "tool_use_id": tool_use["id"], "response": result}


def require_cleanup(report: Mapping[str, Any], root: Path, descriptor: Path) -> None:
    if report.get("cleanup_errors"):
        raise ValueError("disposable service cleanup reported errors")
    if descriptor.exists() or any(root.rglob(".credentials.json")):
        raise ValueError("runtime descriptor or copied credentials remain")
    if report.get("embedding_responses_observed") != 0:
        raise ValueError("provider returned an embedding despite the outage pin")


def require_seed_unchanged(store: Any, snapshots: Mapping[str, Mapping[str, Any]]) -> None:
    """Ensure retrieval metadata did not rewrite the seeded fact identity/content."""
    for fact_id, expected in snapshots.items():
        current = store.get(fact_id)
        if current is None:
            raise ValueError("seeded fact disappeared")
        if current.content != expected["content"] or current.version != expected["version"]:
            raise ValueError("seeded fact content or version changed")
        metadata = expected["metadata"]
        for key in ("scope", "scope_id", "user_profile_id", "source_kind", "assertion_status"):
            if current.metadata.get(key) != metadata.get(key):
                raise ValueError("seeded fact scope or provenance changed")


def read_only_seed_summary(database: Path, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Read the seeded rows through a SQLite read-only connection."""
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        result: dict[str, dict[str, Any]] = {}
        for fact_id in ids:
            row = connection.execute(
                "SELECT content, version, metadata FROM memory_chunks WHERE id=?", (fact_id,)
            ).fetchone()
            if row is None:
                raise ValueError("seeded fact disappeared")
            result[fact_id] = {
                "content": row["content"],
                "version": row["version"],
                "metadata": json.loads(row["metadata"]),
            }
        return result
    finally:
        connection.close()


def require_mcp_response_match(native: Mapping[str, Any], service: Mapping[str, Any]) -> None:
    """Account only for the existing MCP fact schema marker, preserving every field."""
    expected = deepcopy(dict(service))
    for fact in expected.get("facts", []):
        fact.setdefault("schema_version", expected.get("schema_version", 1))
    if native != expected:
        raise ValueError("native result differs from declared MCP projection of service response")
    for fact in native.get("facts", []):
        if type(fact.get("schema_version")) is not int or fact["schema_version"] != 1:
            raise ValueError("native fact schema version differs")


def _check_inputs(repo: Path) -> dict[str, Any]:
    verification = json.loads(CHECKS.read_bytes())
    installed = json.loads(INSTALLED.read_bytes())
    required = {"pytest", "black", "mypy", "javascript", "build", "twine"}
    passed = {
        item["name"]
        for item in verification.get("checks", [])
        if item.get("status") == "passed" and item.get("exit_code") == 0
    }
    if (
        verification.get("status") != "passed"
        or verification.get("source_commit") != PRODUCT_COMMIT
    ):
        raise ValueError("source verification is not passed for the pinned commit")
    if passed != required:
        raise ValueError("all six source checks must pass")
    if installed.get("status") != "passed" or installed.get("commit") != PRODUCT_COMMIT:
        raise ValueError("installed verification is not passed for the pinned commit")
    if Path(installed.get("checks", "")).resolve() != CHECKS.resolve():
        raise ValueError("installed verification is not bound to source verification")
    wheel_rows = [
        item for item in verification.get("artifacts", []) if item["file"].endswith(".whl")
    ]
    if len(wheel_rows) != 1:
        raise ValueError("exactly one verified wheel is required")
    wheel = Path(wheel_rows[0]["file"])
    wheel_sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if wheel_sha != wheel_rows[0]["sha256"]:
        raise ValueError("verified wheel changed")
    package_checks = {
        "source_commit": PRODUCT_COMMIT,
        "verification_sha256": hashlib.sha256(CHECKS.read_bytes()).hexdigest(),
        "installation_sha256": hashlib.sha256(INSTALLED.read_bytes()).hexdigest(),
        "wheel_sha256": wheel_sha,
    }
    for binding_name in ("bindings_before", "bindings_after"):
        binding = installed.get(binding_name, {})
        if (
            binding.get("checks_sha256") != package_checks["verification_sha256"]
            or binding.get("wheel_sha256") != wheel_sha
            or binding.get("commit") != PRODUCT_COMMIT
        ):
            raise ValueError("installed verification bindings changed")
    return {"verification": verification, "installed": installed, "wheel": wheel, **package_checks}


def _check_committed_inputs(repo: Path) -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    names = [
        PROTOCOL,
        "benchmarks/native_hosts/claude_embedding_outage_v2.py",
        "benchmarks/native_hosts/native_processes.py",
        "benchmarks/native_hosts/outage_acceptance.py",
        "benchmarks/native_hosts/claude_service_outage_v1.py",
        "benchmarks/native_hosts/capture_coverage.py",
    ]
    hashes: dict[str, str] = {}
    for name in names:
        raw = (repo / name).read_bytes()
        committed = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{name}"])
        if raw.replace(b"\r\n", b"\n") != committed.replace(b"\r\n", b"\n"):
            raise ValueError("runner inputs must be committed before native execution")
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return commit, hashes


def _copy_verified_plugin(root: Path) -> dict[str, str]:
    """Copy the already verified Claude plugin without reading private config."""
    source_root = Path(
        "C:/Users/Ant/AppData/Local/Temp/mnemos-three-host-native-v7/plugin-snapshot"
    )
    old_report = Path("C:/Users/Ant/AppData/Local/Temp/mnemos-three-host-native-v7/report.json")
    bindings = json.loads(old_report.read_bytes()).get("bindings", {}).get("plugin_files", {})
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("verified Claude plugin file bindings are missing")
    hashes: dict[str, str] = {}
    for name, expected in bindings.items():
        source = source_root / name
        if not source.is_file():
            raise ValueError(f"verified Claude plugin file is missing: {name}")
        destination = root / "plugin" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"verified Claude plugin file changed: {name}")
        destination.write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return hashes


def main() -> int:
    if len(sys.argv) not in {2, 3} or (len(sys.argv) == 3 and sys.argv[2] != "--preflight"):
        raise ValueError("Pass the exact fresh output path, optionally with --preflight")
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[2]
    if root != OUTPUT_ROOT.resolve() or root.exists() or root.is_relative_to(repo):
        raise ValueError("output must be the exact fresh path outside the repository")
    package = _check_inputs(repo)
    commit, input_hashes = _check_committed_inputs(repo)
    from mnemos import __file__ as mnemos_file

    installed_python = Path(package["installed"]["environment"]) / "Scripts/python.exe"
    if Path(sys.executable).resolve() != installed_python.resolve():
        raise ValueError("native runner must use the verified installed interpreter")
    installed_package = Path(mnemos_file).resolve().parent
    if installed_package.is_relative_to(repo):
        raise ValueError("native runner imported mnemos from the source checkout")
    with zipfile.ZipFile(package["wheel"]) as wheel_file:
        for name in wheel_file.namelist():
            if name.startswith("mnemos/") and name.endswith(".py"):
                if (installed_package.parent / name).read_bytes() != wheel_file.read(name):
                    raise ValueError("installed mnemos bytes differ from verified wheel")
    executable = Path.home() / ".local/bin/claude.exe"
    version = subprocess.check_output([str(executable), "--version"], text=True, timeout=15).strip()
    if "2.1.191" not in version.split():
        raise ValueError("Claude version changed")
    if len(sys.argv) == 3:
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "runner_commit": commit,
                    "input_hashes": input_hashes,
                    "package_checks": {k: v for k, v in package.items() if k.endswith("sha256")},
                    "claude_version": version,
                }
            )
        )
        return 0

    from mnemos.client import MnemosClient
    from mnemos.config import MnemosConfig, ModuleConfig
    from mnemos.engine import MnemosEngine
    from mnemos.memory_service import LocalMemoryService
    from mnemos.scopes import ScopeRegistry, context_metadata
    from mnemos.service import MnemosService, _protect_private_path
    from mnemos.types import MemoryChunk
    from mnemos.utils.embeddings import OllamaEmbeddingProvider
    from mnemos.utils.llm import UnavailableLLMProvider
    from mnemos.utils.storage import SQLiteStore
    from mnemos.settings import load_settings

    root.mkdir(parents=True)
    _protect_private_path(root)
    plugin_hashes = _copy_verified_plugin(root)
    project, foreign = root / "project", root / "foreign-project"
    project.mkdir()
    foreign.mkdir()
    profile = "native-claude-embedding-outage-v2"
    registry = ScopeRegistry()
    project_entry = registry.register_project(project, user_profile_id=profile, enabled=True)
    foreign_entry = registry.register_project(foreign, user_profile_id=profile, enabled=True)
    registry.save(root / "scopes.json")
    settings = root / "settings.toml"
    settings.write_text(
        "[modules]\nsurprisal=false\nmutable_rag=false\naffective=false\nsleep=false\nspreading=false\n\n"
        '[storage]\ntype="sqlite"\nsqlite_path=' + json.dumps(str(root / "memory.sqlite")) + "\n",
        encoding="utf-8",
    )
    descriptor = root / "runtime.json"
    database = root / "memory.sqlite"
    config_env = PhaseEnvironment(
        root, project, root / "scopes.json", descriptor, settings, profile, root / "plugin"
    )
    config_env.validate()
    selectors = _selectors(config_env, "claude-code", "claude-embedding-outage")
    resolved = load_settings(env=clean_environment(selectors), cwd=project)
    if (
        resolved.settings.storage.type != "sqlite"
        or Path(resolved.settings.storage.sqlite_path).resolve() != database
    ):
        raise ValueError("disposable SQLite selector did not resolve")
    for key in list(os.environ):
        if key.upper().startswith("MNEMOS_"):
            del os.environ[key]
    os.environ.update(selectors)
    sock = owned_non_listening_socket()
    port = sock.getsockname()[1]
    target_context = registry.resolve(
        project, user_profile_id=profile, session_id="seed-session", host="claude-code"
    )
    foreign_context = registry.resolve(
        foreign, user_profile_id=profile, session_id="foreign-session", host="claude-code"
    )
    target_meta = {
        **context_metadata(target_context, "project"),
        "assertion_status": "asserted",
        "source_kind": "synthetic_fixture",
    }
    foreign_meta = {
        **context_metadata(foreign_context, "project"),
        "assertion_status": "asserted",
        "source_kind": "synthetic_fixture",
    }
    seed = MemoryChunk(
        id="target-maple-heron-719",
        content=TARGET_CONTENT,
        embedding=synthetic_vector(),
        metadata=target_meta,
        version=1,
    )
    foreign_chunk = MemoryChunk(
        id="foreign-canary-849",
        content="Foreign deployment marker is not project memory.",
        embedding=synthetic_vector(2.0),
        metadata=foreign_meta,
        version=1,
    )
    seed_store = SQLiteStore(str(database))
    seed_store.store(seed)
    seed_store.store(foreign_chunk)
    seed_store.close()
    expected_seed = {
        seed.id: {"content": seed.content, "version": seed.version, "metadata": target_meta},
        foreign_chunk.id: {
            "content": foreign_chunk.content,
            "version": foreign_chunk.version,
            "metadata": foreign_meta,
        },
    }
    seed_snapshot = read_only_seed_summary(database, list(expected_seed))
    if seed_snapshot != expected_seed:
        raise ValueError("persisted seed does not match declared synthetic fixture")
    dispatches: list[dict[str, Any]] = []
    health_responses: list[dict[str, Any]] = []
    embedding_responses: list[dict[str, Any]] = []
    observed_modules: list[dict[str, bool]] = []
    module_config = {name: False for name in ModuleConfig.model_fields}
    report: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "runner_commit": commit,
        "input_hashes": input_hashes,
        "package_checks": {k: v for k, v in package.items() if k.endswith("sha256")},
        "plugin_hashes": plugin_hashes,
        "model_requested": "claude-sonnet-4-6",
        "background_processing_enabled": False,
        "modules_enabled": False,
        "injected_module_config": module_config,
        "observed_module_configs": observed_modules,
        "synthetic_embedding_dimension": TARGET_VECTOR_DIMENSION,
        "embedding_endpoint": f"http://127.0.0.1:{port}",
        "owned_socket_bound_non_listening": True,
        "target_id": seed.id,
        "foreign_id": foreign_chunk.id,
    }

    class TracedService(LocalMemoryService):
        async def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
            result = await super().dispatch(operation, payload)
            dispatches.append(
                {"operation": operation, "payload": deepcopy(payload), "result": deepcopy(result)}
            )
            return result

    def factory() -> LocalMemoryService:
        modules = ModuleConfig(**module_config)
        runtime_config = MnemosConfig(modules=modules)
        runtime_config.surprisal.embedding_dim = TARGET_VECTOR_DIMENSION
        observed_modules.append(runtime_config.modules.model_dump())
        return TracedService(
            MnemosEngine(
                config=runtime_config,
                llm=UnavailableLLMProvider(),
                embedder=OllamaEmbeddingProvider(
                    base_url=f"http://127.0.0.1:{port}",
                    timeout=0.2,
                    response_observer=embedding_responses.append,
                ),
                store=SQLiteStore(str(database)),
                scope_registry=registry,
                require_scope_context=True,
                owns_providers=True,
            ),
            registry,
        )

    service: MnemosService | None = None
    try:
        service = MnemosService(
            dispatcher_factory=factory,
            descriptor_path=descriptor,
            service_log_path=root / "service.log",
        )
        service.start()

        async def read_health() -> dict[str, Any]:
            client = MnemosClient(descriptor_path=descriptor, auto_start=False)
            try:
                return await client.request("health", {})
            finally:
                await client.close()

        health = asyncio.run(read_health())
        health_responses.append(health)
        require_health_gate(
            health,
            expected_database=database,
            module_config=observed_modules[-1],
        )
        report["health"] = health
        process = run_claude_phase(
            config_env,
            phase="claude-embedding-outage",
            prompt=PROMPT,
            executable=executable,
            auth_source=Path.home() / ".claude/.credentials.json",
            allowed_tools=[CLAUDE_PREFIX + "mnemos_retrieve"],
        )
        report["native_process"] = process
        if process.get("status") != "completed" or not process.get("auth_copy_removed"):
            raise ValueError("native Claude phase did not complete cleanly")
        paths = list((root / "claude-embedding-outage/claude-home/projects").rglob("*.jsonl"))
        if len(paths) != 1:
            raise ValueError("native transcript file count differs")
        native = [json.loads(line) for line in paths[0].read_bytes().splitlines()]
        native_call = require_native_retrieval(
            native,
            submitted_prompt=PROMPT,
            target_content=TARGET_CONTENT,
            expected_model="claude-sonnet-4-6",
            expected_session_id=process["session_id"],
        )
        report["native_retrieval"] = native_call
        retrieve_rows = [row for row in dispatches if row["operation"] == "retrieve"]
        matching_rows = [
            row for row in retrieve_rows if row["payload"].get("query") == "deployment marker"
        ]
        if len(matching_rows) != 1:
            raise ValueError("native tool must have one matching retrieval dispatch")
        require_mcp_response_match(native_call["response"], matching_rows[0]["result"])
        require_retrieval_response(
            matching_rows[-1]["result"],
            target_id=seed.id,
            target_content=TARGET_CONTENT,
            project_id=project_entry.id,
            foreign_id=foreign_chunk.id,
            foreign_content=foreign_chunk.content,
        )
        final = rows(database)
        if final["fact_count"] != 2 or final["integrity"] != "ok":
            raise ValueError("seeded fact count or SQLite integrity changed")
        actual_seed = read_only_seed_summary(database, list(seed_snapshot))
        if actual_seed != seed_snapshot:
            raise ValueError("seeded fact content, version, or metadata changed")
        report.update(
            status="native_embedding_outage_lexical_retrieval_observed",
            native_transcript_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            dispatch_count=len(dispatches),
            provider_calls=len(embedding_responses),
            seed_snapshot=seed_snapshot,
            final_seed_snapshot=actual_seed,
            final_fact_count=final["fact_count"],
            final_integrity=final["integrity"],
            native_projection="MCP adds per-fact schema_version=1; all remaining fields match",
        )
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        cleanup_errors: list[str] = []
        if service is not None:
            try:
                service.stop()
            except Exception as exc:
                cleanup_errors.append(type(exc).__name__)
        report["cleanup_errors"] = cleanup_errors
        report["embedding_responses_observed"] = len(embedding_responses)
        report["cleanup_established"] = (
            not cleanup_errors
            and not descriptor.exists()
            and not any(root.rglob(".credentials.json"))
        )
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        try:
            require_cleanup(report, root, descriptor)
        except ValueError as exc:
            report.update(status="failed", cleanup_gate_error=str(exc))
        save(root / "dispatch-records.json", dispatches)
        save(root / "health-responses.json", health_responses)
        save(root / "report.json", report)
        sock.close()
    require_cleanup(report, root, descriptor)
    return 0 if report["status"] == "native_embedding_outage_lexical_retrieval_observed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
