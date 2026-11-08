"""Check external revocation through one persistent installed MCP connection.

This is a protocol integration check, not an actual Claude/Codex host run.
Invoke under the Windows job supervisor with the prepared package runtime.
"""

import argparse
import asyncio
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def require_scope_denial(text, *, store_scope_id=None):
    result = json.loads(text)
    expected = {
        "schema_version": 1,
        "status": "degraded",
        "committed": False,
        "degraded": True,
        "error": "Mnemos service returned HTTP 503 (scope_unavailable)",
    }
    if store_scope_id is not None:
        expected.update(event_committed=False, scope="project", scope_id=store_scope_id)
    assert result == expected, "Expected a content-free, uncommitted scope denial"


def snapshot(path):
    result = {}
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        for table in (
            "memory_chunks",
            "episodic_events",
            "memory_jobs",
            "fact_evidence",
            "fact_revisions",
            "memory_suppressions",
        ):
            rows = db.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            result[table] = dict(
                count=len(rows), sha256=hashlib.sha256(repr(rows).encode()).hexdigest()
            )
    return result


async def exercise(root, target, runtime, owner, registry):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    bootstrap = root / "mcp.py"
    bootstrap.write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        + f"sys.path.insert(0,{str(target)!r})\n"
        + "import mnemos\n"
        + f"assert Path(mnemos.__file__).resolve().is_relative_to(Path({str(target)!r}))\n"
        + f"with Path({str(root / 'mcp-owner.json')!r}).open('x') as f: json.dump(dict(pid=os.getpid(),module=mnemos.__file__),f)\n"
        + "from mnemos.mcp_server import main\nmain()\n",
        encoding="utf-8",
    )
    rows = []
    params = StdioServerParameters(
        command=str(runtime),
        args=["-I", str(bootstrap)],
        env=dict(os.environ),
        cwd=str(root / "project"),
    )
    marker = "The registry-session launch label is CEDAR-742."
    blocked = "The revoked write marker is AZALEA-981."
    owner_identity = (owner.descriptor["pid"], owner.owner_thread_id, owner.token)
    with (root / "mcp-stderr.log").open("w", encoding="utf-8") as stderr:
        async with stdio_client(params, errlog=stderr) as streams:
            async with ClientSession(*streams) as session:
                initialized = await session.initialize()
                save(root / "mcp-initialize.json", initialized.model_dump(mode="json"))

                async def call(name, arguments, denied=False):
                    response = await session.call_tool(name, arguments)
                    raw = response.model_dump(mode="json")
                    rows.append(
                        dict(tool=name, arguments=arguments, denied_expected=denied, result=raw)
                    )
                    save(root / "calls.json", rows)
                    text = "\n".join(item.text for item in response.content if item.type == "text")
                    if denied:
                        require_scope_denial(
                            text,
                            store_scope_id=(
                                next(iter(registry.projects)) if name == "mnemos_store" else None
                            ),
                        )
                        assert "CEDAR-742" not in text and "AZALEA-981" not in text
                        return None
                    assert not response.isError, text
                    return json.loads(text)

                stored = await call("mnemos_store", {"content": marker})
                assert stored["status"] == "committed" and stored["job_id"]
                await call("mnemos_consolidate", {})
                job = await call("mnemos_job_status", {"job_id": stored["job_id"]})
                assert job["job"]["status"] == "completed"
                fact_id = job["job"]["result"]["fact_ids"][0]
                inspected = await call("mnemos_inspect", {"fact_id": fact_id})
                assert inspected["fact"]["content"] == marker
                version = inspected["fact"]["version"]
                before = snapshot(root / "memory.sqlite")
                assert (
                    before["memory_chunks"]["count"]
                    == before["episodic_events"]["count"]
                    == before["memory_jobs"]["count"]
                    == 1
                )
                project = next(iter(registry.projects.values()))
                project.enabled = False
                registry.save(root / "scopes.json")
                denied_operations = [
                    ("mnemos_retrieve", {"query": "launch label", "allowed_scopes": "project"}),
                    (
                        "mnemos_retrieve",
                        {
                            "query": "launch label",
                            "allowed_scopes": "project",
                            "view": "captured_evidence",
                            "response_format": "compact",
                        },
                    ),
                    ("mnemos_list", {}),
                    ("mnemos_inspect", {"fact_id": fact_id}),
                    ("mnemos_job_status", {"job_id": stored["job_id"]}),
                    ("mnemos_store", {"content": blocked}),
                    ("mnemos_forget", {"chunk_id": fact_id, "expected_version": version}),
                ]
                for name, arguments in denied_operations:
                    await call(name, arguments, denied=True)
                after_denied = snapshot(root / "memory.sqlite")
                save(root / "revocation-state.json", dict(before=before, after=after_denied))
                assert before == after_denied, "Revoked operations changed durable state"
                project.enabled = True
                registry.save(root / "scopes.json")
                restored = await call("mnemos_inspect", {"fact_id": fact_id})
                assert (
                    restored["fact"]["content"] == marker and restored["fact"]["version"] == version
                )
                recall = await call(
                    "mnemos_retrieve", {"query": marker, "allowed_scopes": "project"}
                )
                assert fact_id in json.dumps(recall) and "CEDAR-742" in json.dumps(recall)
                second = await call(
                    "mnemos_store",
                    {"content": "The re-enabled registry session uses copper labels."},
                )
                assert second["status"] == "committed"
                await call("mnemos_consolidate", {})
                completed = await call("mnemos_job_status", {"job_id": second["job_id"]})
                assert completed["job"]["status"] == "completed"
                assert (
                    owner.descriptor["pid"],
                    owner.owner_thread_id,
                    owner.token,
                ) == owner_identity
                final = snapshot(root / "memory.sqlite")
                assert (
                    final["memory_chunks"]["count"]
                    == final["episodic_events"]["count"]
                    == final["memory_jobs"]["count"]
                    == 2
                )
                save(
                    root / "session-result.json",
                    dict(
                        status="passed",
                        calls=len(rows),
                        denied=len(denied_operations),
                        owner_identity_unchanged=True,
                        mcp_connection_count=1,
                        final=final,
                        initial_fact_id=fact_id,
                        initial_job_id=stored["job_id"],
                        resumed_job_id=second["job_id"],
                    ),
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--packages", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    package = json.loads(args.packages.read_bytes())
    info = package["arms"]["candidate"]
    target = Path(info["installed_target"]).resolve()
    assert info["source_commit"] == "ec4a36b262be61c16d807a2b39006bcad7f8367c"
    assert Path(sys.executable).resolve() == Path(package["runtime_python"]).resolve()
    assert all(sha(target / name) == digest for name, digest in info["package_files"].items())
    root.mkdir(exist_ok=False)
    sys.path.insert(0, str(target))
    from mnemos.scopes import ScopeRegistry
    from mnemos.service import MnemosService, _protect_private_path

    _protect_private_path(root)
    (root / "project").mkdir()
    (root / "runtime").mkdir()
    (root / ".mnemos").mkdir()
    (root / ".mnemos/mnemos.toml").write_text("# Isolated project defaults.\n")
    subprocess.run(
        ["git", "init", "--quiet", str(root / "project")],
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    (root / "settings.toml").write_text(
        '[embedding]\nprovider="simple"\n[storage]\ntype="sqlite"\nsqlite_path='
        + json.dumps(str(root / "memory.sqlite"))
        + "\n"
    )
    registry = ScopeRegistry()
    registry.register_project(root / "project", user_profile_id="registry-session", enabled=True)
    registry.save(root / "scopes.json")
    for key in list(os.environ):
        if key.upper().startswith("MNEMOS_") or key.upper() in ("PYTHONPATH", "PYTHONHOME"):
            del os.environ[key]
    os.environ.update(
        MNEMOS_CONFIG_PATH=str(root / "settings.toml"),
        MNEMOS_SCOPE_REGISTRY=str(root / "scopes.json"),
        MNEMOS_RUNTIME_DIR=str(root / "runtime"),
        MNEMOS_RUNTIME_DESCRIPTOR=str(root / "runtime/memory.json"),
        MNEMOS_USER_PROFILE_ID="registry-session",
        MNEMOS_HOST="claude-code",
        MNEMOS_SESSION_ID="persistent-registry-session",
        MNEMOS_AUTO_START="false",
        MNEMOS_PROJECT_ROOT=str(root / "project"),
    )
    record = dict(
        status="running",
        source_commit=info["source_commit"],
        package=info,
        helper_sha256=sha(__file__),
        kind="persistent MCP protocol integration, not native host acceptance",
    )
    save(root / "result.json", record)
    owner = MnemosService()
    try:
        descriptor = owner.start()
        from mnemos.utils.llm import UnavailableLLMProvider

        assert isinstance(owner._dispatcher.engine.llm, UnavailableLLMProvider)
        assert not any(owner._dispatcher.engine.config.modules.model_dump().values())
        record["owner_port"] = descriptor["port"]
        record["owner_thread_id"] = owner.owner_thread_id
        asyncio.run(exercise(root, target, Path(sys.executable), owner, registry))
        record["status"] = "passed"
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        try:
            owner.stop()
            record["owner_stopped"] = owner._owner_thread is None
            if "owner_port" in record:
                with socket.socket() as sock:
                    record["port_closed"] = (
                        sock.connect_ex(("127.0.0.1", record["owner_port"])) != 0
                    )
        except BaseException as exc:
            record.update(status="failed", cleanup_error=type(exc).__name__)
        record["package_unchanged"] = all(
            sha(target / name) == digest for name, digest in info["package_files"].items()
        )
        if (
            not record["package_unchanged"]
            or not record.get("owner_stopped")
            or not record.get("port_closed")
        ):
            record["status"] = "failed"
        save(root / "result.json", record)
    if record["status"] != "passed":
        raise RuntimeError("Persistent MCP validation failed")


if __name__ == "__main__":
    main()
