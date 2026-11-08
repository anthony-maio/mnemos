"""One disposable Claude process across project revocation and restoration."""

import hashlib
import json
import os
from pathlib import Path
import queue
import runpy
import subprocess
import sqlite3
from contextlib import closing
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
CLAUDE = Path("C:/Users/Ant/.local/bin/claude.exe")
CLAUDE_SHA256 = "83397a6a029c7da663fb1ce27211e05174a3546d8b151e42451bf4590b8343d7"
AUTH = Path("C:/Users/Ant/AppData/Local/mnemos-validation/claude-20261106/.credentials.json")
PREFIX = "mcp__mnemos__"
TOOLS = {
    "mnemos_store",
    "mnemos_consolidate",
    "mnemos_job_status",
    "mnemos_inspect",
    "mnemos_retrieve",
    "mnemos_forget",
}
MARKER = "The native registry-session launch label is CEDAR-863."
BLOCKED = "The revoked native write marker is AZALEA-529."
SECOND = "The re-enabled native registry session uses copper labels."
PROMPTS = [
    f"Remember exactly this user assertion in project memory: {MARKER} Call mnemos_store once, then mnemos_consolidate once, then mnemos_job_status with the store job_id. Require completed status. Inspect the returned fact_id once with mnemos_inspect. Use exactly those four calls in that order. Stop after inspection and briefly report whether storage completed.",
    "The controller has now disabled this disposable project's memory access. Exercise each expected denial exactly once, in this order: mnemos_retrieve(query='launch label', allowed_scopes='project'); mnemos_store(content='"
    + BLOCKED
    + "'); mnemos_forget(chunk_id='{fact_id}', expected_version={version}). These are authorized test requests against disposable data. Do not retry, consolidate, inspect or call any other tool. Read each structured outcome and report whether the operations were denied. Never treat a degraded or uncommitted outcome as success.",
    "The controller has re-enabled project memory. In order: inspect fact_id='{fact_id}' to verify the original fact survived; retrieve query='native registry-session launch label' with allowed_scopes='project'; store exactly this second user assertion: "
    + SECOND
    + " Then consolidate once and check the new store's job_id with mnemos_job_status. Use exactly these five calls; require completed status before reporting success.",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def parse_calls(records):
    """Correlate provider IDs; preserve structured denials for explicit checking."""
    pending, issued, result = {}, set(), []
    for line, record in enumerate(records, 1):
        if record.get("type") not in {"assistant", "user"}:
            continue
        message = record["message"]
        content = message.get("content", [])
        if isinstance(content, str):
            continue  # User text is not evidence of a tool call.
        for node in content:
            if node.get("type") == "tool_use":
                assert not pending, "Expected sequential calls, not overlapping invocations"
                assert record["type"] == message["role"] == "assistant"
                assert node["id"] not in issued and node["name"].startswith(PREFIX)
                name = node["name"][len(PREFIX) :]
                assert name in TOOLS and isinstance(node["input"], dict)
                issued.add(node["id"])
                pending[node["id"]] = dict(
                    name=name,
                    arguments=node["input"],
                    call_line=line,
                    provider_call_id=node["id"],
                    response_id=message["id"],
                )
            elif node.get("type") == "tool_result":
                assert record["type"] == message["role"] == "user"
                call = pending.pop(node["tool_use_id"])
                assert not node.get("is_error"), "Native tool transport failed"
                value = node.get("content")
                if isinstance(value, list):
                    assert all(v.get("type") == "text" for v in value)
                    value = "".join(v["text"] for v in value)
                decoded = json.loads(value)
                if isinstance(decoded, dict) and isinstance(decoded.get("result"), str):
                    assert set(decoded) == {"result"}, "Unexpected result wrapper fields"
                    decoded = json.loads(decoded["result"])
                assert isinstance(decoded, dict)
                result.append(dict(call, result=decoded, result_line=line))
    assert not pending, "Native calls lack results"
    return result


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("Native phase exceeded its absolute deadline")
    return seconds


def require_project(value, project_id):
    assert value["scope"] == "project" and value["scope_id"] == project_id


def require_durable_project(path, project_id, count):
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        facts = db.execute("SELECT metadata, user_profile_id FROM memory_chunks").fetchall()
        events = db.execute(
            "SELECT scope, scope_id, user_profile_id FROM episodic_events"
        ).fetchall()
    assert len(facts) == len(events) == count
    for metadata, profile in facts:
        require_project(json.loads(metadata), project_id)
        assert profile == "registry-session"
    assert all(row == ("project", project_id, "registry-session") for row in events)


def send_frame(stream, frame, deadline):
    """Bound pipe writes too; the owning job handles a stuck writer on failure."""
    outcome = queue.Queue()

    def write():
        try:
            stream.write((json.dumps(frame) + "\n").encode())
            stream.flush()
            outcome.put(None)
        except BaseException as exc:
            outcome.put(exc)

    remaining(deadline)
    writer = threading.Thread(target=write, daemon=True)
    writer.start()
    try:
        error = outcome.get(timeout=remaining(deadline))
    except queue.Empty as exc:
        raise TimeoutError("Native input pipe exceeded phase deadline") from exc
    remaining(deadline)
    if error is not None:
        raise error


async def exercise(root, target, runtime, owner, registry):
    from mnemos.service import _protect_private_path
    from mnemos.mcp_server import create_mcp_server

    common = runpy.run_path(str(HERE / "registry_mcp_session.py"))
    helpers = runpy.run_path(str(HERE / "native_processes.py"))
    auth_data = json.loads(AUTH.read_bytes()).get("claudeAiOauth", {})
    assert (
        auth_data.get("accessToken") and auth_data.get("expiresAt", 0) / 1000 - time.time() >= 3600
    ), "Fresh validation login required"
    del auth_data
    home = root / "claude-home"
    home.mkdir()
    _protect_private_path(home)
    bootstrap = root / "native-mcp.py"
    bootstrap.write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        + f"sys.path.insert(0,{str(target)!r})\nimport mnemos\n"
        + f"assert Path(mnemos.__file__).resolve().is_relative_to(Path({str(target)!r}))\n"
        + f"with Path({str(root/'mcp-owner.json')!r}).open('x') as f:json.dump(dict(pid=os.getpid(),module=mnemos.__file__),f)\n"
        + "from mnemos.mcp_server import main\nmain()\n",
        encoding="utf-8",
    )
    catalog = [tool.name for tool in await create_mcp_server().list_tools()]
    assert TOOLS <= set(catalog) and len(catalog) == len(set(catalog))
    config = root / "mcp-config.json"
    save(
        config,
        {"mcpServers": {"mnemos": {"command": str(runtime), "args": ["-I", str(bootstrap)]}}},
    )
    settings = root / "claude-settings.json"
    save(settings, {"disableAllHooks": True})
    session_id = str(uuid.uuid4())
    env = helpers["clean_environment"](
        {
            **{k: v for k, v in os.environ.items() if k.startswith("MNEMOS_")},
            "CLAUDE_CONFIG_DIR": str(home),
            "DISABLE_AUTOUPDATER": "1",
            "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
            "MCP_CONNECTION_NONBLOCKING": "false",
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        }
    )
    alternate_auth = {
        "CLAUDE_BG_AUTH_SNAPSHOT_PATH",
        "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
        "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
        "CLAUDE_CODE_SIMPLE",
    }
    env = {key: value for key, value in env.items() if key.upper() not in alternate_auth}
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        location = home / key.lower()
        location.mkdir()
        env[key] = str(location)
    profile = Path(env["USERPROFILE"])
    env["HOMEDRIVE"] = profile.drive
    env["HOMEPATH"] = str(profile)[len(profile.drive) :]
    assert sha(CLAUDE) == CLAUDE_SHA256, "Claude binary changed since isolation review"
    version = subprocess.check_output(
        [str(CLAUDE), "--version"], cwd=home, env=env, text=True, timeout=15
    ).strip()
    assert version == "2.1.191 (Claude Code)"
    command = [
        str(CLAUDE),
        "--print",
        "--verbose",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--setting-sources",
        "",
        "--settings",
        str(settings),
        "--strict-mcp-config",
        "--mcp-config",
        str(config),
        "--tools",
        "",
        "--allowed-tools",
        ",".join(PREFIX + name for name in sorted(TOOLS)),
        "--disallowed-tools",
        ",".join(PREFIX + name for name in sorted(set(catalog) - TOOLS)),
        "--disable-slash-commands",
        "--permission-mode",
        "default",
        "--session-id",
        session_id,
        "--model",
        "claude-sonnet-4-6",
        "--max-budget-usd",
        "0.50",
        "--system-prompt",
        "Use only the explicitly allowed Mnemos tools. Follow each requested test sequence exactly. Treat structured scope denials as failed operations even if transport succeeds.",
    ]
    bound = [
        Path(__file__),
        HERE / "registry_mcp_session.py",
        HERE / "native_processes.py",
        HERE / "windows_owned_job.py",
        CLAUDE,
        bootstrap,
        config,
        settings,
        root / "settings.toml",
        root / ".mnemos/mnemos.toml",
    ]
    frozen = dict(
        files={str(p): sha(p) for p in bound},
        version=version,
        prompts=PROMPTS,
        command=command,
        session_id=session_id,
        catalog=catalog,
        isolation_environment={
            key: env[key]
            for key in (
                "CLAUDE_CONFIG_DIR",
                "HOME",
                "USERPROFILE",
                "APPDATA",
                "LOCALAPPDATA",
                "HOMEDRIVE",
                "HOMEPATH",
                "CLAUDE_CODE_DISABLE_CLAUDE_MDS",
                "CLAUDE_CODE_DISABLE_AUTO_MEMORY",
                "DISABLE_AUTOUPDATER",
                "ENABLE_CLAUDEAI_MCP_SERVERS",
                "MCP_CONNECTION_NONBLOCKING",
            )
        },
        auth_source_sha256=sha(AUTH),
        started_ns=time.perf_counter_ns(),
    )
    save(root / "native-freeze.json", frozen)
    source_before = sha(AUTH)
    process = auth_copy = reader = None
    all_records, phase_results = [], []
    record = dict(status="running", phases=[], session_id=session_id)
    owner_identity = (owner.descriptor["pid"], owner.owner_thread_id, owner.token)
    project_id = next(iter(registry.projects))
    events = queue.Queue()
    try:
        auth_copy = helpers["_auth_copy"](AUTH, home, root)
        with (
            (root / "native-stdout.jsonl").open("wb") as stdout_log,
            (root / "native-stderr.log").open("wb") as stderr_log,
        ):
            process = subprocess.Popen(
                command,
                cwd=root / "project",
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_log,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            record["pid"] = process.pid
            save(root / "native-result.json", record)

            def consume():
                try:
                    for line in iter(process.stdout.readline, b""):
                        stdout_log.write(line)
                        stdout_log.flush()
                        events.put(json.loads(line))
                except BaseException as exc:
                    events.put(exc)
                finally:
                    events.put(None)

            reader = threading.Thread(target=consume, daemon=True)
            reader.start()
            fact_id, fact_version, before = None, None, None
            for phase in range(3):
                assert process.poll() is None, "Claude exited between phases"
                source_start_line = len(all_records) + 1
                prompt = PROMPTS[phase].format(fact_id=fact_id, version=fact_version)
                prompt += " For each store call, explicitly set scope='project'."
                frame = dict(
                    type="user",
                    message=dict(role="user", content=prompt),
                    session_id=session_id,
                    parent_tool_use_id=None,
                )
                save(root / f"phase-{phase}-input.json", frame)
                deadline = time.monotonic() + 80
                send_frame(process.stdin, frame, deadline)
                observed = []
                while True:
                    event = events.get(timeout=remaining(deadline))
                    remaining(deadline)
                    if event is None:
                        raise RuntimeError("Claude stream ended before a phase result")
                    if isinstance(event, BaseException):
                        raise event
                    assert event.get("session_id", session_id) == session_id
                    observed.append(event)
                    all_records.append(event)
                    if event.get("type") == "result":
                        assert event.get("subtype") == "success" and not event.get("is_error")
                        phase_results.append(event)
                        break
                calls = parse_calls(observed)
                for call in calls:
                    if call["name"] == "mnemos_store":
                        assert call["arguments"].get("scope") == "project"
                        assert call["arguments"].get("scope_id", project_id) in ("", project_id)
                        require_project(call["result"], project_id)
                    elif call["name"] == "mnemos_retrieve":
                        assert call["arguments"]["allowed_scopes"] == "project"
                        assert call["arguments"].get("current_scope", "project") == "project"
                        assert call["arguments"].get("scope_id", project_id) in ("", project_id)
                    elif call["name"] == "mnemos_inspect":
                        require_project(call["result"]["fact"]["metadata"], project_id)
                save(root / f"phase-{phase}-calls.json", calls)
                if phase == 0:
                    assert [c["name"] for c in calls] == [
                        "mnemos_store",
                        "mnemos_consolidate",
                        "mnemos_job_status",
                        "mnemos_inspect",
                    ]
                    assert calls[0]["arguments"]["content"] == MARKER
                    stored, job, inspected = (
                        calls[0]["result"],
                        calls[2]["result"]["job"],
                        calls[3]["result"]["fact"],
                    )
                    assert (
                        stored["status"] == "committed"
                        and job["id"] == stored["job_id"]
                        and job["status"] == "completed"
                    )
                    fact_id, fact_version = inspected["id"], inspected["version"]
                    assert fact_id in job["result"]["fact_ids"] and inspected["content"] == MARKER
                    before = common["snapshot"](root / "memory.sqlite")
                    require_durable_project(root / "memory.sqlite", project_id, 1)
                    assert (
                        before["memory_chunks"]["count"]
                        == before["episodic_events"]["count"]
                        == before["memory_jobs"]["count"]
                        == 1
                    )
                    next(iter(registry.projects.values())).enabled = False
                    registry.save(root / "scopes.json")
                elif phase == 1:
                    assert [c["name"] for c in calls] == [
                        "mnemos_retrieve",
                        "mnemos_store",
                        "mnemos_forget",
                    ]
                    assert calls[1]["arguments"]["content"] == BLOCKED
                    assert calls[2]["arguments"] == {
                        "chunk_id": fact_id,
                        "expected_version": fact_version,
                    }
                    for call in calls:
                        common["require_scope_denial"](
                            json.dumps(call["result"]),
                            store_scope_id=(
                                next(iter(registry.projects))
                                if call["name"] == "mnemos_store"
                                else None
                            ),
                        )
                    after = common["snapshot"](root / "memory.sqlite")
                    save(root / "native-revocation-state.json", dict(before=before, after=after))
                    assert before == after
                    next(iter(registry.projects.values())).enabled = True
                    registry.save(root / "scopes.json")
                else:
                    assert [c["name"] for c in calls] == [
                        "mnemos_inspect",
                        "mnemos_retrieve",
                        "mnemos_store",
                        "mnemos_consolidate",
                        "mnemos_job_status",
                    ]
                    assert (
                        calls[0]["result"]["fact"]["id"] == fact_id
                        and calls[0]["result"]["fact"]["content"] == MARKER
                    )
                    assert fact_id in json.dumps(calls[1]["result"]) and "CEDAR-863" in json.dumps(
                        calls[1]["result"]
                    )
                    assert (
                        calls[2]["arguments"]["content"] == SECOND
                        and calls[2]["result"]["status"] == "committed"
                    )
                    assert (
                        calls[4]["result"]["job"]["id"] == calls[2]["result"]["job_id"]
                        and calls[4]["result"]["job"]["status"] == "completed"
                    )
                assert (
                    owner.descriptor["pid"],
                    owner.owner_thread_id,
                    owner.token,
                ) == owner_identity
                record["phases"].append(
                    dict(
                        phase=phase,
                        calls=len(calls),
                        owner_unchanged=True,
                        source_start_line=source_start_line,
                        source_end_line=len(all_records),
                        mcp_owner=json.loads((root / "mcp-owner.json").read_bytes()),
                    )
                )
                save(root / "native-result.json", record)
            process.stdin.close()
            process.wait(timeout=20)
            reader.join(timeout=5)
            assert not reader.is_alive() and process.returncode == 0
        complete_records = [
            json.loads(line) for line in (root / "native-stdout.jsonl").read_bytes().splitlines()
        ]
        assert len(parse_calls(complete_records)) == 12
        init = [
            r for r in complete_records if r.get("type") == "system" and r.get("subtype") == "init"
        ]
        assert len(init) == 1 and init[0]["mcp_servers"] == [
            {"name": "mnemos", "status": "connected"}
        ]
        assert init[0]["model"] == "claude-sonnet-4-6"
        assert len({p["mcp_owner"]["pid"] for p in record["phases"]}) == 1
        final = common["snapshot"](root / "memory.sqlite")
        require_durable_project(root / "memory.sqlite", project_id, 2)
        assert (
            final["memory_chunks"]["count"]
            == final["episodic_events"]["count"]
            == final["memory_jobs"]["count"]
            == 2
        )
        record.update(
            status="passed",
            final=final,
            phase_results=phase_results,
            initial_fact_id=fact_id,
            stdout_sha256=sha(root / "native-stdout.jsonl"),
            total_native_calls=12,
        )
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        cleanup_errors = []
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        except BaseException as exc:
            cleanup_errors.append(dict(step="terminate", error_type=type(exc).__name__))
        try:
            if auth_copy is not None:
                assert auth_copy.resolve().is_relative_to(home.resolve())
                auth_copy.unlink(missing_ok=True)
        except BaseException as exc:
            cleanup_errors.append(dict(step="remove_auth_copy", error_type=type(exc).__name__))
        checks = {
            "auth_copy_removed": lambda: not (home / ".credentials.json").exists(),
            "auth_source_unchanged": lambda: sha(AUTH) == source_before,
            "inputs_unchanged": lambda: all(sha(p) == h for p, h in frozen["files"].items()),
        }
        for name, check in checks.items():
            try:
                record[name] = check()
            except BaseException as exc:
                record[name] = False
                cleanup_errors.append(dict(step=name, error_type=type(exc).__name__))
        record["cleanup_errors"] = cleanup_errors
        if (
            not all(
                record[key]
                for key in ("auth_copy_removed", "auth_source_unchanged", "inputs_unchanged")
            )
            or cleanup_errors
        ):
            record["status"] = "failed"
        save(root / "native-result.json", record)
    assert record["status"] == "passed"


if __name__ == "__main__":
    runpy.run_path(str(HERE / "registry_mcp_session.py"))["main"](
        exercise_fn=exercise, kind="persistent native Claude registry revocation check"
    )
