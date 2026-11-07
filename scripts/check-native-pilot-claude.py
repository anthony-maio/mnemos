"""One isolated native check of the private pilot's project MCP and hook settings."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sqlite3
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "private-pilot-20261106"
WHEEL_SHA = "ecd11eb7ecbb9cc4f14552d9a7253dead7a8348879c35063411bb850abe06036"
PREFIX = "mcp__mnemos__"
FACT = "For this disposable project, the release guide is docs/cedar-release.md."
PROMPT = (
    "Pilot integration marker cedar-window-741. Use mnemos_retrieve to find this project's "
    "release guide, then mnemos_inspect on the matching fact to verify its source. "
    "Reply with the guide path and the marker. Do not store or change any memory."
)
INPUTS = (
    "scripts/check-native-pilot-claude.py",
    "scripts/check-native-explicit.py",
    "benchmarks/native_hosts/native_processes.py",
    "benchmarks/native_hosts/shared_trace.py",
    "docs/plans/2026-11-07-native-pilot-claude-v1.md",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def patch_json(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(line.startswith("-") and not line.startswith("---") for line in lines):
        raise ValueError("Expected an additions-only new project file")
    return dict(
        json.loads(
            "\n".join(
                line[1:] for line in lines if line.startswith("+") and not line.startswith("+++")
            )
        )
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import mnemos
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mnemos.scopes import ScopeRegistry
    from mnemos.service import _protect_private_path

    pilot = args.pilot_root.resolve()
    root = args.output.resolve()
    if root.exists() or root.is_relative_to(ROOT) or root.is_relative_to(pilot):
        raise ValueError("Use a fresh external output directory outside the pilot")
    if Path(sys.executable).resolve() != pilot / "venv/Scripts/python.exe":
        raise ValueError("Use the persistent installed pilot interpreter")
    if not Path(mnemos.__file__).resolve().is_relative_to(pilot / "venv"):
        raise ValueError("Package escaped the installed pilot environment")
    state = json.loads((pilot / "installation-state.json").read_bytes())
    if state["status"] != "PREPARED_INACTIVE" or state["wheel_sha256"] != WHEEL_SHA:
        raise ValueError("Persistent installation is not the verified inactive artifact")
    wheel = pilot / "artifacts/mnemos_memory-0.7.0b1-py3-none-any.whl"
    if sha(wheel) != WHEEL_SHA:
        raise ValueError("Wheel hash differs")
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("mnemos/") and not name.endswith("/"):
                if (Path(mnemos.__file__).parent.parent / name).read_bytes() != archive.read(name):
                    raise ValueError("Installed package bytes differ: " + name)
    checkpoint = runpy.run_path(
        str(ROOT / "scripts/check-native-explicit.py"), run_name="pilot_helpers"
    )
    native = checkpoint["helper"]("benchmarks/native_hosts/native_processes.py")
    trace = checkpoint["helper"]("benchmarks/native_hosts/shared_trace.py")
    authentication = checkpoint["auth_preflight"](args.auth)
    prior = json.loads(
        (ROOT / "docs/validation/2026-11-06/native-claude-v5-receipt.json").read_bytes()
    )
    if sha(args.claude) != prior["host_executable_sha256"]:
        raise ValueError("Claude executable differs from the verified runtime")
    if (
        subprocess.check_output([str(args.claude), "--version"], text=True).strip()
        != "2.1.191 (Claude Code)"
    ):
        raise ValueError("Claude version differs")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name in INPUTS:
        committed = subprocess.check_output(["git", "show", f"{head}:{name}"], cwd=ROOT)
        if committed.replace(b"\r\n", b"\n") != (ROOT / name).read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError("Commit all acceptance inputs before inference")
    plan = json.loads((args.patches / "plan.json").read_bytes())
    for item in plan["changes"]:
        if sha(args.patches / item["patch"]) != item["patch_sha256"]:
            raise ValueError("Private patch hash differs")
    watched = {
        pilot / name: sha(pilot / name)
        for name in ("launch.py", "settings.toml", "scopes.json", "installation-state.json")
    }
    watched[args.auth] = sha(args.auth)
    normal_auth = Path.home() / ".claude/.credentials.json"
    watched[normal_auth] = sha(normal_auth)
    if (pilot / "memory.sqlite").exists() or (pilot / "runtime").exists():
        raise ValueError("The live pilot must remain inactive")
    root.mkdir(parents=True)
    _protect_private_path(root)
    project = root / "project"
    project.mkdir()
    environment = native.clean_environment({})
    environment = {
        key: value for key, value in environment.items() if not key.upper().startswith("GIT_")
    }
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=project,
        env=environment,
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    registry = ScopeRegistry()
    registry.register_project(project, user_profile_id=PROFILE, enabled=True)
    context = registry.resolve(
        project, user_profile_id=PROFILE, session_id="seed", host="claude-code"
    )
    registry.save(root / "scopes.json")
    (root / "launch.py").write_bytes((pilot / "launch.py").read_bytes())
    settings = (pilot / "settings.toml").read_text(encoding="utf-8")
    (root / "settings.toml").write_text(
        settings.replace(pilot.as_posix(), root.as_posix()), encoding="utf-8"
    )
    for parent in (root, project):
        (parent / ".mnemos").mkdir()
        (parent / ".mnemos/mnemos.toml").write_text("", encoding="utf-8")
    mcp = patch_json(args.patches / "project-1-mcp.patch")
    entry = mcp["mcpServers"]["mnemos"]
    if (
        Path(entry["command"]).resolve() != Path(sys.executable).resolve()
        or Path(entry["args"][1]).resolve() != pilot / "launch.py"
    ):
        raise ValueError("Draft MCP command differs from the installed launcher")
    entry["args"][1] = (root / "launch.py").as_posix()
    save(project / ".mcp.json", mcp)
    hooks = patch_json(args.patches / "project-1-claude-hooks.patch")
    hooks = json.loads(
        json.dumps(hooks).replace((pilot / "launch.py").as_posix(), (root / "launch.py").as_posix())
    )
    (project / ".claude").mkdir()
    save(project / ".claude/settings.local.json", hooks)
    descriptor = root / "runtime/runtime.json"
    record: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "runner_commit": head,
        "source_commit": "6d0e5ba69f39d55df8623830cb682e14a5b2d607",
        "wheel_sha256": WHEEL_SHA,
        "input_hashes": {name: sha(ROOT / name) for name in INPUTS},
        "launcher_sha256": sha(root / "launch.py"),
        "private_plan_sha256": sha(args.patches / "plan.json"),
        "authentication_preflight": authentication,
        "project_id": context.project_id,
        "profile": PROFILE,
        "service_pids": [],
        "host_version": "2.1.191 (Claude Code)",
        "model": "claude-sonnet-4-6",
        "live_configuration_changed": False,
    }
    report = root / "report.json"

    def remember_owner() -> None:
        if descriptor.exists():
            pid = json.loads(descriptor.read_bytes())["pid"]
            if pid not in record["service_pids"]:
                record["service_pids"].append(pid)

    def stop_service() -> None:
        remember_owner()
        if descriptor.exists():
            stopped = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(root / "launch.py"),
                    "cli",
                    "claude-code",
                    "service-stop",
                ],
                cwd=project,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if stopped.returncode:
                raise ValueError("Owned service stop failed")
        deadline = time.monotonic() + 20
        while descriptor.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        if descriptor.exists():
            raise ValueError("Owned service descriptor remains")

    save(report, record)
    try:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-I", str(root / "launch.py"), "mcp", "claude-code"],
            cwd=str(project),
            env=environment,
        )
        with (root / "seed-mcp.log").open("w", encoding="utf-8") as errors:
            async with stdio_client(params, errlog=errors) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    catalog = await session.list_tools()

                    async def call(name: str, payload: dict[str, Any]) -> dict[str, Any]:
                        result = await session.call_tool(name, payload)
                        if result.isError:
                            raise ValueError("Controller seed MCP failed")
                        return dict(
                            json.loads(
                                "".join(item.text for item in result.content if item.type == "text")
                            )
                        )

                    stored = await call("mnemos_store", {"content": FACT})
                    await call("mnemos_consolidate", {})
                    job = (await call("mnemos_job_status", {"job_id": stored["job_id"]}))["job"]
                    if job["status"] != "completed" or len(job["result"]["fact_ids"]) != 1:
                        raise ValueError("Controller seed did not complete")
                    record["seed_fact_id"] = job["result"]["fact_ids"][0]
                    record["seed_job"] = job
                    record["catalog"] = [tool.name for tool in catalog.tools]
                    remember_owner()
        stop_service()
        phase = root / "native"
        phase.mkdir()
        home = phase / "claude-home"
        home.mkdir()
        _protect_private_path(home)
        session_id = str(uuid4())
        allowed = [PREFIX + name for name in ["mnemos_retrieve", "mnemos_inspect"]]
        denied = [PREFIX + name for name in record["catalog"] if PREFIX + name not in allowed]
        command = [
            str(args.claude),
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--include-hook-events",
            "--setting-sources",
            "project,local",
            "--tools",
            "",
            "--allowed-tools",
            ",".join(allowed),
            "--disallowed-tools",
            ",".join(denied),
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
            "Use only the two permitted Mnemos tools. Verify the requested fact and source.",
            PROMPT,
        ]
        env = native.clean_environment(
            {
                "CLAUDE_CONFIG_DIR": str(home),
                "DISABLE_AUTOUPDATER": "1",
                "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
            }
        )
        process_result: dict[str, Any] = {
            "status": "running",
            "session_id": session_id,
            "allowed_tools": allowed,
            "command": command[:-1] + ["<frozen synthetic prompt>"],
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        }
        (phase / "prompt.txt").write_text(PROMPT, encoding="utf-8")
        auth_copy = None
        process = None
        started = time.monotonic()
        try:
            auth_copy = native._auth_copy(args.auth, home, root)
            with (
                (phase / "stdout.jsonl").open("wb") as stdout,
                (phase / "stderr.log").open("wb") as stderr,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=project,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                process_result["pid"] = process.pid
                save(phase / "process.json", process_result)
                try:
                    await asyncio.to_thread(process.wait, timeout=180)
                    process_result["timed_out"] = False
                except subprocess.TimeoutExpired:
                    process_result["timed_out"] = True
                    native.terminate_owned_process(process)
                process_result.update(
                    returncode=process.returncode,
                    status=(
                        "completed"
                        if process.returncode == 0 and not process_result["timed_out"]
                        else "failed"
                    ),
                )
        finally:
            native._cleanup_phase(phase, process_result, process, auth_copy, started)
            remember_owner()
        rows = [
            json.loads(line)
            for line in (phase / "stdout.jsonl").read_bytes().splitlines()
            if line.strip()
        ]
        record["native"] = checkpoint["verify_native_result"](process_result, rows)
        provenance = trace.TraceProvenance(
            "claude-code",
            process_result["pid"],
            session_id,
            str(phase / "stdout.jsonl"),
            sha(phase / "stdout.jsonl"),
        )
        calls = trace.parse_claude_tool_records(
            trace.load_source_records(phase / "stdout.jsonl", provenance=provenance),
            provenance=provenance,
            tool_prefix=PREFIX,
        )
        if {call.name for call in calls} != {"mnemos_retrieve", "mnemos_inspect"}:
            raise ValueError("Native read/inspection calls missing")
        if not any(
            call.name == "mnemos_retrieve"
            and any(
                fact["id"] == record["seed_fact_id"] and fact["content"] == FACT
                for fact in call.result.get("facts", [])
            )
            for call in calls
        ):
            raise ValueError("Native retrieval did not return the seeded fact")
        if not any(
            call.name == "mnemos_inspect"
            and call.arguments.get("fact_id") == record["seed_fact_id"]
            for call in calls
        ):
            raise ValueError("Native inspection targeted a different fact")
        record["native_tool_calls"] = [
            {
                "name": call.name,
                "call_id": call.call_id,
                "arguments": dict(call.arguments),
                "result": dict(call.result),
            }
            for call in calls
        ]
        final = next(row["result"] for row in rows if row.get("type") == "result")
        if "docs/cedar-release.md" not in final or "cedar-window-741" not in final:
            raise ValueError("Native final answer omitted the guide or marker")
        with sqlite3.connect((root / "memory.sqlite").as_uri() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            events = [
                dict(row)
                for row in db.execute("SELECT * FROM episodic_events WHERE source='claude-code'")
            ]
            captures = [
                row for row in events if not json.loads(row["metadata"]).get("explicit_remember")
            ]
            if not any(row["role"] == "user" and row["content"] == PROMPT for row in captures):
                raise ValueError("Actual user hook capture missing")
            if not any(
                row["role"] == "assistant"
                and "docs/cedar-release.md" in row["content"]
                and "cedar-window-741" in row["content"]
                for row in captures
            ):
                raise ValueError("Actual assistant hook capture missing")
            if not all(
                row["scope"] == "project"
                and row["scope_id"] == context.project_id
                and row["user_profile_id"] == PROFILE
                and row["session_id"] == session_id
                for row in captures
            ):
                raise ValueError("Physical capture identity differs from native session")
            if (
                db.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
                or db.execute("PRAGMA foreign_key_check").fetchall()
            ):
                raise ValueError("SQLite integrity failed")
            record["captures"] = captures
            record["fact_metadata"] = [
                dict(row) for row in db.execute("SELECT id,metadata FROM memory_chunks")
            ]
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        cleanup_errors = []
        try:
            stop_service()
        except Exception as error:
            cleanup_errors.append(type(error).__name__)
        if any(sha(path) != digest for path, digest in watched.items()):
            cleanup_errors.append("WatchedFileChanged")
        if (
            list(root.rglob(".credentials.json"))
            or (pilot / "memory.sqlite").exists()
            or (pilot / "runtime").exists()
        ):
            cleanup_errors.append("IsolationFailure")
        record.update(
            cleanup="passed" if not cleanup_errors else "failed",
            cleanup_errors=cleanup_errors,
            source_credentials_and_pilot_files_unchanged=not cleanup_errors,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        if cleanup_errors:
            record["status"] = "failed"
        save(report, record)
        print(
            json.dumps(
                {key: record.get(key) for key in ["status", "error_type", "error", "cleanup"]}
            ),
            flush=True,
        )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot-root", "patches", "output", "claude", "auth"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = asyncio.run(run(parser.parse_args()))
    raise SystemExit(0 if result["status"] == "passed" else 1)
