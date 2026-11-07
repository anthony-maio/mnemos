"""One isolated Codex check of the private pilot MCP and vetted hook settings."""

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
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "private-pilot-20261106"
WHEEL_SHA = "ecd11eb7ecbb9cc4f14552d9a7253dead7a8348879c35063411bb850abe06036"
LAUNCHER_SHA = "ab2818b0935320e70dbfd35401735ed11b49eb82d4d36668c6e54f4a547ee459"
MODEL = "gpt-6-astra"
FACT = "For this disposable project, the release guide is docs/pine-release.md."
PROMPT = (
    "Pilot integration marker pine-window-529. Use mnemos_retrieve to find this project's "
    "release guide, then mnemos_inspect on the matching fact to verify its source. "
    "Reply with the guide path and the marker. Do not store or change any memory."
)
INPUTS = (
    "scripts/check-native-pilot-codex.py",
    "scripts/check-native-explicit.py",
    "benchmarks/native_hosts/native_processes.py",
    "benchmarks/native_hosts/shared_trace.py",
    "benchmarks/native_hosts/pilot_settings.py",
    "docs/plans/2026-11-07-native-pilot-codex-v1.md",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


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
    authentication = checkpoint["codex_auth_preflight"](args.auth)
    if args.auth.name != "auth.json":
        raise ValueError("Use the Codex auth.json source")
    if sha(args.codex) != checkpoint["CODEX_BINARY_SHA256"]:
        raise ValueError("Codex executable differs from the verified runtime")
    code_mode = args.codex.with_name("codex-code-mode-host.exe")
    if sha(code_mode) != checkpoint["CODE_MODE_BINARY_SHA256"]:
        raise ValueError("Code Mode executable differs from the verified runtime")
    if (
        subprocess.check_output([str(args.codex), "--version"], text=True).strip()
        != checkpoint["CODEX_VERSION"]
    ):
        raise ValueError("Codex version differs")
    pilot_settings = checkpoint["helper"]("benchmarks/native_hosts/pilot_settings.py")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name in INPUTS:
        committed = subprocess.check_output(["git", "show", f"{head}:{name}"], cwd=ROOT)
        if committed.replace(b"\r\n", b"\n") != (ROOT / name).read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError("Commit all acceptance inputs before inference")
    plan = json.loads((args.patches / "plan.json").read_bytes())
    if not (
        sha(pilot / "launch.py")
        == state["launcher_sha256"]
        == plan["launcher_sha256"]
        == LAUNCHER_SHA
    ):
        raise ValueError("Pilot launcher differs from the reviewed source")
    for item in plan["changes"]:
        if sha(args.patches / item["patch"]) != item["patch_sha256"]:
            raise ValueError("Private patch hash differs")
    watched = {
        pilot / name: sha(pilot / name)
        for name in ("launch.py", "settings.toml", "scopes.json", "installation-state.json")
    }
    watched[args.auth] = sha(args.auth)
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
    context = registry.resolve(project, user_profile_id=PROFILE, session_id="seed", host="codex")
    registry.save(root / "scopes.json")
    (root / "launch.py").write_bytes((pilot / "launch.py").read_bytes())
    settings = (pilot / "settings.toml").read_text(encoding="utf-8")
    (root / "settings.toml").write_text(
        settings.replace(pilot.as_posix(), root.as_posix()), encoding="utf-8"
    )
    for parent in (root, project):
        (parent / ".mnemos").mkdir()
        (parent / ".mnemos/mnemos.toml").write_text("", encoding="utf-8")
    phase = root / "native"
    home = phase / "codex-home"
    home.mkdir(parents=True)
    _protect_private_path(home)
    config_text = pilot_settings.codex_patch_settings(
        (args.patches / "codex-config.patch").read_text(encoding="utf-8"),
        python=Path(sys.executable),
        launcher=pilot / "launch.py",
        disposable_launcher=root / "launch.py",
    )
    config_text += '\n[windows]\nsandbox = "unelevated"\n'
    (home / "config.toml").write_text(config_text, encoding="utf-8")
    descriptor = root / "runtime/runtime.json"
    record: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "runner_commit": head,
        "source_commit": "6d0e5ba69f39d55df8623830cb682e14a5b2d607",
        "wheel_sha256": WHEEL_SHA,
        "input_hashes": {name: sha(ROOT / name) for name in INPUTS},
        "launcher_sha256": sha(root / "launch.py"),
        "generated_configuration_hashes": {
            name: sha(root / name)
            for name in (
                "launch.py",
                "settings.toml",
                "scopes.json",
                ".mnemos/mnemos.toml",
                "project/.mnemos/mnemos.toml",
                "native/codex-home/config.toml",
            )
        },
        "private_plan_sha256": sha(args.patches / "plan.json"),
        "watched_file_hashes_before": {str(path): digest for path, digest in watched.items()},
        "authentication_preflight": authentication,
        "project_id": context.project_id,
        "profile": PROFILE,
        "service_pids": [],
        "host_version": checkpoint["CODEX_VERSION"],
        "host_executable_sha256": sha(args.codex),
        "code_mode_executable_sha256": sha(code_mode),
        "model": MODEL,
        "hook_trust": "one-run override after exact six-command validation; no persisted trust",
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
                    "codex",
                    "service-stop",
                ],
                cwd=project,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            record.setdefault("service_stop_results", []).append(
                {
                    "returncode": stopped.returncode,
                    "stdout": stopped.stdout,
                    "stderr": stopped.stderr,
                }
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
            args=["-I", str(root / "launch.py"), "mcp", "codex"],
            cwd=str(project),
            env=environment,
        )
        with (root / "seed-mcp.log").open("w", encoding="utf-8") as errors:
            async with stdio_client(params, errlog=errors) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    catalog = await session.list_tools()
                    inspect_tool = next(
                        tool for tool in catalog.tools if tool.name == "mnemos_inspect"
                    )
                    if "chunk_id" not in inspect_tool.inputSchema.get("required", []):
                        raise ValueError("Installed inspection contract differs")

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
                    deadline = time.monotonic() + 120
                    while True:
                        job = (await call("mnemos_job_status", {"job_id": stored["job_id"]}))["job"]
                        if job["status"] in {"completed", "failed"} or time.monotonic() >= deadline:
                            break
                        await asyncio.sleep(0.25)
                    if job["status"] != "completed" or len(job["result"]["fact_ids"]) != 1:
                        raise ValueError("Controller seed did not complete")
                    record["seed_fact_id"] = job["result"]["fact_ids"][0]
                    record["seed_job"] = job
                    record["catalog"] = [tool.name for tool in catalog.tools]
                    remember_owner()
                    # Close the owner before the Windows stdio client's process tree.
                    await asyncio.to_thread(stop_service)
        save(report, record)
        allowed = ["mnemos_retrieve", "mnemos_inspect"]
        disabled = (
            "shell_tool",
            "shell_snapshot",
            "unified_exec",
            "apps",
            "multi_agent",
            "multi_agent_v2",
            "memories",
            "goals",
            "plugins",
            "remote_plugin",
            "browser_use",
            "browser_use_external",
            "computer_use",
            "image_generation",
            "artifact",
            "workspace_dependencies",
            "skill_search",
        )
        command = [
            str(args.codex),
            "--ask-for-approval",
            "never",
            "--enable",
            "code_mode_host",
            "--enable",
            "hooks",
            *(value for feature in disabled for value in ("--disable", feature)),
            "-c",
            'web_search="disabled"',
            "-c",
            'cli_auth_credentials_store="file"',
            "-c",
            'model_reasoning_effort="high"',
            "-c",
            "mcp_servers.mnemos.enabled_tools=" + json.dumps(allowed),
            "-c",
            'mcp_servers.mnemos.default_tools_approval_mode="prompt"',
            *(
                value
                for tool in allowed
                for value in ("-c", f'mcp_servers.mnemos.tools.{tool}.approval_mode="approve"')
            ),
            "exec",
            "--sandbox",
            "read-only",
            "--dangerously-bypass-hook-trust",
            "--cd",
            str(project),
            "--json",
            "--color",
            "never",
            "--model",
            MODEL,
            PROMPT,
        ]
        trace_root = phase / "rollout-trace"
        trace_root.mkdir()
        _protect_private_path(trace_root)
        env = {
            **environment,
            "CODEX_HOME": str(home),
            "CODEX_ROLLOUT_TRACE_ROOT": str(trace_root.resolve()),
            "PATH": str(args.codex.parent) + os.pathsep + environment.get("PATH", ""),
        }
        process_result: dict[str, Any] = {
            "status": "running",
            "allowed_tools": allowed,
            "disabled_features": list(disabled),
            "command": command[:-1] + ["<frozen synthetic prompt>"],
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "max_seconds": 180,
            "requested_model": MODEL,
            "reasoning_effort": "high",
        }
        (phase / "prompt.txt").write_text(PROMPT, encoding="utf-8")
        auth_copy = None
        process = None
        started = time.monotonic()
        try:
            if any(
                sha(root / name) != digest
                for name, digest in record["generated_configuration_hashes"].items()
            ):
                raise ValueError("Vetted configuration changed before native execution")
            checkpoint["codex_auth_preflight"](args.auth)
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
        if process_result["status"] != "completed" or not process_result["auth_copy_removed"]:
            raise ValueError("Native Codex process or authentication cleanup failed")
        rows = [
            json.loads(line)
            for line in (phase / "stdout.jsonl").read_bytes().splitlines()
            if line.strip()
        ]
        starts = [row for row in rows if row.get("type") == "thread.started"]
        terminal = [row for row in rows if row.get("type") == "turn.completed"]
        if (
            len(starts) != 1
            or not isinstance(starts[0].get("thread_id"), str)
            or not starts[0]["thread_id"]
            or len(terminal) != 1
            or rows[-1].get("type") != "turn.completed"
            or any(row.get("type") in {"error", "turn.failed"} for row in rows)
        ):
            raise ValueError("Native Codex stream lacks a successful final turn")
        session_id = starts[0]["thread_id"]
        sessions = list((home / "sessions").rglob("*.jsonl"))
        if len(sessions) != 1:
            raise ValueError("Require one retained native session")
        session_file = sessions[0]
        session_rows = [
            json.loads(line) for line in session_file.read_bytes().splitlines() if line.strip()
        ]
        if [
            row.get("payload", {}).get("id")
            for row in session_rows
            if row.get("type") == "session_meta"
        ] != [session_id]:
            raise ValueError("Native session identities differ")
        models = native._codex_observed_models(home)
        if (
            not models
            or any(model != MODEL for model in models)
            or native._codex_exec_user_messages(home) != [PROMPT]
        ):
            raise ValueError("Native model or prompt observation differs")
        process_result.update(
            session_id=session_id,
            session_file=str(session_file),
            session_sha256=sha(session_file),
            observed_models=models,
            model_verified=True,
            user_prompt_verified=True,
            reported_usage=terminal[0].get("usage"),
        )
        save(phase / "process.json", process_result)
        record["native"] = {
            key: process_result[key]
            for key in [
                "session_id",
                "observed_models",
                "reported_usage",
                "model_verified",
                "user_prompt_verified",
            ]
        }
        provenance = trace.TraceProvenance(
            "codex", process_result["pid"], session_id, str(session_file), sha(session_file)
        )
        traces = list(trace_root.iterdir())
        if len(traces) != 1 or not traces[0].is_dir():
            raise ValueError("Require one owned native rollout trace")
        calls, trace_hashes = trace.parse_codex_code_mode_trace(
            traces[0], provenance=provenance, model=MODEL
        )
        save(phase / "rollout-trace-hashes.json", trace_hashes)
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
            and call.arguments.get("chunk_id") == record["seed_fact_id"]
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
        messages = [
            row["item"]["text"]
            for row in rows
            if row.get("type") == "item.completed"
            and row.get("item", {}).get("type") == "agent_message"
            and isinstance(row["item"].get("text"), str)
        ]
        if not messages:
            raise ValueError("Native Codex final answer missing")
        final = messages[-1]
        if "docs/pine-release.md" not in final or "pine-window-529" not in final:
            raise ValueError("Native final answer omitted the guide or marker")
        with sqlite3.connect((root / "memory.sqlite").as_uri() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            events = [
                dict(row)
                for row in db.execute("SELECT * FROM episodic_events WHERE source='codex'")
            ]
            captures = [
                row for row in events if not json.loads(row["metadata"]).get("explicit_remember")
            ]
            if not any(row["role"] == "user" and row["content"] == PROMPT for row in captures):
                raise ValueError("Actual user hook capture missing")
            if not any(
                row["role"] == "assistant"
                and "docs/pine-release.md" in row["content"]
                and "pine-window-529" in row["content"]
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
        record["watched_file_hashes_after"] = {
            str(path): sha(path) if path.is_file() else None for path in watched
        }
        watched_unchanged = (
            record["watched_file_hashes_before"] == record["watched_file_hashes_after"]
        )
        if not watched_unchanged:
            cleanup_errors.append("WatchedFileChanged")
        if any(
            not (root / name).is_file() or sha(root / name) != digest
            for name, digest in record["generated_configuration_hashes"].items()
        ):
            cleanup_errors.append("GeneratedConfigurationChanged")
        if any(sha(ROOT / name) != digest for name, digest in record["input_hashes"].items()):
            cleanup_errors.append("AcceptanceInputChanged")
        if (
            list(root.rglob("auth.json"))
            or (pilot / "memory.sqlite").exists()
            or (pilot / "runtime").exists()
        ):
            cleanup_errors.append("IsolationFailure")
        record.update(
            cleanup="passed" if not cleanup_errors else "failed",
            cleanup_errors=cleanup_errors,
            source_credentials_and_pilot_files_unchanged=watched_unchanged,
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
    for name in ("pilot-root", "patches", "output", "codex", "auth"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = asyncio.run(run(parser.parse_args()))
    raise SystemExit(0 if result["status"] == "passed" else 1)
