"""Prepare and execute the fixed, disposable replay acknowledgement comparison.

Prepare with the package runtime; supervise with the Windows job helper runtime.
The independent auditor reads durable records after the supervisor exits.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import shutil
import socket
import subprocess
import sys
import time
import uuid

ARMS = ("baseline", "candidate")
NATIVE_SHA = "b154c81345e4a7473086b946599baf2535daafc627133b1131984d7091914683"
COMMITS = (
    "7a77e40f4ec8dd5b2fbf47ff0b1e3f0862e0129d",
    "ec4a36b262be61c16d807a2b39006bcad7f8367c",
)
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def schedule(project):
    rows = []
    for index in range(55):
        for arm in ARMS if index % 2 == 0 else ARMS[::-1]:
            for position in ("primer", "successor"):
                rows.append(
                    {
                        "key": f"{index}-{arm}-{position}",
                        "round": index,
                        "arm": arm,
                        "position": position,
                        "warmup": index < 5,
                        "payload": {
                            "hook_event_name": "Stop",
                            "cwd": str(project),
                            "session_id": "ack-comparison-session",
                            "turn_id": f"round-{index}-{position}",
                            "timestamp": "2026-11-08T00:00:00+00:00",
                            "last_assistant_message": f"Retain acknowledgement comparison round {index} {position}.",
                        },
                    }
                )
    return rows


def versions():
    return {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}


def environment(root, arm, target):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.upper().startswith("MNEMOS_") and k.upper() not in ("PYTHONPATH", "PYTHONHOME")
    }
    folder = root / arm
    env.update(
        MNEMOS_CONFIG_PATH=str(root / "project/settings.toml"),
        MNEMOS_SCOPE_REGISTRY=str(root / "scopes.json"),
        MNEMOS_RUNTIME_DIR=str(folder),
        MNEMOS_RUNTIME_DESCRIPTOR=str(folder / "memory.json"),
        MNEMOS_USER_PROFILE_ID="ack-comparison",
        MNEMOS_AUTO_START="false",
        MNEMOS_HOOK_WORKER_DESCRIPTOR=str(folder / "worker.json"),
        MNEMOS_COMPARISON_ROOT=str(root),
        MNEMOS_COMPARISON_ARM=arm,
        PYTHONPATH=str(target),
        PYTHONIOENCODING="utf-8",
    )
    return env


def prepare(root, packages, native, protocol, git_manifest):
    if os.name != "nt":
        raise RuntimeError("This comparison requires Windows")
    bindings = read(packages)
    assert Path(sys.executable).resolve() == Path(bindings["runtime_python"]).resolve()
    assert sha(native) == NATIVE_SHA
    assert bindings["wheel_changed_members"] == [
        "mnemos/client.py",
        "mnemos_memory-0.7.0b1.dist-info/RECORD",
    ]
    root.mkdir(exist_ok=False)
    sys.path.insert(0, bindings["arms"]["baseline"]["installed_target"])
    from mnemos.scopes import ScopeRegistry
    from mnemos.service import _protect_private_path
    from mnemos.hook_client import environment_fingerprint

    _protect_private_path(root)
    project = root / "project"
    project.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", str(project)],
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    (project / "settings.toml").write_text("# Disposable comparison defaults.\n")
    (project / ".mnemos").mkdir()
    (project / ".mnemos/mnemos.toml").write_text("# Stop parent configuration discovery here.\n")
    registry = ScopeRegistry()
    registry.register_project(project, user_profile_id="ack-comparison", enabled=True)
    registry.save(root / "scopes.json")
    context = registry.resolve(
        project, user_profile_id="ack-comparison", session_id="ack-comparison-session", host="codex"
    )
    files = {}

    def bind(path):
        files[str(Path(path).resolve())] = sha(path)

    known_git = read(git_manifest)
    git_executable = Path(shutil.which("git")).resolve()
    assert git_executable == Path(known_git["selected"]).resolve()
    bind(git_manifest)
    for path, digest in known_git["hashes"].items():
        assert sha(path) == digest
        bind(path)
    import _sqlite3

    bind(Path(_sqlite3.__file__))
    for path in (
        Path(__file__),
        HERE / "replay_ack_owner.py",
        HERE / "replay_ack_audit.py",
        HERE / "windows_owned_job.py",
        HERE / "windows_process_identity.py",
        native,
        protocol,
        packages,
        project / "settings.toml",
        project / ".mnemos/mnemos.toml",
        project / ".git/config",
        root / "scopes.json",
        Path(sys.executable),
        Path(sys._base_executable),
    ):
        bind(path)
    dependency_files = 0
    selected = (
        "numpy",
        "pydantic",
        "pydantic_core",
        "httpx",
        "httpcore",
        "anyio",
        "h11",
        "platformdirs",
        "typing_extensions",
        "annotated-types",
        "typing-inspection",
    )
    for name in selected:
        dist = importlib.metadata.distribution(name)
        for member in dist.files or ():
            path = Path(dist.locate_file(member))
            if path.is_file() and path.suffix != ".pyc":
                bind(path)
                dependency_files += 1
    for arm, commit in zip(ARMS, COMMITS):
        info = bindings["arms"][arm]
        assert info["source_commit"] == commit
        for name, digest in info["package_files"].items():
            path = Path(info["installed_target"]) / name
            assert sha(path) == digest
            bind(path)
        for name, digest in info["artifacts"].items():
            path = packages.parent / arm / "dist" / name
            assert sha(path) == digest
            bind(path)
        (root / arm).mkdir()
    # Also bind the shared runtime's pre-existing package: the target override
    # must never mutate it, even on a failed native fallback.
    shared = Path(sys.executable).parent.parent / "Lib/site-packages/mnemos"
    for path in shared.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            bind(path)
    env_hashes = {}
    for arm in ARMS:
        env = environment(root, arm, bindings["arms"][arm]["installed_target"])
        env_hashes[arm] = environment_fingerprint(env)
    # distributions() is captured without an arm target on sys.path, matching
    # the runner process before it starts the arm-specific owners.
    sys.path.pop(0)
    freeze = dict(
        schema_version=1,
        job_name="Local\\MnemosReplayComparison-" + uuid.uuid4().hex,
        arms=bindings["arms"],
        context=context.model_dump(),
        runtime_python=str(Path(sys.executable).resolve()),
        base_python=str(Path(sys._base_executable).resolve()),
        native=str(native),
        git_executable=str(git_executable),
        git_runtime_manifest=str(git_manifest),
        git_runtime_limit="Previously observed files only, not dynamic loader closure or cache approval",
        protocol=str(protocol),
        files=files,
        dependencies=versions(),
        selected_dependency_files=dependency_files,
        environment_sha256=env_hashes,
        schedule=schedule(project),
        prepared_ns=time.perf_counter_ns(),
        timeout_seconds=30,
        round_count=55,
        warmup_rounds=5,
        measured_rounds=50,
    )
    save(root / "freeze.json", freeze)
    save(
        root / "ledger.json",
        [{"key": row["key"], "status": "unexecuted"} for row in freeze["schedule"]],
    )
    print(
        json.dumps(
            {"prepared": str(root), "commands": 220, "freeze_sha256": sha(root / "freeze.json")}
        ),
        flush=True,
    )


def unchanged(freeze):
    result = {}
    for path, digest in freeze["files"].items():
        try:
            result[path] = Path(path).is_file() and sha(path) == digest
        except OSError:
            result[path] = False
    return result


def native_command(command, payload, env, cwd):
    start = time.perf_counter_ns()
    try:
        result = subprocess.run(
            command,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            env=env,
            cwd=cwd,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired as exc:
        end = time.perf_counter_ns()

        def text(value):
            return (
                value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
            )

        return dict(
            status="timeout",
            start_ns=start,
            end_ns=end,
            stdout=text(exc.stdout),
            stderr=text(exc.stderr),
            returncode=None,
        )
    except Exception as exc:
        return dict(
            status="failed",
            start_ns=start,
            end_ns=time.perf_counter_ns(),
            error_type=type(exc).__name__,
            returncode=None,
            stdout="",
            stderr="",
        )
    # End timing before interpreting or writing output.
    end = time.perf_counter_ns()
    return dict(
        status="executed",
        start_ns=start,
        end_ns=end,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def run(root):
    freeze_path = root / "freeze.json"
    freeze = read(freeze_path)
    assert not (root / "execution.json").exists(), "Comparison already attempted; never restart"
    execution = dict(status="starting", freeze_sha256=sha(freeze_path), cleanup={}, owners={})
    save(root / "execution.json", execution)
    ledger = None
    owners = {}
    identity = None
    helper = HERE / "replay_ack_owner.py"

    def wait(path, process, seconds=40):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if path.exists():
                return read(path)
            if process.poll() is not None:
                raise RuntimeError("Comparison owner exited; inspect its log")
            time.sleep(0.01)
        raise TimeoutError(f"No owner result at {path.name}")

    def send(arm, request):
        entry = owners[arm]
        save(root / arm / f"control-{entry['sequence']}.json", request)
        entry["sequence"] += 1

    def barrier(arm, label, expected):
        send(arm, dict(operation="drain", expected=expected, round=label))
        state = wait(root / arm / f"barrier-{label}.json", owners[arm]["process"])
        assert state["events"] == state["jobs"] == expected
        assert state["pending"] == state["ack"] == state["replay"] == 0 and state["pumps_idle"]

    try:
        ledger = read(root / "ledger.json")
        assert ledger == [{"key": row["key"], "status": "unexecuted"} for row in freeze["schedule"]]
        identity = runpy.run_path(str(HERE / "windows_process_identity.py"))
        assert Path(sys.executable).resolve() == Path(freeze["runtime_python"]).resolve()
        assert versions() == freeze["dependencies"]
        assert Path(shutil.which("git")).resolve() == Path(freeze["git_executable"]).resolve()
        assert all(unchanged(freeze).values())
        sys.path.insert(0, freeze["arms"]["baseline"]["installed_target"])
        from mnemos.hook_client import environment_fingerprint

        for arm in ARMS:
            env = environment(root, arm, freeze["arms"][arm]["installed_target"])
            assert environment_fingerprint(env) == freeze["environment_sha256"][arm]
            with (root / arm / "owner.log").open("xb") as log:
                process = subprocess.Popen(
                    [sys.executable, "-I", str(helper)],
                    env=env,
                    cwd=root / "project",
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            owners[arm] = dict(process=process, environment=env, sequence=0)
            ready = wait(root / arm / "ready.json", process)
            proof = identity["verify_owner"](process, ready["pid"], helper)
            assert ready["environment_sha256"] == freeze["environment_sha256"][arm]
            assert ready["full_sync"] and ready["integrity_check"] == "ok"
            owners[arm]["ready"] = ready
            execution["owners"][arm] = dict(proof=proof, ready=ready)
        assert (
            len({o["ready"][k] for o in owners.values() for k in ("memory_port", "worker_port")})
            == 4
        )
        execution.update(status="running", started_ns=time.perf_counter_ns())
        save(root / "execution.json", execution)
        command = [freeze["native"], "codex", "Stop", "--python", sys.executable]
        for offset in range(0, len(ledger), 2):
            pair = freeze["schedule"][offset : offset + 2]
            arm, index = pair[0]["arm"], pair[0]["round"]
            # Persist uncertainty before launch. There is no file write, drain,
            # response inspection or sleep between primer and successor.
            for item in ledger[offset : offset + 2]:
                item["status"] = "in_flight"
            save(root / "ledger.json", ledger)
            failed = False
            for n, row in enumerate(pair):
                result = native_command(
                    command, row["payload"], owners[arm]["environment"], root / "project"
                )
                ledger[offset + n].update(result)
                if result["status"] != "executed":
                    failed = True
                    if n == 0:
                        ledger[offset + 1]["status"] = "unexecuted"
                    break
            save(root / "ledger.json", ledger)
            if failed:
                raise RuntimeError(
                    "Native command failed or timed out; schedule aborted without retry"
                )
            for item in ledger[offset : offset + 2]:
                assert item["returncode"] == 0 and json.loads(item["stdout"]) == {"continue": True}
                assert (
                    item["stderr"].strip()
                    == "mnemos hook diagnostic host=codex event=Stop status=degraded"
                )
            barrier(arm, f"{index}-after", 2 * (index + 1))
            if offset % 4 == 2:
                for other in ARMS:
                    barrier(other, f"{index}-end", 2 * (index + 1))
                if (index + 1) % 10 == 0:
                    print(json.dumps({"rounds_completed": index + 1}), flush=True)
        execution["schedule_completed"] = True
    except BaseException as exc:
        execution.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        for arm, entry in owners.items():
            process = entry["process"]
            cleanup = {}
            try:
                if process.poll() is None:
                    send(arm, {"operation": "stop"})
                    process.wait(timeout=25)
                cleanup["exit_code"] = process.returncode
                cleanup["stopped"] = read(root / arm / "stopped.json")
                ready = entry["ready"]
                cleanup["owner_absent"] = identity["process_snapshot"](ready["pid"]) is None
                cleanup["ports_closed"] = {}
                for key in ("memory_port", "worker_port"):
                    with socket.socket() as sock:
                        cleanup["ports_closed"][key] = (
                            sock.connect_ex(("127.0.0.1", ready[key])) != 0
                        )
            except BaseException as exc:
                cleanup["error_type"] = type(exc).__name__
            execution["cleanup"][arm] = cleanup
        try:
            execution["files_unchanged"] = unchanged(freeze)
            selected_git = shutil.which("git")
            execution["git_resolution_unchanged"] = (
                bool(selected_git)
                and Path(selected_git).resolve() == Path(freeze["git_executable"]).resolve()
            )
            execution["freeze_unchanged"] = sha(freeze_path) == execution["freeze_sha256"]
            sys.path[:] = [
                p for p in sys.path if p != freeze["arms"]["baseline"]["installed_target"]
            ]
            execution["dependencies_unchanged"] = versions() == freeze["dependencies"]
        except Exception as exc:
            execution["postcheck_error"] = type(exc).__name__ + ": " + str(exc)
        finalize_status(execution)
        execution["finished_ns"] = time.perf_counter_ns()
        save(root / "execution.json", execution)
        if ledger is not None:
            save(root / "ledger.json", ledger)
    if execution["status"] != "completed":
        raise RuntimeError("Comparison cleanup or frozen-input verification failed")


def finalize_status(execution):
    clean = len(execution["cleanup"]) == 2
    for row in execution["cleanup"].values():
        clean = clean and (
            "error_type" not in row
            and row.get("exit_code") == 0
            and row.get("owner_absent")
            and row.get("ports_closed") == {"memory_port": True, "worker_port": True}
            and row.get("stopped", {}).get("stopped")
            and row.get("stopped", {}).get("cleanup_errors") == []
        )
    intact = (
        bool(execution.get("files_unchanged"))
        and all(execution["files_unchanged"].values())
        and execution.get("git_resolution_unchanged")
        and execution.get("freeze_unchanged")
        and execution.get("dependencies_unchanged")
        and "postcheck_error" not in execution
    )
    execution["status"] = (
        "completed"
        if clean
        and intact
        and execution.get("schedule_completed")
        and "error_type" not in execution
        else "failed"
    )


def require_job(job_name):
    """Verify membership in this named supervisor job, not any enclosing app job."""
    import ctypes

    active = ctypes.c_int()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.OpenJobObjectW.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_wchar_p)
    kernel.OpenJobObjectW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel.IsProcessInJob.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    )
    job = kernel.OpenJobObjectW(0x0004, False, job_name)
    if not job:
        raise RuntimeError("Comparison supervisor job is absent")
    try:
        if (
            not kernel.IsProcessInJob(kernel.GetCurrentProcess(), job, ctypes.byref(active))
            or not active.value
        ):
            raise RuntimeError("Comparison driver is outside its supervisor job")
    finally:
        kernel.CloseHandle(job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "supervise", "run"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--packages", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--git-manifest", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        prepare(
            root,
            args.packages.resolve(),
            args.native.resolve(),
            args.protocol.resolve(),
            args.git_manifest.resolve(),
        )
    elif args.mode == "supervise":
        freeze = read(root / "freeze.json")
        runpy.run_path(str(HERE / "windows_owned_job.py"))["run_owned_job"](
            [freeze["runtime_python"], "-I", str(Path(__file__).resolve()), "run", str(root)],
            cwd=root,
            env=dict(os.environ),
            output=root / "supervisor",
            timeout=300,
            job_name=freeze["job_name"],
        )
    else:
        require_job(read(root / "freeze.json")["job_name"])
        run(root)


if __name__ == "__main__":
    main()
