"""Execute the frozen three-host lifecycle with disposable native clients.

Run with the verified installed interpreter and -I, outside the repository.
No inference begins until the preceding queue and Hermes lifecycle pass their
prerequisites. Codex phases require a terminal and controller exit after each
completed native turn. Failures retain the original attempt without retry.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

if not __package__:
    # Expose only the benchmark package; Mnemos must come from the installed wheel.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "native_hosts"

from .hermes_lifecycle_run_v10 import (
    DIGESTS,
    HERMES_CHECKOUT,
    HERMES_COMMIT,
    HERMES_LIVE,
    HERMES_LIVE_COMMIT,
    HERMES_ENVIRONMENT_SHA256,
    MODEL,
    api,
    controls_unchanged,
    process_exists,
    sha,
    verify_committed_inputs,
    wsl,
)
from .three_host_evidence import hermes_evidence, require_scope_exclusion
from .ollama_readiness_v2 import run_readiness
from .native_processes import (
    CLAUDE_PREFIX,
    PhaseEnvironment,
    run_claude_phase,
    run_codex_phase,
    terminate_owned_process,
)
from .shared_trace import (
    TraceProvenance,
    load_source_records,
    parse_claude_tool_records,
    parse_codex_mcp_records,
)
from .three_host_contract import (
    Scenario,
    asserted_fact,
    require_absent,
    require_correct,
    require_forget,
    require_native_calls,
    require_recalled,
    require_teach,
)
from .three_host_runtime import ThreeHostRuntime, PROTOCOL, MAINTENANCE_PROFILE

CODEX_MODEL = "gpt-5.5"


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def host_version(executable: Path, expected: str) -> dict[str, str]:
    value = subprocess.check_output([str(executable), "--version"], text=True, timeout=15).strip()
    if expected not in value.split():
        raise ValueError("Native host version changed: " + value)
    return {"version": value, "executable": str(executable), "sha256": sha(executable)}


def host_environment(output: Path) -> dict[str, str]:
    """Recheck the independently maintained live interpreter without changing it."""

    def git(*command: str) -> str:
        return subprocess.check_output(
            ["wsl.exe", "-d", "Debian", "--", "git", "-C", HERMES_LIVE, *command],
            text=True,
            timeout=30,
        ).strip()

    revision = git("rev-parse", "HEAD")
    if revision != HERMES_LIVE_COMMIT or git("status", "--porcelain"):
        raise ValueError("Live Hermes revision or tracked source changed")
    inventory = subprocess.check_output(
        ["wsl.exe", "-d", "Debian", "--", HERMES_LIVE + "/venv/bin/python", "-m", "pip", "freeze"],
        text=True,
        timeout=30,
    )
    normalized = "\n".join(sorted(inventory.splitlines())) + "\n"
    output.write_text(normalized, encoding="utf-8")
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    if digest != HERMES_ENVIRONMENT_SHA256:
        raise ValueError("Hermes shared interpreter environment changed")
    return {"revision": revision, "environment_sha256": digest}


def require_phase_cleanup(root: Path) -> None:
    """Do not promote failed child cleanup into a clean parent result."""
    for path in root.glob("*/process.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        pid = row.get("launcher_pid") if row.get("host") == "hermes" else row.get("pid")
        if pid is not None and (type(pid) is not int or process_exists(pid)):
            raise ValueError("Native phase process exit is unverified")
        if row.get("host") == "hermes":
            if row.get("cleanup_established") is not True:
                raise ValueError("Native Hermes process cleanup is unverified")
        elif row.get("auth_copy_removed") is not True or row.get("cleanup_errors"):
            raise ValueError("Native Windows phase cleanup failed")
    if any(root.rglob("auth.json")) or any(root.rglob(".credentials.json")):
        raise ValueError("Disposable authentication survived cleanup")


def windows_trace(result: dict[str, Any], directory: Path) -> dict[str, Any]:
    host = result["host"]
    if result["status"] != "completed" or not result["auth_copy_removed"]:
        raise ValueError("Native process did not complete with authentication cleanup")
    if host == "codex":
        paths = list((Path(result["home"]) / "sessions").rglob("rollout*.jsonl"))
        if len(paths) != 1:
            raise ValueError("Expected one fresh Codex session")
        path = paths[0]
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        sessions = [row["payload"]["id"] for row in rows if row.get("type") == "session_meta"]
        if len(sessions) != 1:
            raise ValueError("Codex session identity is ambiguous")
        session_id = sessions[0]
        parser = parse_codex_mcp_records
        answers = [
            row["payload"].get("message", "")
            for row in rows
            if row.get("type") == "event_msg"
            and row.get("payload", {}).get("type") == "agent_message"
            and row["payload"].get("phase") == "final_answer"
        ]
    else:
        path = directory / "stdout.jsonl"
        session_id = result["session_id"]
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        if any(row.get("session_id", session_id) != session_id for row in rows):
            raise ValueError("Claude stream contains another session")
        terminal = [row for row in rows if row.get("type") == "result"]
        if (
            len(terminal) != 1
            or terminal[0].get("is_error") is not False
            or terminal[0].get("subtype") != "success"
        ):
            raise ValueError("Claude stream lacks a successful terminal result")
        answers = [terminal[0].get("result", "")]
        parser = parse_claude_tool_records
    if not answers or not all(isinstance(answer, str) and answer.strip() for answer in answers):
        raise ValueError("Native final answer was not recorded")
    provenance = TraceProvenance(host, result["pid"], session_id, str(path), sha(path))
    calls = parser(load_source_records(path, provenance=provenance), provenance=provenance)
    return {
        "calls": [
            {
                "name": call.name,
                "args": dict(call.arguments),
                "result": dict(call.result),
                "model_call_correlated": call.model_call_correlated,
                "call_id": call.call_id,
                "line": call.line_number,
                "line_sha256": call.line_sha256,
            }
            for call in calls
        ],
        "provenance": asdict(provenance),
        "final_answer": "\n".join(answers),
        "recorded_context": json.dumps(rows, ensure_ascii=False),
        "context_coverage": "native session/stream records; provider wire input is not intercepted",
    }


def run_hermes(
    runtime: ThreeHostRuntime,
    bindings: dict[str, Any],
    phase: str,
    prompt: str,
    *,
    mode: str = "model",
) -> dict[str, Any]:
    if mode not in {"model", "preflight"}:
        raise ValueError("Unknown native Hermes mode")
    root = runtime.root / phase
    root.mkdir()
    host_environment(root / "host-environment-before.txt")
    profile = api("show", {"model": MODEL})
    save(root / "sampling-profile-before.json", profile)
    parameters = dict(line.split(None, 1) for line in profile["parameters"].splitlines())
    expected = {
        "temperature": 0,
        "presence_penalty": 0,
        "seed": 42,
        "num_ctx": 65536,
        "top_k": 20,
        "top_p": 0.95,
    }
    if any(float(parameters.get(key, "nan")) != value for key, value in expected.items()):
        raise ValueError("Pinned Hermes sampling profile changed")
    for helper in (
        "hermes_native_bootstrap_v7.py",
        "hermes_native_bootstrap_v6.py",
        "hermes_request_observer.py",
        "ollama_readiness_v2.py",
    ):
        (root / (helper + ".txt")).write_bytes(Path(__file__).with_name(helper).read_bytes())
    home = root / "hermes-home"
    home.mkdir()
    save(
        home / "config.yaml",
        {
            "model": {
                "default": MODEL,
                "provider": "custom",
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "ollama",
                "context_length": 65536,
                "ollama_num_ctx": 65536,
            },
            "memory": {"provider": "mnemos"},
            "auxiliary": {"title_generation": {"model_upgrade_enabled": False}},
        },
    )
    bridge_entry = root / "bridge_entry.py"
    bridge_entry.write_text(
        "import os,runpy,sys\n"
        "for key in list(os.environ):\n"
        "    if key.startswith('MNEMOS_'): del os.environ[key]\n"
        f"os.environ['MNEMOS_CONFIG_PATH']={str(runtime.root / 'invalid-autostart.toml')!r}\n"
        f"os.chdir({str(runtime.root)!r})\n"
        f"sys.argv=['mnemos.bridge','--descriptor',{str(runtime.descriptor)!r}]\n"
        "runpy.run_module('mnemos.bridge',run_name='__main__')\n",
        encoding="utf-8",
    )
    cfg = {
        "phase_root": wsl(root),
        "wheel": wsl(Path(bindings["wheel"])),
        "wheel_sha256": bindings["wheel_sha256"],
        "project": wsl(runtime.project),
        "hermes_checkout": HERMES_CHECKOUT,
        "model": MODEL,
        "mode": mode,
        "expected_model_digest": DIGESTS[MODEL],
        "expected_context_length": 65536,
        "session_id": "native-three-host-" + phase,
        "old_label": "cedar-ridge-731",
        "new_label": "maple-ridge-842",
        "foreign_label": "forbidden-foreign-ridge-992",
        "env": {
            "HERMES_HOME": wsl(home),
            "HERMES_ENABLE_PROJECT_PLUGINS": "0",
            "MNEMOS_SCOPE_REGISTRY": wsl(runtime.registry_path),
            "MNEMOS_USER_PROFILE_ID": runtime.profile,
            "MNEMOS_ROUTING_REVIEW_DIR": wsl(runtime.root / "routing-review"),
            "MNEMOS_BRIDGE_COMMAND": json.dumps(
                [wsl(Path(sys.executable)), "-I", str(bridge_entry)]
            ),
            "OPENAI_BASE_URL": "http://127.0.0.1:11434/v1",
            "OPENAI_API_KEY": "ollama",
            "PYTHONIOENCODING": "utf-8",
        },
    }
    save(root / "config.json", cfg)
    (root / "query.txt").write_text(prompt, encoding="utf-8")
    process = None
    timed_out = False
    started = time.monotonic()
    deadline = 60 if mode == "preflight" else 240
    try:
        with (root / "stdout.log").open("wb") as stdout, (root / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(
                [
                    "wsl.exe",
                    "-d",
                    "Debian",
                    "--",
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=5s",
                    str(deadline) + "s",
                    HERMES_LIVE + "/venv/bin/python",
                    wsl(root / "hermes_native_bootstrap_v7.py.txt"),
                    wsl(root / "config.json"),
                ],
                stdout=stdout,
                stderr=stderr,
            )
            process.wait(timeout=deadline + 20)
        if process.returncode != 0:
            raise ValueError("Native Hermes phase failed")
    except subprocess.TimeoutExpired:
        timed_out = True
        raise
    finally:
        if process is not None:
            terminate_owned_process(process)
            save(
                root / "process.json",
                {
                    "host": "hermes",
                    "launcher_pid": process.pid,
                    "returncode": process.returncode,
                    "seconds": time.monotonic() - started,
                    "cleanup_established": not timed_out and process.returncode == 0,
                },
            )
        host_environment(root / "host-environment-after.txt")
    trace = [json.loads(line) for line in (root / "trace.jsonl").read_bytes().splitlines()]
    entered = [row for row in trace if row["operation"] == "bootstrap"]
    exited = [row for row in trace if row["operation"] == "bootstrap_exit"]
    if (
        len(entered) != 1
        or len(exited) != 1
        or any(entered[0].get(key) != exited[0].get(key) for key in ("origins", "adapter_origin"))
    ):
        raise ValueError("Hermes module origins changed or were not verified at exit")
    profile_after = api("show", {"model": MODEL})
    save(root / "sampling-profile-after.json", profile_after)
    if profile_after != profile:
        raise ValueError("Hermes sampling profile changed during phase")
    if mode == "preflight":
        from mnemos.adapters.hermes_tools import get_tool_schemas

        tools = {row["name"] for row in get_tool_schemas()}
        if not any(
            row["operation"] == "preflight_complete" and set(row.get("tool_names", [])) == tools
            for row in trace
        ) or any(
            row["operation"]
            in {
                "model_request",
                "model_response",
                "tool_result",
                "bridge_response",
                "provider_error",
            }
            for row in trace
        ):
            raise ValueError("Hermes preflight did not remain import-only with eight tools")
        return {
            "status": "preflight_passed",
            "tool_names": sorted(tools),
            "trace_sha256": sha(root / "trace.jsonl"),
        }
    from mnemos.adapters.hermes_tools import get_tool_schemas

    descriptions = {row["name"]: row["description"] for row in get_tool_schemas()}
    if any(
        row.get("memory_tool_descriptions") != descriptions
        for row in trace
        if row["operation"] == "model_request"
    ):
        raise ValueError("Hermes model did not receive the installed tool descriptions")
    return hermes_evidence(trace, root, model=MODEL, digest=DIGESTS[MODEL])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checks",
        "installed-checks",
        "prior-queue-state",
        "hermes-acceptance",
        "output",
        "codex-executable",
        "claude-executable",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--exclude-pid", type=int, action="append", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.preflight_only and (not sys.stdin.isatty() or not sys.stdout.isatty()):
        raise ValueError("Three-host execution requires a terminal for native Codex phases")
    repo = Path(__file__).resolve().parents[2]
    runtime = ThreeHostRuntime(
        args.output,
        repo,
        args.checks,
        args.installed_checks,
        args.prior_queue_state,
        args.exclude_pid,
        args.hermes_acceptance,
    )
    bindings = runtime.verify()
    inputs = [
        str(path.relative_to(repo)).replace("\\", "/")
        for path in Path(__file__).parent.glob("*.py")
    ]
    inputs.append(PROTOCOL)
    inputs.append("docs/plans/2026-09-20-three-host-continuity-protocol-v4.md")
    inputs.append("docs/plans/2026-09-20-three-host-continuity-protocol-v3.md")
    inputs.append("docs/plans/2026-09-20-three-host-continuity-protocol-v2.md")
    inputs.append("docs/validation/2026-09-20/codex-tool-mode-diagnostic-v1/selected-catalog.json")
    inputs.append("docs/plans/2026-09-19-three-host-continuity-protocol-v1.md")
    inputs.append("docs/validation/2026-09-20/native-model-profile-v2/profile.json")
    inputs.append("docs/validation/2026-09-19/native-hermes-local-v4/host-compatibility.patch")
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    verify_committed_inputs(repo, inputs, commit)
    if subprocess.check_output(
        ["git", "-C", str(repo), "diff", "HEAD", "--name-only"], text=True
    ).strip():
        raise ValueError("Commit prospective runner changes before native execution")
    versions = {
        "codex": host_version(args.codex_executable, "0.146.0"),
        "claude": host_version(args.claude_executable, "2.1.191"),
    }
    tags = {row["name"]: row["digest"] for row in api("tags")["models"]}
    if any(tags.get(name) != digest for name, digest in DIGESTS.items()):
        raise ValueError("Ollama model digest differs from frozen protocol")

    def host_git(*command: str) -> str:
        return subprocess.check_output(
            ["wsl.exe", "-d", "Debian", "--", "git", "-C", HERMES_CHECKOUT, *command], text=True
        ).strip()

    if host_git("rev-parse", "HEAD") != HERMES_COMMIT:
        raise ValueError("Hermes revision changed")
    if host_git("ls-files", "--others", "--exclude-standard"):
        raise ValueError("Isolated Hermes checkout contains untracked source")
    expected_patch = (
        repo / "docs/validation/2026-09-19/native-hermes-local-v4/host-compatibility.patch"
    )
    if host_git("diff").replace("\r\n", "\n").strip() != expected_patch.read_text().strip():
        raise ValueError("Hermes patch differs from validated isolated checkout")
    runtime.prepare()
    scenario = Scenario()
    report: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "bindings": bindings,
        "runner_commit": commit,
        "hosts": versions,
        "models": DIGESTS,
        "input_hashes": {name: sha(repo / name) for name in inputs},
        "scenario": asdict(scenario),
        "phases": [],
        "live_configuration_changed": False,
        "preflight_only": args.preflight_only,
        "codex_mode": "pinned gpt-5.5 with --disable code_mode_host; direct tools require native ID correlation",
        "codex_requested_model": CODEX_MODEL,
        "maintenance_profile": MAINTENANCE_PROFILE,
    }
    save(runtime.root / "report.json", report)
    env = PhaseEnvironment(
        runtime.root,
        runtime.project,
        runtime.registry_path,
        runtime.descriptor,
        runtime.settings,
        runtime.profile,
        runtime.plugin_snapshot,
    )

    def phase(host: str, name: str, prompt: str, allowed: set[str]) -> dict[str, Any]:
        if report["phases"] and report["phases"][-1]["status"] == "observed":
            report["phases"][-1]["status"] = "passed"
        runtime.set_phase(name)
        row: dict[str, Any] = {"name": name, "host": host, "status": "running"}
        report["phases"].append(row)
        save(runtime.root / "report.json", report)
        print(json.dumps({"phase": name, "host": host, "status": "starting"}), flush=True)
        if host == "hermes":
            evidence = run_hermes(runtime, bindings, name, prompt)
        else:
            if host == "codex":
                process = run_codex_phase(
                    env,
                    phase=name,
                    prompt=prompt,
                    executable=args.codex_executable,
                    auth_source=Path.home() / ".codex/auth.json",
                    disable_code_mode_host=True,
                    model=CODEX_MODEL,
                    migration_notice_target="gpt-5.6-sol",
                )
            else:
                process = run_claude_phase(
                    env,
                    phase=name,
                    prompt=prompt,
                    executable=args.claude_executable,
                    auth_source=Path.home() / ".claude/.credentials.json",
                    allowed_tools=sorted(CLAUDE_PREFIX + tool for tool in allowed),
                )
            evidence = windows_trace(process, runtime.root / name)
        (runtime.root / name / "prompt.txt").write_text(prompt, encoding="utf-8")
        save(runtime.root / name / "normalized-evidence.json", evidence)
        require_native_calls(evidence["calls"], allowed)
        require_scope_exclusion(
            evidence["calls"],
            forbidden_ids={
                row["id"] for row in controls["facts"] if row["content"] == scenario.foreign
            },
            forbidden_scope_ids={runtime.contexts["foreign-project"]["project_id"]},
        )
        if "forbidden-foreign-ridge-992" in evidence["recorded_context"]:
            raise ValueError("Foreign project label appeared in native records")
        runtime.set_phase("controller-maintenance")
        state = runtime.maintain()
        if not controls_unchanged(controls, state):
            raise ValueError("Native phase changed an unrelated or foreign control")
        if report.get("retained_id"):
            retained_fact = asserted_fact(
                state, scenario.retained, runtime.contexts["project"]["project_id"]
            )
            if retained_fact["id"] != report["retained_id"] or retained_fact["version"] != 1:
                raise ValueError("Native phase changed the retained review convention")
        save(runtime.root / name / "state.json", state)
        captures = [
            item
            for item in runtime.dispatch_records
            if item["phase"] == name and item["operation"] == "capture"
        ]
        row["capture"] = {
            "observed_receipts": len(captures),
            "committed": sum(
                item.get("result", {}).get("status") == "committed" for item in captures
            ),
            "errors": sum("error_type" in item for item in captures),
            "boundary": "service arrivals only; native callback completion remains separate",
        }
        evidence["state"] = state
        row["status"] = "observed"
        save(runtime.root / "dispatch-records.json", runtime.dispatch_records)
        return evidence

    def start_runtime() -> None:
        runtime.start()
        health = runtime.dispatch("health", {})
        report.setdefault("runtime_starts", []).append(health)
        save(runtime.root / "report.json", report)
        if health.get("maintenance", {}).get("scheduler_running") is not True:
            raise ValueError("Native runtime background maintenance is not running")

    def restart() -> None:
        runtime.stop()
        start_runtime()

    base = "Use only the requested Mnemos tools and this project's scope. Do not add conversational text as an explicit fact. "
    recall_tools = {"mnemos_retrieve"}
    teach_tools = {"mnemos_store", "mnemos_consolidate", "mnemos_job_status"}
    scope_id = runtime.contexts["project"]["project_id"]
    try:
        report["host_environment_before"] = host_environment(
            runtime.root / "host-environment-before.txt"
        )
        if args.preflight_only:
            report["hermes_preflight"] = run_hermes(
                runtime, bindings, "preflight", "", mode="preflight"
            )
            report["status"] = "preflight_passed"
        else:
            report["model_readiness"] = run_readiness(
                runtime.root / "model-readiness",
                model=MODEL,
                expected_digest=DIGESTS[MODEL],
                context_length=65536,
            )
            save(runtime.root / "report.json", report)
            start_runtime()
            runtime.set_phase("controller-controls")
            for label, content in (
                ("project", scenario.unrelated),
                ("foreign-project", scenario.foreign),
            ):
                runtime.dispatch(
                    "remember",
                    {
                        "context": runtime.contexts[label],
                        "event_id": "control-" + label,
                        "content": content,
                    },
                )
            controls = runtime.maintain()
            if len(controls["facts"]) != 2:
                raise ValueError("Control seeding did not produce exactly two facts")
            targets = {}
            for host, name, content in (
                ("codex", "teach-release", scenario.old),
                ("claude", "teach-review", scenario.retained),
            ):
                evidence = phase(
                    host,
                    name,
                    base
                    + f"Call mnemos_store exactly once with content {json.dumps(content)}, role user, scope project. Then consolidate, then use mnemos_job_status with the returned job ID to establish completion. Report the actual result.",
                    teach_tools,
                )
                require_teach(evidence["calls"], content)
                targets[name] = asserted_fact(evidence["state"], content, scope_id)
                if not controls_unchanged(controls, evidence["state"]):
                    raise ValueError("Teaching changed a protected control")
            target = targets["teach-release"]["id"]
            retained = targets["teach-review"]["id"]
            report["target_id"], report["retained_id"] = target, retained
            restart()
            recall_prompt = (
                base
                + f"Call mnemos_retrieve twice, once with query {json.dumps(scenario.target_query)} and once with query {json.dumps(scenario.retained_query)}. Report the asserted facts and their IDs and versions. Do not change memory."
            )
            evidence = phase("hermes", "hermes-recall-both", recall_prompt, recall_tools)
            require_recalled(
                evidence["calls"],
                fact_id=target,
                content=scenario.old,
                version=1,
                scope_id=scope_id,
                query=scenario.target_query,
            )
            require_recalled(
                evidence["calls"],
                fact_id=retained,
                content=scenario.retained,
                version=1,
                scope_id=scope_id,
                query=scenario.retained_query,
            )
            evidence = phase(
                "hermes",
                "hermes-correct",
                base
                + f"Retrieve this project's asserted release label, inspect its exact ID and version, then correct that same ID to exactly {json.dumps(scenario.new)} using request_id three-host-correct-1 and the inspected version. Use mnemos_job_status to establish completion. Do not add a fact.",
                {"mnemos_retrieve", "mnemos_inspect", "mnemos_correct", "mnemos_job_status"},
            )
            require_correct(evidence["calls"], target, scenario.new)
            corrected = asserted_fact(evidence["state"], scenario.new, scope_id)
            if corrected["id"] != target or corrected["version"] != 2:
                raise ValueError("Correction did not preserve target identity at version two")
            restart()
            for host in ("codex", "claude", "hermes"):
                evidence = phase(host, host + "-recall-v2", recall_prompt, recall_tools)
                require_recalled(
                    evidence["calls"],
                    fact_id=target,
                    content=scenario.new,
                    version=2,
                    scope_id=scope_id,
                    query=scenario.target_query,
                )
                require_recalled(
                    evidence["calls"],
                    fact_id=retained,
                    content=scenario.retained,
                    version=1,
                    scope_id=scope_id,
                    query=scenario.retained_query,
                )
                if scenario.forgotten_labels[0] in evidence["final_answer"]:
                    raise ValueError("Native final answer repeated the superseded release label")
            originals = [
                row
                for row in runtime.dispatch_records
                if row["operation"] in {"remember", "capture"}
                and isinstance(row.get("result"), dict)
                and any(label in json.dumps(row["payload"]) for label in scenario.forgotten_labels)
            ]
            if not originals:
                raise ValueError("No original teaching/capture identities were recorded for replay")
            evidence = phase(
                "claude",
                "claude-forget",
                base
                + "Retrieve the current asserted project release label and inspect its ID/version. Preview related forgetting at that ID/version, inspect each additional fact you intend to select and review its source events, then forget the target and only copies containing its current or historical release labels using the preview digest, matching preview limit, and exact selected IDs. Preserve the separate review-label convention and unrelated data. Do not repeat any release label in your final answer.",
                {
                    "mnemos_retrieve",
                    "mnemos_inspect",
                    "mnemos_forget_preview",
                    "mnemos_forget_related",
                },
            )
            require_forget(evidence["calls"], target)
            require_absent(evidence["state"], target, scenario.forgotten_labels)
            restart()
            runtime.set_phase("scripted-original-source-replay")
            replay = []
            for row in originals:
                result = runtime.dispatch(row["operation"], row["payload"])
                replay.append({"operation": row["operation"], "result": result})
                if result.get("processing_status") != "suppressed" or result.get("event_id") != row[
                    "result"
                ].get("event_id"):
                    raise ValueError(
                        "Original source replay was not suppressed at its original identity"
                    )
            save(runtime.root / "scripted-replay.json", replay)
            require_absent(runtime.maintain(), target, scenario.forgotten_labels)
            for host in ("codex", "claude", "hermes"):
                evidence = phase(host, host + "-recall-forgotten", recall_prompt, recall_tools)
                require_recalled(
                    evidence["calls"],
                    fact_id=retained,
                    content=scenario.retained,
                    version=1,
                    scope_id=scope_id,
                    query=scenario.retained_query,
                )
                if not any(
                    call["args"].get("query") == scenario.target_query for call in evidence["calls"]
                ):
                    raise ValueError("Forgotten topic was not queried independently")
                require_absent(evidence["state"], target, scenario.forgotten_labels)
                if any(
                    label in evidence["recorded_context"] for label in scenario.forgotten_labels
                ):
                    raise ValueError("Forgotten release label appeared in native records")
                if any(
                    row["contains_old_label"] or row["contains_new_label"]
                    for row in evidence.get("model_requests", [])
                ):
                    raise ValueError("Forgotten release label reached Hermes provider input")
            if not controls_unchanged(controls, runtime.rows()):
                raise ValueError("Unrelated or foreign controls changed")
            report["phases"][-1]["status"] = "passed"
            report["status"] = "bounded_lifecycle_observed"
            report["limits"] = [
                "Codex/Claude provider wire input not intercepted",
                "Capture callback completeness requires separate hook diagnostics",
                "Automatic cognition disabled",
                "Desktop and daily-use pilot not covered",
            ]
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        if isinstance(exc, subprocess.TimeoutExpired):
            report["cleanup_error"] = "Native launcher timed out; WSL process cleanup is unverified"
        if report["phases"] and report["phases"][-1]["status"] in {"running", "observed"}:
            report["phases"][-1]["status"] = "failed"
        (runtime.root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        try:
            runtime.stop()
            save(runtime.root / "dispatch-records.json", runtime.dispatch_records)
            if (runtime.root / "memory.sqlite").exists():
                save(runtime.root / "final-state.json", runtime.rows())
            report["host_environment_after"] = host_environment(
                runtime.root / "host-environment-after.txt"
            )
            if runtime.descriptor.exists():
                raise ValueError("Runtime descriptor survived cleanup")
            require_phase_cleanup(runtime.root)
        except Exception as exc:
            report.update(status="failed", cleanup_error=type(exc).__name__)
        report["cleanup_established"] = (
            not report.get("cleanup_error") and not runtime.descriptor.exists()
        )
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(runtime.root / "report.json", report)
        print(
            json.dumps({"status": report["status"], "report": str(runtime.root / "report.json")}),
            flush=True,
        )
    return 0 if report["status"] in {"bounded_lifecycle_observed", "preflight_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
