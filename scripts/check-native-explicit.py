"""One bounded native lifecycle against a verified installed Mnemos wheel.

Run with the isolated installation's Python and -I. Native authentication is
copied only by the existing phase runner and removed after every phase. All
memory, capture, config and trace writes belong to a new external output root.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-sonnet-4-6"
CODEX_MODEL = "gpt-6-astra"
CODEX_VERSION = "codex-cli 0.164.0-alpha.11.1"
CODEX_BINARY_SHA256 = "1103e9e09c71fa2cd0afd6d8f3b9a2d63174fe8efd7cb5bd09f1cb14ef58a098"
CODE_MODE_BINARY_SHA256 = "2727abaa2b30eb4afbe64b1da51a26cb594dd18dda82c6292b5373b15a758257"
PROFILE = "native-explicit-20261106"
INPUTS = (
    "scripts/check-native-explicit.py",
    "benchmarks/native_hosts/native_processes.py",
    "benchmarks/native_hosts/claude_isolation.py",
    "benchmarks/native_hosts/shared_trace.py",
    "benchmarks/native_hosts/three_host_contract.py",
    "examples/restart_recall.py",
    "docs/plans/2026-11-06-native-explicit-memory.md",
    "docs/plans/2026-11-06-native-explicit-auth-v2.md",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def helper(relative: str) -> Any:
    name = "native_checkpoint_" + Path(relative).stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load frozen acceptance helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def auth_preflight(path: Path, *, now: float | None = None) -> dict[str, Any]:
    """Do not ask a copied, expired login to rotate the live profile's tokens."""
    now = time.time() if now is None else now
    try:
        credential = json.loads(path.read_text(encoding="utf-8")).get("claudeAiOauth", {})
        expiry = credential.get("expiresAt")
        access = credential.get("accessToken")
        valid = (
            type(expiry) is int
            and expiry / 1000 - now >= 3600
            and isinstance(access, str)
            and bool(access)
        )
    except (OSError, ValueError, AttributeError):
        valid = False
    if not valid:
        raise ValueError(
            "A fresh isolated Claude login with at least one hour remaining is required"
        )
    return {"expires_at_ms": expiry, "minimum_remaining_seconds": 3600}


def codex_auth_preflight(path: Path, *, now: float | None = None) -> dict[str, Any]:
    """Inspect expiry metadata only; the native request must establish login."""
    now = time.time() if now is None else now
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        access = value.get("tokens", {}).get("access_token", "")
        parts = access.split(".")
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
        expiry = payload.get("exp")
        valid = (
            value.get("auth_mode") == "chatgpt"
            and not value.get("OPENAI_API_KEY")
            and len(parts) == 3
            and type(expiry) is int
            and expiry - now >= 3600
        )
    except (OSError, ValueError, AttributeError, IndexError, TypeError, binascii.Error):
        valid = False
    if not valid:
        raise ValueError("A fresh Codex login with at least one hour remaining is required")
    return {"auth_mode": "chatgpt", "expires_at": expiry, "minimum_remaining_seconds": 3600}


def verify_native_result(process: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    initializations = [
        row for row in rows if row.get("type") == "system" and row.get("subtype") == "init"
    ]
    allowed = process.get("allowed_tools")
    advertised = initializations[0].get("tools") if len(initializations) == 1 else None
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(not isinstance(name, str) or not name for name in allowed)
        or len(set(allowed)) != len(allowed)
        or not isinstance(advertised, list)
        or any(not isinstance(name, str) or not name for name in advertised)
        or len(set(advertised)) != len(advertised)
        or set(advertised) != set(allowed)
    ):
        raise ValueError("Native advertised tool set differs from the phase allowlist")
    if process.get("status") != "completed" or process.get("auth_copy_removed") is not True:
        raise ValueError("Native process or authentication cleanup failed")
    session = process["session_id"]
    if any(row.get("session_id", session) != session for row in rows):
        raise ValueError("Native session identity changed")
    terminal = [row for row in rows if row.get("type") == "result"]
    if (
        len(terminal) != 1
        or terminal[0].get("is_error") is not False
        or terminal[0].get("subtype") != "success"
        or not isinstance(terminal[0].get("result"), str)
        or not terminal[0]["result"].strip()
    ):
        raise ValueError("Native stream lacks a successful final result")
    models = [row.get("message", {}).get("model") for row in rows if row.get("type") == "assistant"]
    if not models or any(model != MODEL for model in models):
        raise ValueError("Native model observation differs from the requested model")
    return {
        "model": MODEL,
        "reported_cost_usd": terminal[0].get("total_cost_usd"),
        "advertised_tools": sorted(advertised),
        "tool_catalog_verified": True,
    }


def original_replay_records(
    records: list[dict[str, Any]], labels: tuple[str, ...]
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in records
        if row.get("operation") in {"remember", "capture"}
        and row.get("result", {}).get("status") == "committed"
        and any(label in row.get("payload", {}).get("content", "") for label in labels)
    ]
    if not selected:
        raise ValueError("No committed original source identities were recorded")
    return selected


def state(database: Path) -> dict[str, Any]:
    queries = {
        "facts": "SELECT id,content,version,metadata FROM memory_chunks ORDER BY id",
        "events": "SELECT id,role,content,source,scope_id,metadata,occurred_at FROM episodic_events ORDER BY id",
        "jobs": "SELECT id,kind,status,payload,result,last_error FROM memory_jobs ORDER BY id",
        "evidence": "SELECT fact_id,source_event_id,excerpt FROM fact_evidence ORDER BY fact_id,source_event_id",
        "revisions": "SELECT fact_id,version,content FROM fact_revisions ORDER BY fact_id,version",
        "suppressions": "SELECT fact_id,source_event_id,canonical_key FROM memory_suppressions ORDER BY suppression_id",
        "edges": "SELECT source_id,target_id,edge_type,weight FROM memory_edges ORDER BY source_id,target_id,edge_type",
    }
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Database integrity check failed")
        return {
            name: [dict(row) for row in connection.execute(sql)] for name, sql in queries.items()
        }
    finally:
        connection.close()


def serve(root: Path) -> None:
    """Preserve the installed default dispatcher and scheduler; record arrivals."""
    from mnemos.service import MnemosService

    class ObservedService(MnemosService):
        def _make_default_dispatcher(self) -> Any:
            dispatcher = super()._make_default_dispatcher()
            original = dispatcher.dispatch

            async def observed(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
                result = await original(operation, payload)
                if operation in {"capture", "remember", "correct"}:
                    with (root / "arrivals.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps(
                                {"operation": operation, "payload": payload, "result": result}
                            )
                            + "\n"
                        )
                return dict(result)

            dispatcher.dispatch = observed
            return dispatcher

    service = ObservedService(startup_timeout=120, dispatch_timeout=60)
    service.start()
    if service._external:
        raise ValueError("Fresh checkpoint unexpectedly reused an existing service")
    signal.signal(signal.SIGINT, lambda *_: service.shutdown_requested.set())
    signal.signal(signal.SIGTERM, lambda *_: service.shutdown_requested.set())
    try:
        while not service.shutdown_requested.wait(0.5):
            pass
    finally:
        service.stop()


def bindings(args: argparse.Namespace) -> dict[str, Any]:
    import mnemos

    host = getattr(args, "host", "claude-code")
    batch_inspection = getattr(args, "batch_inspection", False)
    if batch_inspection and host != "claude-code":
        raise ValueError("This batch-inspection protocol is bound to Claude")
    authentication = (
        codex_auth_preflight(args.auth) if host == "codex" else auth_preflight(args.auth)
    )
    package = Path(mnemos.__file__).resolve().parent
    if not package.is_relative_to(Path(sys.prefix).resolve()) or package.is_relative_to(ROOT):
        raise ValueError("Native verification requires an installed package outside the checkout")
    smoke = json.loads(args.smoke.read_text(encoding="utf-8"))
    if smoke.get("status") != "passed" or smoke.get("wheel_sha256") != sha(args.wheel):
        raise ValueError("Installed smoke evidence does not bind this wheel")
    archive_bytes = subprocess.check_output(
        ["git", "archive", args.source_commit, "mnemos", "agents"], cwd=ROOT
    )
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
        source = {member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}  # type: ignore[union-attr]
    with zipfile.ZipFile(args.wheel) as wheel:
        for member in wheel.namelist():
            if not member.startswith("mnemos/") or member.endswith("/"):
                continue
            raw = wheel.read(member)
            if (package.parent / member).read_bytes() != raw:
                raise ValueError("Installed package differs from the wheel: " + member)
            source_name = member.replace("mnemos/resources/agents/", "agents/")
            if source_name in source and source[source_name].replace(b"\r\n", b"\n") != raw.replace(
                b"\r\n", b"\n"
            ):
                raise ValueError("Wheel differs from its declared source: " + member)
    runner_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    inputs = INPUTS + (
        (
            "docs/plans/2026-11-06-native-codex-explicit-v1.md",
            "docs/plans/2026-11-06-native-codex-runtime-v2.md",
            "docs/plans/2026-11-06-native-codex-code-mode-v3.md",
            "docs/plans/2026-11-06-native-codex-trace-v4.md",
            "docs/plans/2026-11-06-native-codex-dispatch-v5.md",
            "docs/plans/2026-11-06-native-codex-job-guidance-v6.md",
            "docs/plans/2026-11-06-native-codex-review-selection-v7.md",
        )
        if host == "codex"
        else (
            "docs/plans/2026-11-06-native-claude-explicit-v2.md",
            "docs/plans/2026-11-06-native-claude-forget-guidance-v3.md",
            "docs/plans/2026-11-06-native-claude-review-pages-v4.md",
            "docs/plans/2026-11-06-native-claude-review-selection-v5.md",
        )
    )
    require_inventory = getattr(args, "require_tool_inventory", False)
    if batch_inspection:
        inputs += ("docs/plans/2026-11-08-native-claude-batch-v7.md",)
    lifecycle_prompts = {}
    if require_inventory:
        if host != "codex":
            raise ValueError("Native tool inventory observation requires Codex")
        prompt_manifest = "docs/plans/2026-11-07-native-codex-inventory-v9-prompts.json"
        inputs += ("docs/plans/2026-11-07-native-codex-inventory-v9.md", prompt_manifest)
        manifest = json.loads((ROOT / prompt_manifest).read_bytes())
        lifecycle_prompts = manifest["lifecycle_prompt_sha256"]
        if manifest.get("schema_version") != 1 or set(lifecycle_prompts) != {
            "teach",
            "recall-v1",
            "correct",
            "recall-v2",
            "forget",
            "recall-forgotten",
        }:
            raise ValueError("Frozen lifecycle prompt manifest is invalid")
    for relative in inputs:
        committed = subprocess.check_output(
            ["git", "show", f"{runner_commit}:{relative}"], cwd=ROOT
        )
        if committed.replace(b"\r\n", b"\n") != (ROOT / relative).read_bytes().replace(
            b"\r\n", b"\n"
        ):
            raise ValueError("Acceptance inputs must be committed before inference")
    executable = args.codex if host == "codex" else args.claude
    if host == "claude-code":
        isolation = helper("benchmarks/native_hosts/claude_isolation.py")
        native = helper("benchmarks/native_hosts/native_processes.py")
        isolation.verify_binary(executable)
        with tempfile.TemporaryDirectory(prefix="mnemos-claude-version-") as temporary:
            probe_root = Path(temporary)
            native._private(probe_root)
            probe_home = probe_root / "profile"
            probe_home.mkdir()
            probe_env = isolation.environment(native.clean_environment({}), probe_home, probe_root)
            host_version = subprocess.check_output(
                [str(executable), "--version"], cwd=probe_home, env=probe_env,
                text=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW,
            ).strip()
    else:
        host_version = subprocess.check_output([str(executable), "--version"], text=True).strip()
    expected_version = CODEX_VERSION if host == "codex" else "2.1.191 (Claude Code)"
    if host_version != expected_version:
        raise ValueError("Native host version differs from this protocol")
    host_digest = sha(executable)
    if host == "codex" and host_digest != CODEX_BINARY_SHA256:
        raise ValueError("Native Codex binary differs from the frozen runtime")
    code_mode_digest = None
    if host == "codex":
        code_mode_digest = sha(executable.with_name("codex-code-mode-host.exe"))
        if code_mode_digest != CODE_MODE_BINARY_SHA256:
            raise ValueError("Native Code Mode binary differs from the frozen runtime")
    return {
        "source_commit": args.source_commit,
        "runner_commit": runner_commit,
        "wheel_sha256": sha(args.wheel),
        "installed_smoke_sha256": sha(args.smoke),
        "package_path": str(package),
        "python": sys.version,
        "host_version": host_version,
        "host_executable_sha256": host_digest,
        "code_mode_executable_sha256": code_mode_digest,
        "host": host,
        "tool_inventory_required": require_inventory,
        "lifecycle_prompt_sha256": lifecycle_prompts,
        "model": CODEX_MODEL if host == "codex" else MODEL,
        "authentication_preflight": authentication,
        "input_hashes": {relative: sha(ROOT / relative) for relative in inputs},
        "batch_inspection": batch_inspection,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output.resolve()
    if root.is_relative_to(ROOT) or root.exists():
        raise ValueError("Output must be a fresh directory outside the checkout")
    bound = bindings(args)
    root.mkdir(parents=True)
    from mnemos.service import _protect_private_path

    _protect_private_path(root)
    example = helper("examples/restart_recall.py")
    native = helper("benchmarks/native_hosts/native_processes.py")
    trace = helper("benchmarks/native_hosts/shared_trace.py")
    contract = helper("benchmarks/native_hosts/three_host_contract.py")
    scenario = contract.Scenario()
    report: dict[str, Any] = {
        **bound,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "phases": [],
        "runtime_starts": [],
        "live_configuration_changed": False,
    }
    save(root / "report.json", report)
    settings = root / "settings.toml"
    settings.write_text(
        '[judgment]\nprovider="none"\n[modules]\nsurprisal=false\nmutable_rag=false\naffective=false\nsleep=false\nspreading=false\n'
        '[retrieval]\nprofile="baseline"\naffective_enabled=false\nspreading_enabled=false\nquery_state_source="none"\n',
        encoding="utf-8",
    )
    selectors = {
        key: value for key, value in example.isolated_env(root).items() if key.startswith("MNEMOS_")
    }
    selectors.update(
        MNEMOS_CONFIG_PATH=str(settings),
        MNEMOS_RUNTIME_DIR=str(root),
        MNEMOS_RUNTIME_DESCRIPTOR=str(root / "runtime.json"),
        MNEMOS_USER_PROFILE_ID=PROFILE,
    )
    env = {
        key: value
        for key, value in native.clean_environment(selectors).items()
        if not key.upper().startswith("GIT_")
    }
    project, foreign = root / "project", root / "foreign"
    for path in (root, project, foreign):
        path.mkdir(exist_ok=True)
        (path / ".mnemos").mkdir()
        (path / ".mnemos/mnemos.toml").write_text("", encoding="utf-8")
    for path in (project, foreign):
        example.run_process(["git", "init", "--quiet"], cwd=path, env=env)
    plugin = root / "plugin-snapshot"
    plugin_paths = subprocess.check_output(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            args.source_commit,
            "--",
            ".claude-plugin",
            "hooks",
            "commands",
            "agents",
            "skills",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    for relative in plugin_paths:
        destination = plugin / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            subprocess.check_output(["git", "show", f"{args.source_commit}:{relative}"], cwd=ROOT)
        )
    report["plugin_hashes"] = {name: sha(plugin / name) for name in plugin_paths}
    example.run_process(
        [
            sys.executable,
            "-I",
            "-c",
            "from mnemos.utils.embeddings import FastembedProvider; p=FastembedProvider(); assert len(p.embed('release label'))==384",
        ],
        cwd=root,
        env=env,
        timeout=900,
    )
    process: subprocess.Popen[bytes] | None = None
    controls: dict[str, dict[str, Any]] = {}
    with (root / "service.log").open("w+b") as log:

        def cli(arguments: list[str], cwd: Path = project) -> dict[str, Any]:
            return dict(example.cli(arguments, env=env, cwd=cwd))

        def start() -> None:
            nonlocal process
            process = subprocess.Popen(
                [sys.executable, "-I", str(Path(__file__).resolve()), "--serve", str(root)],
                cwd=root,
                env=env,
                stdout=log,
                stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            deadline = time.monotonic() + 120
            while not (root / "runtime.json").is_file():
                if process.poll() is not None:
                    raise RuntimeError("Isolated native service exited during startup")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Isolated native service startup timed out")
                time.sleep(0.1)
            health = cli(["stats"])
            if (
                health["runtime"]["cognition_provider"] != "UnavailableLLMProvider"
                or health["runtime"]["embedding_provider"] != "FastembedProvider"
            ):
                raise ValueError("Native runtime provider mismatch")
            report["runtime_starts"].append({"pid": process.pid, "health": health})
            save(root / "report.json", report)

        def stop() -> None:
            nonlocal process
            if process is not None:
                example.stop_service(process, env, project)
                process = None

        def maintain() -> dict[str, Any]:
            cli(["consolidate"])
            return state(root / "memory.sqlite")

        def phase(name: str, prompt: str, allowed: set[str]) -> list[dict[str, Any]]:
            row: dict[str, Any] = {"phase": name, "status": "running"}
            if bound["tool_inventory_required"]:
                row["lifecycle_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
                prompt = trace.inventory_observation_prompt(
                    prompt, expected_sha256=bound["lifecycle_prompt_sha256"][name]
                )
            report["phases"].append(row)
            save(root / "report.json", report)
            config = native.PhaseEnvironment(
                root,
                project,
                root / "scopes.json",
                root / "runtime.json",
                settings,
                PROFILE,
                plugin,
            )
            if bound["host"] == "codex":
                codex_auth_preflight(args.auth)
                result = native.run_codex_exec_phase(
                    config,
                    phase=name,
                    prompt=prompt,
                    executable=args.codex,
                    auth_source=args.auth,
                    model=CODEX_MODEL,
                    allowed_tools=allowed,
                )
            else:
                result = native.run_claude_phase(
                    config,
                    phase=name,
                    prompt=prompt,
                    executable=args.claude,
                    auth_source=args.auth,
                    allowed_tools=[native.CLAUDE_PREFIX + tool for tool in sorted(allowed)],
                )
            directory = root / name
            (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
            source = (
                Path(result["session_file"])
                if bound["host"] == "codex"
                else directory / "stdout.jsonl"
            )
            rows = [json.loads(line) for line in source.read_bytes().splitlines()]
            if bound["host"] == "codex":
                if (
                    result.get("status") != "completed"
                    or result.get("auth_copy_removed") is not True
                ):
                    raise ValueError("Native Codex process or authentication cleanup failed")
                row.update(
                    model=CODEX_MODEL,
                    reported_usage=result.get("reported_usage"),
                    session_file=str(source),
                    session_sha256=sha(source),
                )
            else:
                row.update(verify_native_result(result, rows))
            provenance = trace.TraceProvenance(
                bound["host"], result["pid"], result["session_id"], str(source), sha(source)
            )
            if bound["host"] == "codex":
                trace_root = Path(result["rollout_trace_root"])
                traces = list(trace_root.iterdir())
                if len(traces) != 1 or not traces[0].is_dir():
                    raise ValueError("Exactly one owned native rollout trace is required")
                if bound["tool_inventory_required"]:
                    calls, trace_hashes, inventory = trace.parse_codex_tool_inventory(
                        traces[0], provenance=provenance, model=CODEX_MODEL, allowed_tools=allowed
                    )
                    save(directory / "tool-inventory.json", inventory)
                    row["tool_inventory_verified"] = True
                    row["tool_inventory_sha256"] = sha(directory / "tool-inventory.json")
                else:
                    calls, trace_hashes = trace.parse_codex_code_mode_trace(
                        traces[0], provenance=provenance, model=CODEX_MODEL
                    )
                save(directory / "rollout-trace-hashes.json", trace_hashes)
                row["rollout_trace_hashes_sha256"] = sha(directory / "rollout-trace-hashes.json")
                row["correlation"] = "native_model_id_to_cell_id_to_mcp_runtime_id"
            else:
                calls = trace.parse_claude_tool_records(
                    trace.load_source_records(source, provenance=provenance), provenance=provenance
                )
            normalized = [
                {
                    "name": call.name,
                    "args": dict(call.arguments),
                    "result": dict(call.result),
                    "model_call_correlated": call.model_call_correlated,
                    "call_id": call.call_id,
                    "provider_call_id": call.provider_call_id,
                    "trace_file": call.provenance.source_file,
                    "trace_sha256": call.provenance.source_file_sha256,
                    "line": call.line_number,
                    "line_sha256": call.line_sha256,
                    "invocation_line": call.invocation_line_number,
                    "invocation_line_sha256": call.invocation_line_sha256,
                }
                for call in calls
            ]
            contract.require_native_calls(normalized, allowed)
            if scenario.foreign in json.dumps(rows):
                raise ValueError("Foreign-project content reached the native host")
            current = maintain()
            current_facts = {fact["id"]: fact for fact in current["facts"]}
            if any(
                current_facts.get(identity) != original for identity, original in controls.items()
            ):
                raise ValueError("Native phase changed a retained or foreign control")
            save(directory / "state.json", current)
            save(directory / "normalized-calls.json", normalized)
            row.update(
                status="observed",
                tool_count=len(normalized),
                stdout_sha256=sha(directory / "stdout.jsonl"),
                trace_file=str(source),
                trace_sha256=sha(source),
            )
            save(root / "report.json", report)
            print(
                json.dumps({"phase": name, "status": "observed", "tool_count": len(normalized)}),
                flush=True,
            )
            return normalized

        try:
            start()
            registered = cli(["enable-project", "--profile", PROFILE, "--path", str(project)])
            scope_id = registered["project"]["id"]
            other = cli(["enable-project", "--profile", PROFILE, "--path", str(foreign)], foreign)
            if scope_id == other["project"]["id"]:
                raise ValueError("Native fixture projects share an identity")
            for content, path in (
                (scenario.retained, project),
                (scenario.unrelated, project),
                (scenario.foreign, foreign),
            ):
                cli(["store", content, "--scope", "project"], path)
            seeded = maintain()
            controls = {fact["id"]: fact for fact in seeded["facts"]}
            retained = contract.asserted_fact(seeded, scenario.retained, scope_id)["id"]
            report.update(
                project_id=scope_id, foreign_project_id=other["project"]["id"], retained_id=retained
            )
            base = "Use only the requested Mnemos tools in this project's scope. Do not store conversational text as a fact. "
            calls = phase(
                "teach",
                base
                + f"Explicitly remember exactly this user fact: {scenario.old} Call mnemos_store once and retain its returned job_id. Call mnemos_consolidate, then mnemos_job_status for the job_id from mnemos_store. Consolidation returns aggregate maintenance status, not a job_id. Report completion only when the original job is completed.",
                {"mnemos_store", "mnemos_consolidate", "mnemos_job_status"},
            )
            contract.require_teach(calls, scenario.old)
            target = contract.asserted_fact(state(root / "memory.sqlite"), scenario.old, scope_id)[
                "id"
            ]
            report["target_id"] = target
            report["phases"][-1]["status"] = "passed"
            recall_prompt = (
                base
                + f'Call mnemos_retrieve with query exactly "{scenario.target_query}". Inspect the returned asserted release fact with mnemos_inspect and report its ID, version, content and retained source evidence. Do not change memory.'
            )
            for name, content, version in (
                ("recall-v1", scenario.old, 1),
                ("recall-v2", scenario.new, 2),
            ):
                stop()
                start()
                calls = phase(name, recall_prompt, {"mnemos_retrieve", "mnemos_inspect"})
                contract.require_recalled(
                    calls,
                    fact_id=target,
                    content=content,
                    version=version,
                    scope_id=scope_id,
                    query=scenario.target_query,
                )
                if not any(
                    call["name"] == "mnemos_inspect"
                    and call["result"].get("fact", {}).get("id") == target
                    and call["result"].get("evidence")
                    for call in calls
                ):
                    raise ValueError("Native recall omitted source evidence inspection")
                report["phases"][-1]["status"] = "passed"
                if version == 1:
                    calls = phase(
                        "correct",
                        base
                        + f'First retrieve query "{scenario.target_query}" and inspect the asserted fact. Correct that same ID at its inspected version to exactly: {scenario.new} Retain the job_id returned by mnemos_correct. Call mnemos_consolidate, then mnemos_job_status for that original correction job_id. Consolidation returns aggregate maintenance status, not a job_id. Report completion only after that job is completed.',
                        {
                            "mnemos_retrieve",
                            "mnemos_inspect",
                            "mnemos_correct",
                            "mnemos_consolidate",
                            "mnemos_job_status",
                        },
                    )
                    contract.require_correct(
                        [call for call in calls if call["name"] != "mnemos_consolidate"],
                        target,
                        scenario.new,
                    )
                    corrected = contract.asserted_fact(
                        state(root / "memory.sqlite"), scenario.new, scope_id
                    )
                    if corrected["id"] != target or corrected["version"] != 2:
                        raise ValueError(
                            "Correction changed fact identity or failed to advance version"
                        )
                    report["phases"][-1]["status"] = "passed"
            originals = original_replay_records(
                [json.loads(line) for line in (root / "arrivals.jsonl").read_text().splitlines()],
                scenario.forgotten_labels,
            )
            batch_guidance = (
                " Use mnemos_inspect_batch for every additional fact you intend to select, "
                "with explicit fact_ids in batches of at most twenty. Read every page "
                "using the same ordered IDs and returned cursor before applying. "
                "Check each selected fact's exact ID, version, content and source evidence. "
                "A final page alone is not a complete inspection. Do not select any "
                "additional fact whose full inspection has not been read."
                if bound["batch_inspection"] else ""
            )
            calls = phase(
                "forget",
                base
                + f'Retrieve query "{scenario.target_query}" and inspect the corrected release fact. Forget that release convention and its reviewed copies in this project. Use mnemos_forget_preview, inspect any additional selected facts, then mnemos_forget_related with the unchanged apply_preconditions and explicitly reviewed eligible source/fact IDs from that preview. Preserve the review convention and unrelated facts. Report the returned deletion coverage; do not repeat deleted labels in your final reply.'
                + batch_guidance,
                {
                    "mnemos_retrieve",
                    "mnemos_inspect",
                    "mnemos_forget_preview",
                    "mnemos_forget_related",
                } | ({"mnemos_inspect_batch"} if bound["batch_inspection"] else set()),
            )
            contract.require_forget(calls, target, require_batch=bound["batch_inspection"])
            contract.require_absent(
                state(root / "memory.sqlite"), target, scenario.forgotten_labels
            )
            report["phases"][-1]["status"] = "passed"
            stop()
            start()
            from mnemos.client import MnemosClient

            async def replay() -> list[dict[str, Any]]:
                client = MnemosClient(
                    descriptor_path=root / "runtime.json",
                    outbox_path=root / "replay-outbox.sqlite",
                    auto_start=False,
                )
                results = []
                for original in originals:
                    result = await client.request(original["operation"], original["payload"])
                    if result.get("processing_status") != "suppressed" or result.get(
                        "event_id"
                    ) != original["result"].get("event_id"):
                        raise ValueError(
                            "Original source replay was not suppressed at the original identity"
                        )
                    results.append(result)
                return results

            report["source_replay_count"] = len(asyncio.run(replay()))
            contract.require_absent(maintain(), target, scenario.forgotten_labels)
            stop()
            start()
            calls = phase(
                "recall-forgotten",
                base
                + f'Call mnemos_retrieve separately with query "{scenario.target_query}" and query "{scenario.retained_query}". Report whether the release fact is absent and whether the review convention remains. Do not store or change memory.',
                {"mnemos_retrieve"},
            )
            contract.require_recalled(
                calls,
                fact_id=retained,
                content=scenario.retained,
                version=1,
                scope_id=scope_id,
                query=scenario.retained_query,
            )
            if not any(call["args"].get("query") == scenario.target_query for call in calls):
                raise ValueError("Native host did not query the forgotten topic")
            contract.require_absent(
                state(root / "memory.sqlite"), target, scenario.forgotten_labels
            )
            final_text = (root / "recall-forgotten/stdout.jsonl").read_text(encoding="utf-8")
            if bound["host"] == "codex":
                final_text += Path(report["phases"][-1]["session_file"]).read_text(encoding="utf-8")
            if any(label in final_text for label in scenario.forgotten_labels):
                raise ValueError("Forgotten labels reached the fresh native context")
            report["phases"][-1]["status"] = "passed"
            foreign_recall = cli(
                ["retrieve", scenario.target_query, "--allowed-scopes", "project"], foreign
            )
            if not foreign_recall["facts"] or any(
                fact["scope_id"] != other["project"]["id"] for fact in foreign_recall["facts"]
            ):
                raise ValueError("Foreign project control retrieval failed")
            save(root / "foreign-control.json", foreign_recall)
            report["status"] = "passed"
        except BaseException as exc:
            report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            if report["phases"] and report["phases"][-1]["status"] != "passed":
                report["phases"][-1].update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            try:
                stop()
                if (
                    (root / "runtime.json").exists()
                    or list(root.rglob(".credentials.json"))
                    or list(root.rglob("auth.json"))
                ):
                    raise ValueError("Native service or copied authentication survived cleanup")
                report["cleanup"] = "passed"
            except BaseException as exc:
                report.update(status="failed", cleanup=type(exc).__name__)
                raise
            finally:
                report["finished_at"] = datetime.now(timezone.utc).isoformat()
                save(root / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--claude", type=Path)
    parser.add_argument("--codex", type=Path)
    parser.add_argument("--host", choices=("claude-code", "codex"), default="claude-code")
    parser.add_argument("--auth", type=Path)
    parser.add_argument("--require-tool-inventory", action="store_true")
    parser.add_argument("--batch-inspection", action="store_true")
    args = parser.parse_args()
    if args.serve:
        serve(args.serve.resolve())
        return
    if any(
        getattr(args, name) is None
        for name in (
            "wheel",
            "smoke",
            "source_commit",
            "output",
            "auth",
            "codex" if args.host == "codex" else "claude",
        )
    ):
        parser.error("All checkpoint and native-host arguments are required")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
