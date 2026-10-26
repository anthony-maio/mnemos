"""Run the pinned Hermes CLI lifecycle against a verified, isolated Mnemos wheel.

Invoke this file with the installed wheel environment's Python from outside
the repository. --preflight-only imports the real host without inference.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
from importlib.machinery import SourceFileLoader
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import traceback
from typing import Any, Callable
import urllib.request
import zipfile

HERMES_COMMIT = "14a346345486418aa85ed6526549412ee2665ade"
HERMES_LIVE_COMMIT = "ebe3fcb2f85ebdbe0e469e9d295636ac930246b4"
HERMES_ENVIRONMENT_SHA256 = "a175bfc46955bc72ebb1c854ce7646f3dad3fd22716d3219d3a42d8928a30f7f"
HERMES_LIVE = "/home/anthony/.hermes/hermes-agent"
HERMES_CHECKOUT = "/tmp/mnemos-hermes-cwd-compat-v1"
MODEL = "mnemos-native-qwen35-9b:greedy-ctx64k-v2"
DIGESTS = {
    MODEL: "87bc5156d86918e0435431ab0286167a4c79c4a4a259b82fba5f3f75022006e2",
    "qwen3.5:9b": "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7",
    "nomic-embed-text:latest": "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f",
}
OLD = "The project release label is cedar-ridge-731."
NEW = "The project release label is maple-ridge-842."
CONTROL = "The build artifact checksum marker is slate-safe-553."
FOREIGN = "The project release label is forbidden-foreign-ridge-992."


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wsl(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) < 3 or value[1:3] != ":/":
        raise ValueError("Expected an absolute Windows path")
    return "/mnt/" + value[0].lower() + value[2:]


def api(route: str, payload: Any = None) -> Any:
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/" + route,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180 if payload is not None else 10) as response:
        return json.load(response)


def process_exists(pid: int) -> bool:
    if sys.platform != "win32":
        raise OSError("This native Hermes lifecycle process probe requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:
            return False
        raise OSError(ctypes.get_last_error(), "Cannot establish prior process exit")
    try:
        code = ctypes.c_ulong()
        kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise OSError("Cannot read prior process state")
        return code.value == 259
    finally:
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel.CloseHandle(handle)


def require_prior_queue_terminal(
    path: Path, pids: list[int], *, probe: Callable[[int], bool] = process_exists
) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("status") not in {
        "complete",
        "completed",
        "incomplete",
        "passed",
        "failed",
        "finished",
        "finished_with_incomplete",
    }:
        raise ValueError("Prior model queue is not terminal")
    if (
        not pids
        or any(type(pid) is not int or pid <= 0 for pid in pids)
        or state.get("pid") not in pids
    ):
        raise ValueError("Provide valid process IDs including the recorded coordinator")
    if any(probe(pid) for pid in pids):
        raise ValueError("A prior model process is still live")
    return {"state": state, "excluded_pids": pids}


def verify_committed_inputs(repo: Path, paths: list[str], commit: str) -> None:
    for relative in paths:
        path = Path(relative)
        try:
            committed = subprocess.check_output(
                ["git", "-C", str(repo), "show", f"{commit}:{path.as_posix()}"],
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"Input is not committed: {relative}") from exc
        if committed.replace(b"\r\n", b"\n") != (repo / path).read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError(f"Input differs from committed bytes: {relative}")


def controls_unchanged(
    before: dict[str, list[dict[str, Any]]], after: dict[str, list[dict[str, Any]]]
) -> bool:
    identifiers = {row["id"] for row in before["facts"]}
    contents = {row["content"] for row in before["facts"]}
    sources = {row["source_event_id"] for row in before["evidence"]}

    def ordered(rows: list[dict[str, Any]]) -> list[str]:
        return sorted(json.dumps(row, sort_keys=True) for row in rows)

    return (
        ordered(before["facts"])
        == ordered(
            [
                row
                for row in after["facts"]
                if row["id"] in identifiers or row["content"] in contents
            ]
        )
        and ordered(before["evidence"])
        == ordered([row for row in after["evidence"] if row["fact_id"] in identifiers])
        and ordered(before["events"])
        == ordered([row for row in after["events"] if row["id"] in sources])
    )


def read_rows(database: Path) -> dict[str, list[dict[str, Any]]]:
    queries = {
        "facts": "SELECT id,content,version,metadata FROM memory_chunks ORDER BY id",
        "events": "SELECT id,role,content,source,scope_id,metadata,occurred_at FROM episodic_events ORDER BY id",
        "jobs": "SELECT id,kind,status,payload,result,last_error FROM memory_jobs ORDER BY id",
        "evidence": "SELECT fact_id,source_event_id,excerpt FROM fact_evidence ORDER BY fact_id,source_event_id",
        "revisions": "SELECT fact_id,version,content FROM fact_revisions ORDER BY fact_id,version",
        "suppressions": "SELECT fact_id,source_event_id,canonical_key FROM memory_suppressions ORDER BY suppression_id",
    }
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        return {
            name: [dict(row) for row in connection.execute(sql)] for name, sql in queries.items()
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--installed-checks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--prior-queue-state", type=Path)
    parser.add_argument("--exclude-pid", type=int, action="append", default=[])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = args.output.resolve()
    if root.is_relative_to(repo):
        raise ValueError("Disposable execution state must be outside the repository")
    root.mkdir(exist_ok=False)

    def save(name: str, value: Any) -> None:
        (root / name).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")

    def host_git(checkout: str, *arguments: str) -> str:
        return subprocess.check_output(
            ["wsl.exe", "-d", "Debian", "--", "git", "-C", checkout, *arguments], text=True
        ).strip()

    def host_environment() -> str:
        result = subprocess.check_output(
            [
                "wsl.exe",
                "-d",
                "Debian",
                "--",
                HERMES_LIVE + "/venv/bin/python",
                "-m",
                "pip",
                "freeze",
            ],
            text=True,
            timeout=30,
        )
        return "\n".join(sorted(result.splitlines())) + "\n"

    record: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "preflight_only": args.preflight_only,
        "experiment_kind": "native-copyable-preconditions-package-lifecycle-v14",
        "phases": [],
        "live_configuration_changed": False,
        "scope": (
            "native import/configuration preflight; no inference or lifecycle result"
            if args.preflight_only
            else "model-driven native Hermes baseline lifecycle with effective context checks"
        ),
    }
    service: Any = None
    save("report.json", record)
    try:
        checks = json.loads(args.checks.read_text(encoding="utf-8"))
        installed = json.loads(args.installed_checks.read_text(encoding="utf-8"))
        if checks["source_commit"] != "e6887deb89608dca34f7bbb463b5d0a3ba973011":
            raise ValueError("Native v14 requires the frozen copyable-preconditions package")
        if checks["status"] != "passed" or installed["status"] != "passed":
            raise ValueError("Package evidence did not pass")
        wheel_row = next(row for row in checks["artifacts"] if row["file"].endswith(".whl"))
        wheel = Path(wheel_row["file"])
        wheel_sha = sha(wheel)
        if (
            wheel_sha != "3d37b864d35916dbed0c22bcb0428a4e091d6b6c31a0177491130fa18da34fa1"
            or wheel_sha != wheel_row["sha256"]
            or wheel_sha != installed["bindings_after"]["wheel_sha256"]
        ):
            raise ValueError("Wheel does not match verified package")
        if checks["source_commit"] != installed["bindings_after"]["commit"]:
            raise ValueError("Source and installed checkpoints differ")
        if sha(args.checks) != installed["bindings_after"]["checks_sha256"]:
            raise ValueError("Source verification report changed")
        if Path(sys.prefix).resolve() != Path(installed["environment"]).resolve():
            raise ValueError("Interpreter differs from the verified installed environment")
        import mnemos
        from mnemos.scopes import ScopeRegistry

        if not Path(mnemos.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError("Run with the installed wheel Python outside the checkout")
        installed_root = Path(mnemos.__file__).resolve().parent.parent
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.startswith("mnemos/") and not name.endswith("/"):
                    if (installed_root / name).read_bytes().replace(b"\r\n", b"\n") != archive.read(
                        name
                    ).replace(b"\r\n", b"\n"):
                        raise ValueError("Installed package bytes differ from the verified wheel")
        record.update(
            source_commit=checks["source_commit"],
            wheel_sha256=wheel_sha,
            installed_package=mnemos.__file__,
            hermes_commit=HERMES_COMMIT,
        )
        for checkout, expected in (
            (HERMES_LIVE, HERMES_LIVE_COMMIT),
            (HERMES_CHECKOUT, HERMES_COMMIT),
        ):
            if host_git(checkout, "rev-parse", "HEAD") != expected:
                raise ValueError("Hermes revision changed")
        if host_git(HERMES_LIVE, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError("Live Hermes has tracked modifications")
        environment = host_environment()
        (root / "host-environment-before.txt").write_text(environment, encoding="utf-8")
        environment_digest = hashlib.sha256(environment.encode()).hexdigest()
        record.update(
            live_hermes_commit=HERMES_LIVE_COMMIT, host_environment_sha256=environment_digest
        )
        if environment_digest != HERMES_ENVIRONMENT_SHA256:
            raise ValueError("Hermes shared environment changed")
        patch = host_git(HERMES_CHECKOUT, "diff") + "\n"
        expected_patch = (
            repo / "docs/validation/2026-09-19/native-hermes-local-v4/host-compatibility.patch"
        )
        if (
            patch.replace("\r\n", "\n").strip()
            != expected_patch.read_text(encoding="utf-8").strip()
        ):
            raise ValueError("Isolated host patch differs from the validated cwd-only patch")
        (root / "host-compatibility.patch").write_text(patch, encoding="utf-8")
        protocol = repo / "docs/plans/2026-09-20-hermes-native-lifecycle-protocol-v14.md"
        record["input_hashes"] = {
            str(path.relative_to(repo)): sha(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("hermes_native_bootstrap_v6.py"),
                Path(__file__).with_name("hermes_request_observer.py"),
                repo / "docs/plans/2026-09-20-hermes-native-lifecycle-protocol-v8.md",
                Path(__file__).with_name("hermes_trace.py"),
                Path(__file__).with_name("hermes_trace_v2.py"),
                protocol,
                repo / "docs/plans/2026-09-20-hermes-native-lifecycle-protocol-v10.md",
                repo / "docs/plans/2026-09-20-hermes-native-lifecycle-protocol-v9.md",
                Path(__file__).with_name("ollama_readiness_v2.py"),
                repo / "docs/plans/2026-09-20-hermes-native-lifecycle-protocol-v7.md",
                repo / "docs/plans/2026-09-19-hermes-native-lifecycle-protocol-v6.md",
                repo / "docs/plans/2026-09-19-hermes-native-lifecycle-protocol-v3.md",
                repo / "docs/plans/2026-09-19-hermes-native-lifecycle-protocol-v4.md",
                repo / "docs/plans/2026-09-19-hermes-native-lifecycle-protocol-v5.md",
                repo / "docs/validation/2026-09-20/native-model-profile-v2/profile.json",
            )
        }
        for relative in record["input_hashes"]:
            path = repo / relative
            suffix = ".txt" if path.suffix == ".py" else ""
            (root / (path.name + suffix)).write_bytes(path.read_bytes())
        record["runner_commit"] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if not args.preflight_only:
            verify_committed_inputs(repo, list(record["input_hashes"]), record["runner_commit"])
            if subprocess.check_output(
                ["git", "-C", str(repo), "diff", "HEAD", "--name-only"], text=True
            ).strip():
                raise ValueError(
                    "Commit the prospective runner and protocol before model execution"
                )
            if args.prior_queue_state is None or not args.exclude_pid:
                raise ValueError("Provide terminal prior queue evidence and its exact process IDs")
            record["prior_queue"] = require_prior_queue_terminal(
                args.prior_queue_state, args.exclude_pid
            )
            tags = {row["name"]: row["digest"] for row in api("tags")["models"]}
            if any(tags.get(name) != digest for name, digest in DIGESTS.items()):
                raise ValueError("Pinned Ollama model digest changed")
            profile = api("show", {"model": MODEL})
            save("sampling-profile-before.json", profile)
            params = dict(line.split(None, 1) for line in profile["parameters"].splitlines())
            if any(
                float(params.get(key, "nan")) != value
                for key, value in {
                    "temperature": 0,
                    "presence_penalty": 0,
                    "seed": 42,
                    "num_ctx": 65536,
                }.items()
            ):
                raise ValueError("Native sampling profile changed")
            record["models"] = DIGESTS
            record["ollama_version"] = api("version")
        registry = ScopeRegistry()
        contexts = {}
        for label in ("project", "foreign-project"):
            directory = root / label
            directory.mkdir()
            project = registry.register_project(
                directory, user_profile_id="native-lifecycle", enabled=True
            )
            project.aliases.append(wsl(directory))
            contexts[label] = registry.resolve(
                directory,
                user_profile_id="native-lifecycle",
                session_id="controller",
                host="manual",
            ).model_dump(mode="json")
        registry.save(root / "scopes.json")
        home = root / "hermes-home"
        home.mkdir()
        (home / "config.yaml").write_text(
            json.dumps(
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
                }
            ),
            encoding="utf-8",
        )
        (root / "invalid-autostart.toml").write_text(
            "[intentionally unclosed table\n", encoding="utf-8"
        )
        bridge_entry = root / "bridge_entry.py"
        bridge_entry.write_text(
            "import os, runpy, sys\n"
            "for key in list(os.environ):\n"
            "    if key.startswith('MNEMOS_'): del os.environ[key]\n"
            f"os.environ['MNEMOS_CONFIG_PATH'] = {str(root / 'invalid-autostart.toml')!r}\n"
            f"os.chdir({str(root)!r})\n"
            f"sys.argv = ['mnemos.bridge', '--descriptor', {str(root / 'runtime.json')!r}]\n"
            "runpy.run_module('mnemos.bridge', run_name='__main__')\n",
            encoding="utf-8",
        )
        env = {
            "HERMES_HOME": wsl(home),
            "HERMES_ENABLE_PROJECT_PLUGINS": "0",
            "MNEMOS_SCOPE_REGISTRY": wsl(root / "scopes.json"),
            "MNEMOS_USER_PROFILE_ID": "native-lifecycle",
            "MNEMOS_ROUTING_REVIEW_DIR": wsl(root / "routing-review"),
            "MNEMOS_BRIDGE_COMMAND": json.dumps(
                [wsl(Path(sys.executable)), "-I", str(bridge_entry)]
            ),
            "OPENAI_BASE_URL": "http://127.0.0.1:11434/v1",
            "OPENAI_API_KEY": "ollama",
            "PYTHONIOENCODING": "utf-8",
        }

        def run_host(
            phase: str, *, mode: str = "model", prompt: str = "", **extra: Any
        ) -> list[dict[str, Any]]:
            phase_root = root / phase
            phase_root.mkdir()
            cfg = {
                "phase_root": wsl(phase_root),
                "wheel": wsl(wheel),
                "wheel_sha256": wheel_sha,
                "project": wsl(root / "project"),
                "hermes_checkout": HERMES_CHECKOUT,
                "model": MODEL,
                "env": env,
                "mode": mode,
                "session_id": "preflight",
                "expected_model_digest": DIGESTS[MODEL],
                "expected_context_length": 65536,
                "old_label": "cedar-ridge-731",
                "new_label": "maple-ridge-842",
                "foreign_label": "forbidden-foreign-ridge-992",
                **extra,
            }
            (phase_root / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            (phase_root / "query.txt").write_text(prompt, encoding="utf-8")
            deadline = 60 if mode == "preflight" else 240
            started = time.monotonic()
            with (
                (phase_root / "stdout.log").open("wb") as stdout,
                (phase_root / "stderr.log").open("wb") as stderr,
            ):
                result = subprocess.run(
                    [
                        "wsl.exe",
                        "-d",
                        "Debian",
                        "--",
                        "timeout",
                        "--signal=TERM",
                        "--kill-after=5s",
                        f"{deadline}s",
                        HERMES_LIVE + "/venv/bin/python",
                        wsl(root / "hermes_native_bootstrap_v6.py.txt"),
                        wsl(phase_root / "config.json"),
                    ],
                    stdout=stdout,
                    stderr=stderr,
                    timeout=deadline + 20,
                )
            entry = {
                "phase": phase,
                "mode": mode,
                "seconds": time.monotonic() - started,
                "exit_code": result.returncode,
            }
            record["phases"].append(entry)
            save("report.json", record)
            print(json.dumps(entry), flush=True)
            if result.returncode != 0:
                raise ValueError(f"Native host failed in {phase}; retained output is authoritative")
            return [
                json.loads(line)
                for line in (phase_root / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        if args.preflight_only:
            trace = run_host("preflight", mode="preflight")
            expected_tools = {
                "mnemos_remember",
                "mnemos_retrieve",
                "mnemos_inspect",
                "mnemos_correct",
                "mnemos_forget",
                "mnemos_forget_preview",
                "mnemos_forget_related",
                "mnemos_job_status",
            }
            if not any(
                row["operation"] == "preflight_complete"
                and set(row.get("tool_names", [])) == expected_tools
                for row in trace
            ) or any(
                row["operation"]
                in {"model_request", "model_response", "tool_result", "bridge_response"}
                for row in trace
            ):
                raise ValueError("Preflight did not remain import/configuration only")
            record["status"] = "preflight_passed"
        else:
            from mnemos.config import MnemosConfig, ModuleConfig
            from mnemos.engine import MnemosEngine
            from mnemos.memory_service import LocalMemoryService
            from mnemos.service import MnemosService
            from mnemos.utils.embeddings import OllamaEmbeddingProvider
            from mnemos.utils.llm import UnavailableLLMProvider
            from mnemos.utils.storage import SQLiteStore

            config = MnemosConfig(
                modules=ModuleConfig(**{name: False for name in ModuleConfig.model_fields})
            )
            config.surprisal.embedding_dim = 768
            record["service_config"] = config.model_dump(mode="json")
            record["service_providers"] = {
                "cognition": "UnavailableLLMProvider",
                "embeddings": "OllamaEmbeddingProvider/nomic-embed-text",
                "automatic_capture": True,
            }

            def start_service() -> Any:
                def factory() -> LocalMemoryService:
                    engine = MnemosEngine(
                        config=config,
                        llm=UnavailableLLMProvider(),
                        embedder=OllamaEmbeddingProvider(
                            base_url="http://127.0.0.1:11434", timeout=30
                        ),
                        store=SQLiteStore(str(root / "memory.sqlite")),
                        scope_registry=registry,
                        require_scope_context=True,
                        owns_providers=True,
                    )
                    return LocalMemoryService(engine, registry)

                current = MnemosService(
                    dispatcher_factory=factory,
                    descriptor_path=root / "runtime.json",
                    startup_timeout=10,
                    dispatch_timeout=45,
                )
                current.start()
                return current

            service = start_service()

            def maintain() -> dict[str, list[dict[str, Any]]]:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    service.submit("maintenance", {})
                    state = read_rows(root / "memory.sqlite")
                    if any(row["last_error"] for row in state["jobs"]):
                        raise ValueError("Maintenance reported a processing error")
                    if all(row["status"] == "completed" for row in state["jobs"]):
                        return state
                    time.sleep(0.05)
                raise TimeoutError("Maintenance did not finish within its observation budget")

            for label, content in (("project", CONTROL), ("foreign-project", FOREIGN)):
                service.submit(
                    "remember",
                    {
                        "context": contexts[label],
                        "event_id": "control-" + label,
                        "content": content,
                    },
                )
            controls = maintain()
            if len(controls["facts"]) != 2:
                raise ValueError("Control seeding did not complete")
            save("controls.json", controls)
            readiness_loader = SourceFileLoader(
                "native_ollama_readiness", str(root / "ollama_readiness_v2.py.txt")
            )
            readiness_spec = importlib.util.spec_from_loader(
                "native_ollama_readiness", readiness_loader
            )
            if readiness_spec is None or readiness_spec.loader is None:
                raise ValueError("Cannot load pinned readiness helper")
            readiness_module = importlib.util.module_from_spec(readiness_spec)
            readiness_spec.loader.exec_module(readiness_module)
            record["model_readiness"] = readiness_module.run_readiness(
                root / "model-readiness",
                model=MODEL,
                expected_digest=DIGESTS[MODEL],
                context_length=65536,
            )
            save("report.json", record)
            loader = SourceFileLoader("native_hermes_trace", str(root / "hermes_trace_v2.py.txt"))
            spec = importlib.util.spec_from_loader("native_hermes_trace", loader)
            if spec is None or spec.loader is None:
                raise ValueError("Cannot load pinned trace evaluator")
            evaluator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluator)
            instructions = "Use only the mnemos_* tools. Do not use built-in memory or other tools. Do not store the conversation as a new explicit fact. "
            prompts = {
                "teach": instructions
                + f"Use mnemos_remember to save the exact content in this JSON string (including its final period): {json.dumps(OLD)}. Use request_id native-teach-1. Check its returned job ID once with mnemos_job_status, then report the ID and actual status. Pending is acceptable; do not claim completion unless confirmed. Do not correct or forget anything.",
                "recall_v1": instructions
                + "Use mnemos_retrieve to find this project's release label. Report the asserted fact's ID, version, and label. Do not change memory.",
                "correct": instructions
                + f"Retrieve the asserted fact about this project's release label, then inspect its exact ID and version. Correct that same fact to the exact content in this JSON string (including its final period): {json.dumps(NEW)}. Use request_id native-correct-1 and the inspected version. Do not add a separate fact.",
                "recall_v2": instructions
                + "Use mnemos_retrieve to find the current asserted project release label. Report its ID, version, and label. Do not change memory.",
                "forget": instructions
                + "Forget this project's release label, including retained conversational copies containing it. Retrieve the asserted target and inspect its exact current fact ID and version. Call mnemos_forget_preview at that version. Review the exact, historical, and advisory candidates and their source dependencies. Inspect each selected additional fact by its exact ID and current version. Call mnemos_forget_related with the unchanged preview_digest, matching preview_limit, explicitly reviewed fact IDs, and the selected source event IDs required by those facts. Preserve unrelated facts. Do not apply an incomplete preview or mismatched versions, and do not claim complete erasure from a bounded semantic search. Do not add any memory.",
                "recall_forgotten": instructions
                + "Use mnemos_retrieve to check this project's release label. Report whether a current label is known; if absent, say unknown. Do not change memory.",
            }
            target_id = None
            known_project_fact_ids: set[str] = set()
            teaching: list[dict[str, Any]] = []
            for phase, prompt in prompts.items():
                current_models = {row["name"]: row["digest"] for row in api("tags")["models"]}
                if any(current_models.get(name) != digest for name, digest in DIGESTS.items()):
                    raise ValueError("Pinned native model changed between phases")
                phase_profile = api("show", {"model": MODEL})
                save(phase + "-sampling-profile.json", phase_profile)
                if phase_profile != profile:
                    raise ValueError("Pinned sampling profile changed between phases")
                if phase in {"recall_v1", "recall_v2", "recall_forgotten"}:
                    service.stop()
                    service = start_service()
                if phase == "recall_forgotten":
                    taught = next(
                        row
                        for row in evaluator.correlate_native_tools(teaching)
                        if row["name"] == "mnemos_remember" and row["args"].get("content") == OLD
                    )
                    transcript = next(
                        row for row in reversed(teaching) if row["operation"] == "transcript"
                    )
                    replay = run_host(
                        "source-replay",
                        mode="replay",
                        session_id=transcript["session_id"],
                        messages=transcript["messages"],
                        remember_args=taught["args"],
                    )
                    save("source-replay-records.json", replay)
                    replayed = [
                        row
                        for row in replay
                        if row["operation"] == "tool_result" and row["name"] == "mnemos_remember"
                    ]
                    replay_gates = {
                        "original_write_suppressed": len(replayed) == 1
                        and replayed[0]["result"].get("processing_status") == "suppressed"
                        and replayed[0]["result"].get("event_id") == taught["result"]["event_id"],
                        "native_replay_flushed": any(
                            row["operation"] == "replay_complete" for row in replay
                        ),
                        "no_provider_error": not any(
                            row["operation"] == "provider_error" for row in replay
                        ),
                    }
                    record["phases"][-1]["gates"] = replay_gates
                    save("report.json", record)
                    if not all(replay_gates.values()):
                        raise ValueError("Original source replay was not demonstrably suppressed")
                    maintain()
                trace = run_host(phase, prompt=prompt)
                gates = evaluator.evaluate_phase(
                    trace, phase=phase, target_id=target_id, old_content=OLD, new_content=NEW
                )
                gates["no_foreign_tool_result"] = all(
                    "forbidden-foreign-ridge-992" not in json.dumps(row.get("result", {}))
                    for row in trace
                    if row["operation"] == "tool_result"
                )
                requests = [row for row in trace if row["operation"] == "model_request"]
                gates["real_pinned_model_request"] = bool(requests) and all(
                    row["model"] == MODEL
                    and row["reasoning_effort"] == "none"
                    and (row.get("options") or {}).get("num_ctx") == 65536
                    and not row["contains_foreign_label"]
                    for row in requests
                )
                responses = [row for row in trace if row["operation"] == "model_response"]
                loaded_checks = [
                    row for row in trace if row["operation"] == "loaded_profile_verified"
                ]
                gates["effective_context_verified_after_each_response"] = bool(responses) and (
                    [row.get("response_number") for row in loaded_checks]
                    == list(range(1, len(responses) + 1))
                )
                gates["pinned_profile_without_conflicting_override"] = bool(requests) and all(
                    all(
                        row.get(key) in (None, value)
                        and (row.get("options") or {}).get(key, value) == value
                        for key, value in {
                            "temperature": 0,
                            "presence_penalty": 0,
                            "seed": 42,
                        }.items()
                    )
                    for row in requests
                )
                from mnemos.adapters.hermes_tools import get_tool_schemas

                expected_descriptions = {
                    row["name"]: row["description"] for row in get_tool_schemas()
                }
                gates["write_followup_guidance_present"] = bool(requests) and all(
                    row.get("memory_tool_descriptions") == expected_descriptions for row in requests
                )
                gates["no_provider_error"] = not any(
                    row["operation"] == "provider_error" for row in trace
                )
                state = maintain()
                save(phase + "-state.json", state)
                known_project_fact_ids.update(
                    row["id"]
                    for row in state["facts"]
                    if json.loads(row["metadata"]).get("scope") == "project"
                    and json.loads(row["metadata"]).get("scope_id")
                    == contexts["project"]["project_id"]
                    and json.loads(row["metadata"]).get("user_profile_id")
                    == contexts["project"]["user_profile_id"]
                )
                record["known_project_fact_ids"] = sorted(known_project_fact_ids)
                gates["controls_unchanged"] = controls_unchanged(controls, state)
                if phase in {"forget", "recall_forgotten"}:
                    gates["no_retained_label_in_scope"] = evaluator.no_retained_labels(
                        state,
                        context=contexts["project"],
                        labels=["cedar-ridge-731", "maple-ridge-842"],
                        known_fact_ids=known_project_fact_ids,
                    )
                if phase == "forget":
                    gates["reviewed_deletions_persisted"] = evaluator.reviewed_deletions_persisted(
                        trace, state
                    )
                captures = [
                    row
                    for row in trace
                    if row["operation"] == "bridge_response"
                    and row["request_operation"] == "capture"
                ]
                gates["native_capture_committed"] = bool(captures) and all(
                    row["result"].get("status") == "committed" for row in captures
                )
                gates["native_capture_retained_or_suppressed"] = bool(captures) and all(
                    row["result"].get("processing_status") == "suppressed"
                    or any(
                        event["id"] == row["result"].get("event_id") for event in state["events"]
                    )
                    for row in captures
                )
                if phase == "teach":
                    teaching = trace
                    matching = [
                        fact
                        for fact in state["facts"]
                        if fact["content"] == OLD
                        and json.loads(fact["metadata"]).get("assertion_status") == "asserted"
                    ]
                    gates["single_asserted_target"] = len(matching) == 1
                    if matching:
                        target_id = matching[0]["id"]
                        record["target_id"] = target_id
                if phase == "correct":
                    gates["same_id_version_two"] = any(
                        fact["id"] == target_id and fact["version"] == 2 and fact["content"] == NEW
                        for fact in state["facts"]
                    )
                if phase in {"forget", "recall_forgotten"}:
                    gates["target_absent"] = all(fact["id"] != target_id for fact in state["facts"])
                    gates["target_suppressed"] = any(
                        row["fact_id"] == target_id for row in state["suppressions"]
                    )
                if phase == "recall_forgotten":
                    gates["no_forgotten_label_in_request"] = all(
                        not row["contains_old_label"] and not row["contains_new_label"]
                        for row in requests
                    )
                record["phases"][-1]["gates"] = gates
                save("report.json", record)
                if not all(gates.values()):
                    raise ValueError(f"Native lifecycle gate failed in {phase}")
            profile_after = api("show", {"model": MODEL})
            save("sampling-profile-after.json", profile_after)
            if profile_after != profile:
                raise ValueError("Native sampling profile changed during the lifecycle")
            record["status"] = "passed"
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        if isinstance(exc, subprocess.TimeoutExpired):
            record["cleanup_error"] = "Native launcher timed out; WSL process cleanup is unverified"
        (root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        try:
            if service is not None:
                service.stop()
            if host_git(HERMES_LIVE, "status", "--porcelain", "--untracked-files=no"):
                record.update(status="failed", cleanup_error="Live Hermes tracked source changed")
            if host_git(HERMES_LIVE, "rev-parse", "HEAD") != HERMES_LIVE_COMMIT:
                record.update(
                    status="failed", cleanup_error="Live Hermes revision changed during run"
                )
            environment_after = host_environment()
            (root / "host-environment-after.txt").write_text(environment_after, encoding="utf-8")
            if hashlib.sha256(environment_after.encode()).hexdigest() != HERMES_ENVIRONMENT_SHA256:
                record.update(
                    status="failed", cleanup_error="Hermes shared environment changed during run"
                )
        except Exception as exc:
            record.update(status="failed", cleanup_error=type(exc).__name__ + ": " + str(exc))
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        save("report.json", record)
        print(
            json.dumps({"status": record["status"], "report": str(root / "report.json")}),
            flush=True,
        )
    return 0 if record["status"] in {"passed", "preflight_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
