"""Bounded shared runtime for the prospective three-host continuity protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import subprocess
import sqlite3
import time
from typing import Any, TYPE_CHECKING
import zipfile
from copy import deepcopy

from .hermes_lifecycle_run import (
    read_rows,
    require_prior_queue_terminal,
    sha,
    verify_committed_inputs,
)

if TYPE_CHECKING:
    from mnemos.scopes import ScopeRegistry
    from mnemos.service import MnemosService

WHEEL_SHA256 = "8552345d2c17c4c037347961eb3043d13b20f790f54a9ccc86a096add6197b0f"
SOURCE_COMMIT = "f2f741443e60f838b392ae8ea1d1a0b4d8c6c1f2"
PROFILE = "native-three-host-v4"
PROTOCOL = "docs/plans/2026-09-20-three-host-continuity-protocol-v4.md"
HERMES_INPUTS = (
    "benchmarks/native_hosts/hermes_lifecycle_run_v10.py",
    "benchmarks/native_hosts/hermes_native_bootstrap_v6.py",
    "benchmarks/native_hosts/hermes_trace.py",
    "benchmarks/native_hosts/hermes_trace_v2.py",
    "benchmarks/native_hosts/hermes_request_observer.py",
    "benchmarks/native_hosts/ollama_readiness_v2.py",
    "docs/plans/2026-09-19-hermes-native-lifecycle-protocol-v3.md",
    "docs/plans/2026-09-19-hermes-native-lifecycle-protocol-v4.md",
    "docs/plans/2026-09-19-hermes-native-lifecycle-protocol-v5.md",
    "docs/plans/2026-09-19-hermes-native-lifecycle-protocol-v6.md",
    "docs/plans/2026-09-20-hermes-native-lifecycle-protocol-v7.md",
    "docs/plans/2026-09-20-hermes-native-lifecycle-protocol-v8.md",
    "docs/plans/2026-09-20-hermes-native-lifecycle-protocol-v9.md",
    "docs/plans/2026-09-20-hermes-native-lifecycle-protocol-v10.md",
    "docs/validation/2026-09-20/native-model-profile-v2/profile.json",
)


def verify_hermes_input_binding(repo: Path, report: dict[str, Any]) -> None:
    """Bind the accepted Hermes report to its committed executed inputs."""

    runner_commit = report.get("runner_commit")
    hashes = report.get("input_hashes")
    if not isinstance(runner_commit, str) or not runner_commit.strip():
        raise ValueError("Hermes acceptance report lacks runner_commit")
    if not isinstance(hashes, dict):
        raise ValueError("Hermes acceptance report lacks input_hashes")
    normalized = {str(key).replace("\\", "/"): value for key, value in hashes.items()}
    for relative in HERMES_INPUTS:
        path = repo / relative
        expected = normalized.get(relative)
        if not path.is_file() or not isinstance(expected, str):
            raise ValueError("Hermes acceptance report lacks required input: " + relative)
        if sha(path) != expected:
            raise ValueError("Hermes accepted input hash differs: " + relative)
    verify_committed_inputs(repo, list(HERMES_INPUTS), runner_commit)


def require_hermes_acceptance(hermes: dict[str, Any]) -> None:
    if (
        hermes.get("status") != "passed"
        or hermes.get("source_commit") != SOURCE_COMMIT
        or hermes.get("wheel_sha256") != WHEEL_SHA256
        or not hermes.get("finished_at")
        or hermes.get("cleanup_error")
    ):
        raise ValueError("Hermes acceptance report is not passed at the verified checkpoint")
    common = {
        "correlated",
        "phase_tools_allowed",
        "mutations_inspected",
        "no_foreign_tool_result",
        "real_pinned_model_request",
        "effective_context_verified_after_each_response",
        "pinned_profile_without_conflicting_override",
        "write_followup_guidance_present",
        "no_provider_error",
        "controls_unchanged",
        "native_capture_committed",
        "native_capture_retained_or_suppressed",
    }
    required = {
        "teach": common | {"teach", "single_asserted_target"},
        "recall_v1": common | {"recall_v1"},
        "correct": common | {"correct", "same_id_version_two"},
        "recall_v2": common | {"recall_v2"},
        "forget": common
        | {
            "forget",
            "target_absent",
            "target_suppressed",
            "reviewed_forget_verified",
            "reviewed_deletions_persisted",
            "no_retained_label_in_scope",
        },
        "recall_forgotten": common
        | {
            "recall_forgotten",
            "target_absent",
            "target_suppressed",
            "no_forgotten_label_in_request",
            "no_retained_label_in_scope",
        },
        "source-replay": {
            "original_write_suppressed",
            "native_replay_flushed",
            "no_provider_error",
        },
    }
    phases = hermes.get("phases", [])
    if len(phases) != len(required) or {row.get("phase") for row in phases} != set(required):
        raise ValueError("Hermes acceptance lacks all required native phases")
    for row in phases:
        gates = row.get("gates", {})
        if (
            row.get("exit_code") != 0
            or not required[row["phase"]].issubset(gates)
            or any(value is not True for value in gates.values())
        ):
            raise ValueError("Hermes acceptance lacks required successful phase gates")


class ThreeHostRuntime:
    def __init__(
        self,
        root: Path,
        repo: Path,
        checks: Path,
        installed_checks: Path,
        prior_queue_state: Path,
        excluded_pids: list[int],
        hermes_acceptance: Path,
    ):
        self.root = root.resolve()
        self.repo = repo.resolve()
        self.checks = checks.resolve()
        self.installed_checks = installed_checks.resolve()
        self.prior_queue_state = prior_queue_state.resolve()
        self.excluded_pids = list(excluded_pids)
        self.hermes_acceptance = hermes_acceptance.resolve()
        if self.root.is_relative_to(self.repo):
            raise ValueError("runtime root must be outside the repository")
        self.project = self.root / "project"
        self.registry_path = self.root / "scopes.json"
        self.descriptor = self.root / "runtime.json"
        self.settings = self.root / "settings.toml"
        self.plugin_snapshot = self.root / "plugin-snapshot"
        self.profile = PROFILE
        self.contexts: dict[str, dict[str, Any]] = {}
        self._registry: ScopeRegistry | None = None
        self._verified = False
        self._plugin_bytes: dict[str, bytes] = {}
        self.service: MnemosService | None = None
        self.dispatch_records: list[dict[str, Any]] = []
        self.current_phase = "unassigned"

    def verify(self) -> dict[str, Any]:
        self._verified = False
        self._plugin_bytes = {}
        protocol = self.repo / PROTOCOL
        text = protocol.read_text(encoding="utf-8")
        if SOURCE_COMMIT not in text or WHEEL_SHA256 not in text:
            raise ValueError(
                "three-host protocol binding does not contain the frozen source and wheel"
            )
        checks = json.loads(self.checks.read_text(encoding="utf-8"))
        installed = json.loads(self.installed_checks.read_text(encoding="utf-8"))
        if (
            checks.get("status") != "passed"
            or checks.get("source_commit") != SOURCE_COMMIT
            or installed.get("status") != "passed"
        ):
            raise ValueError("package verification reports must both pass")
        wheels = [
            row for row in checks.get("artifacts", []) if str(row.get("file", "")).endswith(".whl")
        ]
        if len(wheels) != 1 or wheels[0].get("sha256") != WHEEL_SHA256:
            raise ValueError("package report must bind exactly one verified wheel")
        wheel = Path(wheels[0]["file"])
        if sha(wheel) != WHEEL_SHA256:
            raise ValueError("verified wheel bytes do not match the three-host protocol")
        bindings = installed.get("bindings_after", {})
        if bindings.get("commit") != SOURCE_COMMIT or bindings.get("wheel_sha256") != WHEEL_SHA256:
            raise ValueError("installed binding differs from the protocol checkpoint")
        if bindings.get("checks_sha256") != sha(self.checks):
            raise ValueError("installed report was produced from different package checks")
        if Path(sys.prefix).resolve() != Path(installed.get("environment", "")).resolve():
            raise ValueError("verified interpreter prefix differs")
        prior = require_prior_queue_terminal(self.prior_queue_state, self.excluded_pids)
        import mnemos

        if not Path(mnemos.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError("imported mnemos escaped the verified environment")
        with zipfile.ZipFile(wheel) as archive:
            installed_root = Path(mnemos.__file__).resolve().parent.parent
            for member in archive.namelist():
                if member.startswith("mnemos/") and not member.endswith("/"):
                    if (installed_root / member).read_bytes() != archive.read(member):
                        raise ValueError("installed mnemos bytes differ from the verified wheel")
        source_checkout = Path(checks.get("source_checkout", ""))
        if not (source_checkout / ".claude-plugin/plugin.json").is_file():
            raise ValueError("verified plugin source checkout is unavailable")
        tracked = subprocess.check_output(
            [
                "git",
                "-C",
                str(self.repo),
                "ls-tree",
                "-r",
                "--name-only",
                SOURCE_COMMIT,
                "--",
                ".claude-plugin",
                "hooks",
                "commands",
                "agents",
                "skills",
            ],
            text=True,
        ).splitlines()
        required_plugin = {
            ".claude-plugin/plugin.json",
            ".claude-plugin/run_mnemos_hook.py",
            ".claude-plugin/run_mnemos_mcp.py",
            "hooks/hooks.json",
        }
        if not required_plugin.issubset(tracked):
            raise ValueError("Verified source lacks required plugin files")
        for relative in tracked:
            path = source_checkout / relative
            if not path.resolve().is_relative_to(source_checkout.resolve()):
                raise ValueError("Plugin source path escapes its verified checkout")
            committed = subprocess.check_output(
                ["git", "-C", str(self.repo), "show", f"{SOURCE_COMMIT}:{relative}"],
                stderr=subprocess.DEVNULL,
            )
            raw = path.read_bytes()
            if committed.replace(b"\r\n", b"\n") != raw.replace(b"\r\n", b"\n"):
                raise ValueError("verified plugin source differs from committed bytes")
            self._plugin_bytes[relative] = raw
        hermes = json.loads(self.hermes_acceptance.read_text(encoding="utf-8"))
        verify_hermes_input_binding(self.repo, hermes)
        require_hermes_acceptance(hermes)
        cohort_path = self.hermes_acceptance.parent.parent / "cohort.json"
        cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
        attempts = cohort.get("attempts", [])
        if (
            cohort.get("status") not in {"passed", "completed_with_failures"}
            or not cohort.get("finished_at")
            or cohort.get("planned_attempts") != 3
            or len(attempts) != 3
            or any(row.get("cleanup_established") is not True for row in attempts)
            or sum(
                row.get("report_sha256") == sha(self.hermes_acceptance)
                and row.get("status") == "passed"
                for row in attempts
            )
            != 1
        ):
            raise ValueError("Hermes prerequisite lacks its terminal cohort and cleanup evidence")
        self._verified = True
        return {
            "source_commit": SOURCE_COMMIT,
            "wheel_sha256": WHEEL_SHA256,
            "wheel": str(wheel),
            "prior_queue": prior,
            "hermes": hermes,
            "checks_sha256": sha(self.checks),
            "installed_checks_sha256": sha(self.installed_checks),
            "hermes_acceptance_sha256": sha(self.hermes_acceptance),
            "hermes_cohort_sha256": sha(cohort_path),
            "hermes_cohort_results": [row["status"] for row in attempts],
            "plugin_files": {
                name: hashlib.sha256(raw).hexdigest() for name, raw in self._plugin_bytes.items()
            },
        }

    def prepare(self) -> None:
        if not self._verified or not self._plugin_bytes:
            raise RuntimeError("verify() must pass before prepare()")
        if self.root.exists():
            raise FileExistsError("runtime root must be fresh")
        self.root.mkdir(parents=True)
        from mnemos.service import _protect_private_path

        _protect_private_path(self.root)
        (self.root / "project").mkdir()
        (self.root / "foreign-project").mkdir()
        self.settings.write_text(
            "[modules]\nsurprisal=false\nmutable_rag=false\naffective=false\nsleep=false\nspreading=false\n",
            encoding="utf-8",
        )
        (self.root / "invalid-autostart.toml").write_text("[invalid\n", encoding="utf-8")
        for relative, raw in self._plugin_bytes.items():
            target = self.plugin_snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        from mnemos.scopes import ScopeRegistry

        registry = ScopeRegistry()
        for label in ("project", "foreign-project"):
            path = self.root / label
            identity = registry.register_project(path, user_profile_id=self.profile, enabled=True)
            identity.aliases.append(str(path))
            if label == "project":
                identity.aliases.append(
                    "/mnt/"
                    + str(path).replace("\\", "/")[0].lower()
                    + str(path).replace("\\", "/")[2:]
                )
            self.contexts[label] = registry.resolve(
                path, user_profile_id=self.profile, session_id="controller", host="manual"
            ).model_dump(mode="json")
        registry.save(self.registry_path)
        self._registry = registry

    def start(self) -> dict[str, Any]:
        if self.service is not None:
            return self.service.health()
        if not self._verified or self._registry is None:
            raise RuntimeError("prepare() is required before start()")
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
        registry = self._registry
        runtime = self

        class TracedLocalMemoryService(LocalMemoryService):
            async def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
                try:
                    result = await super().dispatch(operation, payload)
                except Exception as exc:
                    runtime.dispatch_records.append(
                        {
                            "phase": runtime.current_phase,
                            "operation": operation,
                            "payload": deepcopy(payload),
                            "error_type": type(exc).__name__,
                            "source": "service-dispatch",
                        }
                    )
                    raise
                runtime.dispatch_records.append(
                    {
                        "phase": runtime.current_phase,
                        "operation": operation,
                        "payload": deepcopy(payload),
                        "result": deepcopy(result),
                        "source": "service-dispatch",
                    }
                )
                return result

        def factory() -> LocalMemoryService:
            engine = MnemosEngine(
                config=config,
                llm=UnavailableLLMProvider(),
                embedder=OllamaEmbeddingProvider(base_url="http://127.0.0.1:11434", timeout=30),
                store=SQLiteStore(str(self.root / "memory.sqlite")),
                scope_registry=registry,
                require_scope_context=True,
                owns_providers=True,
            )
            return TracedLocalMemoryService(engine, registry)

        self.service = MnemosService(
            dispatcher_factory=factory,
            descriptor_path=self.descriptor,
            startup_timeout=10,
            dispatch_timeout=45,
        )
        return self.service.start()

    def stop(self) -> None:
        if self.service is not None:
            self.service.stop()
            self.service = None

    def set_phase(self, name: str) -> None:
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name):
            raise ValueError("phase must be a simple identifier")
        self.current_phase = name

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.service is None:
            raise RuntimeError("runtime is not started")
        before = len(self.dispatch_records)
        result = self.service.submit(operation, dict(payload))
        if len(self.dispatch_records) == before:
            self.dispatch_records.append(
                {
                    "phase": self.current_phase,
                    "operation": operation,
                    "payload": deepcopy(payload),
                    "result": deepcopy(result),
                    "source": "controller-dispatch",
                }
            )
        return result

    def maintain(self) -> dict[str, list[dict[str, Any]]]:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            self.dispatch("maintenance", {})
            state = self.rows()
            if any(row["last_error"] for row in state["jobs"]):
                raise RuntimeError("maintenance reported a processing error")
            if all(row["status"] == "completed" for row in state["jobs"]):
                return state
            time.sleep(0.05)
        raise TimeoutError("maintenance did not finish within 60 seconds")

    def rows(self) -> dict[str, list[dict[str, Any]]]:
        database = self.root / "memory.sqlite"
        if not database.is_file():
            raise ValueError("Native runtime database does not exist")
        rows = read_rows(database)
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            rows["edges"] = [
                dict(row)
                for row in connection.execute(
                    "SELECT source_id,target_id,edge_type,weight FROM memory_edges ORDER BY source_id,target_id,edge_type"
                )
            ]
        return rows
