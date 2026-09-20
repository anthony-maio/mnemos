"""Bounded synthetic model comparison for the reviewed Mnemos forget workflow.

This diagnostic uses an actual installed Mnemos service and Hermes provider
dispatch, but synthetic facts and a deterministic local retriever. It is not a
native-host acceptance run and never claims native model behavior.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sqlite3
import sys
import time
from typing import Any, cast
from types import SimpleNamespace
import urllib.error
import urllib.request
import zipfile

from mnemos.adapters.hermes_tools import dispatch_tool, get_tool_schemas
from mnemos.client import MnemosClient
from mnemos.config import MnemosConfig, ModuleConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.service import MnemosService
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore

CASES = (
    {
        "name": "backup-window",
        "old": "The backup window is olive-harbor-316.",
        "new": "The backup window is silver-marsh-529.",
        "control": "The backup checksum is granite-safe-184.",
        "foreign": "The foreign backup window is hidden-cove-773.",
    },
    {
        "name": "deployment-lane",
        "old": "The deployment lane is copper-meadow-617.",
        "new": "The deployment lane is violet-canal-283.",
        "control": "The deployment checksum is granite-safe-296.",
        "foreign": "The foreign deployment lane is hidden-cove-884.",
    },
    {
        "name": "archive-bucket",
        "old": "The archive bucket is amber-orbit-451.",
        "new": "The archive bucket is cobalt-field-628.",
        "control": "The archive checksum is granite-safe-357.",
        "foreign": "The foreign archive bucket is hidden-cove-995.",
    },
)
VERIFIED_RUNTIME_COMMIT = "e454a85f7b4f79ce2b483f738d68d88a15bd72d6"
VERIFIED_WHEEL_SHA256 = "94011300cda669ddcabefe9e9077af0f2aa01b6ed5f23bc13c8e6b7f2a5fbf35"
TOOLS = get_tool_schemas()
MODEL_OPTIONS = {
    "temperature": 0,
    "seed": 42,
    "presence_penalty": 0,
    "num_ctx": 32768,
    "num_predict": 2048,
    "top_k": 20,
    "top_p": 0.95,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _api(endpoint: str, route: str, payload: Any | None, timeout: float) -> Any:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/api/" + route,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {error.code}: {body}") from error


def _trace_module() -> Any:
    path = Path(__file__).with_name("hermes_trace_v2.py")
    spec = importlib.util.spec_from_file_location("mnemos_development_frozen_trace_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen trace evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_record(wheel: Path) -> dict[str, Any]:
    import mnemos

    origin = Path(mnemos.__file__).resolve()
    prefix = Path(sys.prefix).resolve()
    if not origin.is_relative_to(prefix):
        raise RuntimeError("mnemos import is outside the verified interpreter")
    if not sys.flags.isolated:
        raise RuntimeError("Use the installed interpreter with -I")
    checked = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not name.startswith("mnemos/") or name.endswith("/"):
                continue
            installed = origin.parent.parent / name
            if not installed.is_file() or installed.read_bytes() != archive.read(name):
                raise RuntimeError(f"Installed package differs from pinned wheel: {name}")
            checked[name] = _sha(installed)
    if not checked:
        raise RuntimeError("Pinned wheel contains no package files")
    origins = {}
    for name, module in list(sys.modules.items()):
        if name == "mnemos" or name.startswith("mnemos."):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if not path.is_relative_to(origin.parent):
                    raise RuntimeError(f"Mixed package origin: {name}")
                origins[name] = str(path)
    return {
        "origin": str(origin),
        "version": getattr(mnemos, "__version__", None),
        "sys_prefix": str(prefix),
        "executable": sys.executable,
        "isolated": bool(sys.flags.isolated),
        "installed_files_sha256": checked,
        "loaded_module_origins": origins,
    }


def _context(project: Path, registry: ScopeRegistry, session: str) -> Any:
    return registry.resolve(
        project,
        user_profile_id="model-forget-development-v1",
        session_id=session,
        host="hermes",
    )


def _service(root: Path) -> tuple[MnemosService, LocalMemoryService, Any, Any, Path, Path]:
    project = root / "project"
    foreign = root / "foreign-project"
    project.mkdir()
    foreign.mkdir()
    registry = ScopeRegistry()
    registry.register_project(project, user_profile_id="model-forget-development-v1", enabled=True)
    registry.register_project(foreign, user_profile_id="model-forget-development-v1", enabled=True)
    config = MnemosConfig(
        modules=ModuleConfig(
            surprisal=False,
            mutable_rag=False,
            affective=False,
            sleep=False,
            spreading=False,
        )
    )
    store = SQLiteStore(str(root / "memory.sqlite"))
    engine = MnemosEngine(
        config=config,
        llm=UnavailableLLMProvider(),
        embedder=SimpleEmbeddingProvider(),
        store=store,
        scope_registry=registry,
        require_scope_context=True,
    )
    domain = LocalMemoryService(engine, registry)
    service = MnemosService(
        dispatcher=domain,
        descriptor_path=root / "runtime" / "descriptor.json",
        lock_path=root / "runtime" / "service.lock",
        startup_timeout=3,
        dispatch_timeout=10,
    )
    service.start()
    return (
        service,
        domain,
        _context(project, registry, "model-forget-project"),
        _context(foreign, registry, "model-forget-foreign"),
        project,
        foreign,
    )


def _seed(
    service: MnemosService,
    domain: LocalMemoryService,
    project_context: Any,
    foreign_context: Any,
    case: dict[str, str],
) -> dict[str, Any]:
    def payload(context: Any, content: str, event_id: str, role: str = "user") -> dict[str, Any]:
        return {
            "context": context.model_dump(mode="json"),
            "event_id": event_id,
            "content": content,
            "role": role,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "source_kind": "synthetic_model_forget_fixture",
            "scope": "project",
            "source_ref": event_id,
        }

    old, new = case["old"], case["new"]
    target_result = service.submit(
        "remember",
        {
            "context": project_context.model_dump(mode="json"),
            "event_id": "seed-target-" + case["name"],
            "content": old,
        },
    )
    service.submit("maintenance", {})
    target = next(row for row in _state(domain)["facts"] if row["content"] == old)
    service.submit(
        "correct",
        {
            "context": project_context.model_dump(mode="json"),
            "event_id": "seed-correction-" + case["name"],
            "fact_id": target["id"],
            "expected_version": target["version"],
            "content": new,
        },
    )
    service.submit("maintenance", {})
    for index, content in enumerate(
        (
            f"Assistant repeated the prior setting: {old}",
            f"A captured instruction recorded the prior setting {old}",
            f"Assistant confirmed the current setting: {new}",
        )
    ):
        service.submit(
            "capture",
            payload(
                project_context,
                content,
                f"seed-copy-{case['name']}-{index}",
                "user" if index == 1 else "assistant",
            ),
        )
    service.submit("maintenance", {})
    service.submit(
        "remember",
        {
            "context": project_context.model_dump(mode="json"),
            "event_id": "seed-control-" + case["name"],
            "content": case["control"],
        },
    )
    service.submit(
        "remember",
        {
            "context": foreign_context.model_dump(mode="json"),
            "event_id": "seed-foreign-" + case["name"],
            "content": case["foreign"],
        },
    )
    service.submit("maintenance", {})
    state = _state(domain)
    target_row = next(row for row in state["facts"] if row["content"] == new)
    known = [
        row
        for row in state["facts"]
        if row["id"] != target_row["id"] and (old in row["content"] or new in row["content"])
    ]
    known_ids = {row["id"] for row in known}
    known_sources = {
        row["source_event_id"]
        for row in state["evidence"]
        if row["fact_id"] in known_ids | {target_row["id"]}
    }
    control = next(row for row in state["facts"] if row["content"] == case["control"])
    foreign_control = next(row for row in state["facts"] if row["content"] == case["foreign"])
    context = project_context.model_dump(mode="json")
    known_copy_rows = []
    for row in known:
        source_ids = sorted(
            evidence["source_event_id"]
            for evidence in state["evidence"]
            if evidence["fact_id"] == row["id"]
        )
        known_copy_rows.append(
            {
                "id": row["id"],
                "content": row["content"],
                "version": row["version"],
                "required_source_event_ids": source_ids,
            }
        )
    authoritative_facts = [
        row
        for row in state["facts"]
        if (
            (
                json.loads(row.get("metadata", "{}"))
                if isinstance(row.get("metadata"), str)
                else row.get("metadata", {})
            ).get("project_id")
            == context.get("project_id")
            and (
                json.loads(row.get("metadata", "{}"))
                if isinstance(row.get("metadata"), str)
                else row.get("metadata", {})
            ).get("user_profile_id")
            == context.get("user_profile_id")
        )
    ]
    return {
        "target_id": target_row["id"],
        "known_copy_ids": sorted(known_ids),
        "known_source_event_ids": sorted(known_sources),
        "control_id": control["id"],
        "foreign_control_id": foreign_control["id"],
        "control_row": control,
        "foreign_control_row": foreign_control,
        "control_snapshot": _control_snapshot(state, control["id"]),
        "foreign_control_snapshot": _control_snapshot(state, foreign_control["id"]),
        "old": old,
        "new": new,
        "old_label": old.rstrip(".").split()[-1],
        "new_label": new.rstrip(".").split()[-1],
        "context": context,
        "labels": [old.rstrip(".").split()[-1], new.rstrip(".").split()[-1]],
        "known_fact_ids": sorted(known_ids | {target_row["id"]}),
        "authoritative_facts": authoritative_facts,
        "known_copy_rows": known_copy_rows,
        "control": case["control"],
        "foreign": case["foreign"],
        "seed_result": target_result,
    }


def _state(domain: Any) -> dict[str, list[dict[str, Any]]]:
    connection = domain.engine.store._conn  # noqa: SLF001
    connection.row_factory = __import__("sqlite3").Row
    queries = {
        "facts": "SELECT id, content, version, metadata FROM memory_chunks ORDER BY id",
        "events": "SELECT id, content, metadata FROM episodic_events ORDER BY id",
        "evidence": "SELECT fact_id, source_event_id, excerpt FROM fact_evidence ORDER BY fact_id, source_event_id",
        "revisions": "SELECT fact_id, version, content FROM fact_revisions ORDER BY fact_id, version",
        "jobs": "SELECT id, payload, result, status FROM memory_jobs ORDER BY id",
        "suppressions": "SELECT fact_id, source_event_id, reason FROM memory_suppressions ORDER BY suppression_id",
    }
    return {
        name: [dict(row) for row in connection.execute(query)] for name, query in queries.items()
    }


def _control_snapshot(state: dict[str, list[dict[str, Any]]], fact_id: str) -> dict[str, Any]:
    fact = next((row for row in state["facts"] if row["id"] == fact_id), None)
    if fact is None:
        return {"fact": None, "evidence": [], "events": [], "revisions": []}
    evidence = [row for row in state["evidence"] if row["fact_id"] == fact_id]
    sources = {row["source_event_id"] for row in evidence}
    return {
        "fact": fact,
        "evidence": evidence,
        "events": [row for row in state["events"] if row["id"] in sources],
        "revisions": [row for row in state["revisions"] if row["fact_id"] == fact_id],
    }


def _score(
    records: list[dict[str, Any]], seed: dict[str, Any], state: dict[str, list[dict[str, Any]]]
) -> dict[str, bool]:
    trace = _trace_module()
    rows: list[dict[str, Any]] = []
    gates = {
        "trace_correlated": False,
        "forget_phase_verified": False,
        "reviewed_persisted": False,
        "scoped_markers_absent": False,
    }
    try:
        pending: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        for record in records:
            if record.get("operation") == "model_response":
                for call in record["tool_calls"]:
                    if call["id"] in seen:
                        raise ValueError("duplicate transport ID")
                    seen.add(call["id"])
                    pending[call["id"]] = call
            elif record.get("operation") == "tool_result":
                call = pending.pop(record["call_id"])
                if call["name"] != record["name"] or call["arguments"] != record["args"]:
                    raise ValueError("result does not match transport call")
            else:
                raise ValueError("unsupported trace record")
        if pending or not seen:
            raise ValueError("incomplete tool trace")
        rows = trace.correlate_native_tools(records, include_positions=True)
        gates["trace_correlated"] = True
        gates["forget_phase_verified"] = all(
            trace.evaluate_phase(
                records,
                phase="forget",
                target_id=seed["target_id"],
                old_content=seed["old"],
                new_content=seed["new"],
            ).values()
        )
        gates["reviewed_persisted"] = trace.reviewed_deletions_persisted(records, state)
        gates["scoped_markers_absent"] = trace.no_retained_labels(
            state,
            context=seed["context"],
            labels=seed["labels"],
            known_fact_ids=set(seed["known_fact_ids"]),
        )
    except (TypeError, ValueError, KeyError, IndexError):
        gates["trace_correlated"] = False
    applies = [row for row in rows if row["name"] == "mnemos_forget_related"]
    allowed = {
        "mnemos_retrieve",
        "mnemos_inspect",
        "mnemos_forget_preview",
        "mnemos_forget_related",
    }
    selected_facts = {
        value for row in applies for value in row["args"].get("selected_fact_ids", [])
    }
    selected_sources = {
        value for row in applies for value in row["args"].get("selected_source_event_ids", [])
    }
    expected_sources = {
        source
        for row in seed.get("known_copy_rows", [])
        for source in row["required_source_event_ids"]
    }
    known_facts = set(seed["known_copy_ids"]) | {seed["target_id"]}
    known_sources = set(seed["known_source_event_ids"])
    gates.update(
        {
            "tools_allowed": bool(rows) and all(row["name"] in allowed for row in rows),
            "retrieved_target_before_apply": any(
                row["name"] == "mnemos_retrieve"
                and isinstance(row["result"].get("facts"), list)
                and any(
                    fact.get("id") == seed["target_id"] and fact.get("content") == seed["new"]
                    for fact in row["result"]["facts"]
                )
                and any(
                    row["result_trace_index"] < applied["call_trace_index"] for applied in applies
                )
                for row in rows
            ),
            "all_copies_selected": bool(applies) and set(seed["known_copy_ids"]) <= selected_facts,
            "required_sources_selected": bool(applies) and expected_sources <= selected_sources,
            "known_facts_absent": all(row["id"] not in known_facts for row in state["facts"]),
            "known_sources_absent": all(row["id"] not in known_sources for row in state["events"]),
            "known_facts_suppressed": known_facts
            <= {row.get("fact_id") for row in state["suppressions"]},
            "known_sources_suppressed": known_sources
            <= {row.get("source_event_id") for row in state["suppressions"]},
            "controls_retained": _control_snapshot(state, seed["control_id"])
            == seed.get("control_snapshot", {}),
            "foreign_control_retained": _control_snapshot(state, seed["foreign_control_id"])
            == seed.get("foreign_control_snapshot", {}),
        }
    )
    return gates


def _save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


async def _bounded_api(endpoint: str, route: str, payload: Any, timeout: float) -> Any:
    if timeout <= 0:
        raise TimeoutError("request deadline exhausted")
    return await asyncio.wait_for(
        asyncio.to_thread(_api, endpoint, route, payload, timeout), timeout
    )


async def _run_case(
    root: Path, endpoint: str, model: str, think: bool | str, case: dict[str, str], index: int
) -> dict[str, Any]:
    case_root = root / f"case-{index + 1}-{case['name']}"
    case_root.mkdir()
    record: dict[str, Any] = {
        "case": case["name"],
        "status": "running",
        "automatic_conversation_capture": False,
        "scope": "synthetic installed-service tool diagnostic; not native-host acceptance",
        "transport_id_normalization": "Every call receives response-N-call-M; original IDs remain in untouched raw response. These IDs are not native Hermes correlation.",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "requests": [],
        "responses": [],
        "tool_calls": [],
        "tool_results": [],
        "trace": [],
    }
    report_path = case_root / "report.json"
    _save(report_path, record)
    service = None
    domain = None
    client = None
    started = time.monotonic()
    deadline = started + 300
    try:
        service, domain, context, foreign_context, project, foreign = _service(case_root)
        client = MnemosClient(
            descriptor_path=service.descriptor_path, auto_start=False, request_timeout=60
        )
        seed = _seed(service, domain, context, foreign_context, case)
        record["seed"] = seed
        record["state_before"] = _state(domain)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "Use only the supplied mnemos_* tools. Do not add memory. Review and erase this topic and its retained conversational copies. Retrieve and inspect the asserted target, preview related forget, inspect every selected additional fact, then apply the unchanged digest and limit with exact selected IDs and required source IDs. Preserve unrelated facts.",
            },
            {
                "role": "user",
                "content": f"Forget the setting about {case['name']}, including retained conversational copies, and confirm what remains.",
            },
        ]
        for round_index in range(8):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("case deadline exceeded")
            payload = deepcopy(
                {
                    "model": model,
                    "messages": messages,
                    "tools": [{"type": "function", "function": schema} for schema in TOOLS],
                    "stream": False,
                    "think": think,
                    "options": MODEL_OPTIONS,
                }
            )
            request_started = time.monotonic()
            request_record: dict[str, Any] = {"round": round_index, "payload": payload}
            record["requests"].append(request_record)
            _save(report_path, record)
            try:
                response = await _bounded_api(endpoint, "chat", payload, min(60, remaining))
            except Exception as exc:
                request_record.update(
                    seconds=time.monotonic() - request_started,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise
            request_record.update(
                seconds=time.monotonic() - request_started,
                prompt_eval_count=response.get("prompt_eval_count"),
                eval_count=response.get("eval_count"),
            )
            record["responses"].append(deepcopy(response))
            _save(report_path, record)
            if (
                response.get("done") is not True
                or response.get("error")
                or response.get("done_reason") == "length"
            ):
                raise ValueError("model response failed or exhausted output budget")
            message = deepcopy(response.get("message", {}))
            if message.get("role") != "assistant":
                raise ValueError("model response has no assistant message")
            messages.append(message)
            tool_calls = message.get("tool_calls", [])
            if not isinstance(tool_calls, list) or len(tool_calls) > 16:
                raise ValueError("malformed or excessive tool calls")
            normalized = []
            for call_index, call in enumerate(tool_calls):
                function = call.get("function", {})
                name = function.get("name")
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if name not in {schema["name"] for schema in TOOLS} or not isinstance(
                    arguments, dict
                ):
                    raise ValueError("malformed or unsupported tool call")
                normalized.append(
                    {
                        "id": f"response-{round_index}-call-{call_index}",
                        "name": name,
                        "arguments": deepcopy(arguments),
                        "raw_call_id": call.get("id"),
                    }
                )
            record["trace"].append(
                {"operation": "model_response", "round": round_index, "tool_calls": normalized}
            )
            if not normalized:
                if not isinstance(message.get("content"), str) or not message["content"].strip():
                    raise ValueError("empty terminal assistant response")
                break
            for call in normalized:
                name, arguments, call_id = call["name"], call["arguments"], call["id"]
                record["tool_calls"].append(
                    {
                        "operation": "tool_call",
                        "round": round_index,
                        "call_id": call_id,
                        "raw_call_id": call["raw_call_id"],
                        "name": name,
                        "args": deepcopy(arguments),
                    }
                )
                _save(report_path, record)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("case deadline exceeded before tool dispatch")
                client.request_timeout = min(60, remaining)
                tool_started = time.monotonic()
                result_text = await asyncio.wait_for(
                    dispatch_tool(client, context, name, arguments), min(60, remaining)
                )
                result = json.loads(result_text)
                result_record = {
                    "operation": "tool_result",
                    "round": round_index,
                    "call_id": call_id,
                    "name": name,
                    "args": deepcopy(arguments),
                    "result": result,
                    "seconds": time.monotonic() - tool_started,
                    "raw_result_text": result_text,
                }
                record["tool_results"].append(result_record)
                record["trace"].append(deepcopy(result_record))
                # Ollama's native API associates a tool response by tool_name and order.
                messages.append({"role": "tool", "tool_name": name, "content": result_text})
                _save(report_path, record)
                if (
                    not isinstance(result, dict)
                    or result.get("status")
                    in {"degraded", "rejected", "error", "failed", "failure"}
                    or result.get("error")
                ):
                    raise ValueError("unsuccessful tool result; case fails without recovery retry")
        else:
            raise RuntimeError("tool round limit exceeded")
        if time.monotonic() > deadline:
            raise TimeoutError("case deadline exceeded after final response")
        record["state"] = _state(domain)
        record["gates"] = _score(record["trace"], seed, record["state"])
        record["status"] = "passed" if all(record["gates"].values()) else "failed"
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        cleanup_errors = []
        if client is not None:
            try:
                await client.close()
            except Exception as exc:
                cleanup_errors.append("client: " + repr(exc))
        if service is not None:
            try:
                service.stop()
            except Exception as exc:
                cleanup_errors.append("service: " + repr(exc))
        if domain is not None:
            try:
                with sqlite3.connect(
                    (case_root / "memory.sqlite").as_uri() + "?mode=ro", uri=True
                ) as connection:
                    view = SimpleNamespace(
                        engine=SimpleNamespace(store=SimpleNamespace(_conn=connection))
                    )
                    record["state_after_cleanup"] = _state(view)
                    if "seed" in record:
                        record["gates_after_cleanup"] = _score(
                            record["trace"], record["seed"], record["state_after_cleanup"]
                        )
                        if not all(record["gates_after_cleanup"].values()):
                            record["status"] = "failed"
            except Exception as exc:
                cleanup_errors.append("state: " + repr(exc))
            try:
                cast(SQLiteStore, domain.engine.store).close()
            except Exception as exc:
                cleanup_errors.append("store: " + repr(exc))
        record["cleanup"] = {"ok": not cleanup_errors, "errors": cleanup_errors}
        if cleanup_errors:
            record["status"] = "failed"
        record["elapsed_seconds"] = time.monotonic() - started
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        _save(report_path, record)
    return record


async def _main(args: argparse.Namespace) -> int:
    root = args.output.resolve()
    repo = Path(__file__).resolve().parents[2]
    if root.is_relative_to(repo) or Path.cwd().resolve().is_relative_to(repo):
        raise ValueError("Run from an external working directory with an external output directory")
    root.mkdir(exist_ok=False)
    record: dict[str, Any] = {
        "status": "running",
        "model": args.model,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cases": [],
        "preflight": {},
    }
    report_path = root / "report.json"
    _save(report_path, record)
    try:
        wheel = args.wheel.resolve()
        if not wheel.is_file() or _sha(wheel) != VERIFIED_WHEEL_SHA256:
            raise ValueError("Wheel does not match verified e454a85 package")
        record["module"] = _module_record(wheel)
        protocol = repo / "docs/plans/2026-09-19-local-model-forget-development-v1.md"
        profiles_path = repo / "docs/plans/2026-09-19-local-model-forget-profiles-v1.json"
        profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
        profile = next(row for row in profiles["profiles"] if row["model"] == args.model)
        think = "low" if args.think == "low" else False
        if (
            args.endpoint != profiles["endpoint"]
            or args.expected_digest != profile["digest"]
            or think != profile["think"]
            or args.num_ctx != 32768
            or args.repetitions != 1
            or MODEL_OPTIONS != profiles["options"]
        ):
            raise ValueError("Arguments differ from frozen model profile")
        inputs = [
            Path(__file__).resolve(),
            protocol,
            profiles_path,
            Path(__file__).with_name("hermes_trace_v2.py").resolve(),
        ]
        relative = [p.relative_to(repo).as_posix() for p in inputs]
        for name in relative:
            subprocess.run(
                ["git", "-C", str(repo), "ls-files", "--error-unmatch", name],
                check=True,
                capture_output=True,
            )
        subprocess.run(
            ["git", "-C", str(repo), "diff", "--exit-code", "HEAD", "--", *relative],
            check=True,
            capture_output=True,
        )
        record.update(
            expected_digest=args.expected_digest,
            think=think,
            num_ctx=args.num_ctx,
            repetitions=1,
            options=MODEL_OPTIONS,
            verified_runtime_commit=VERIFIED_RUNTIME_COMMIT,
            verified_wheel_sha256=VERIFIED_WHEEL_SHA256,
            wheel_path=str(wheel),
            source_commit=subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip(),
            input_hashes={p.relative_to(repo).as_posix(): _sha(p) for p in inputs},
        )
        _save(report_path, record)
        for route, payload in (
            ("version", None),
            ("tags", None),
            ("show", {"model": args.model}),
            ("ps", None),
        ):
            record["preflight"][route] = await _bounded_api(args.endpoint, route, payload, 20)
            _save(report_path, record)
        tags = record["preflight"]["tags"]
        selected = next(
            (row for row in tags.get("models", []) if row.get("name") == args.model), None
        )
        if selected is None or selected.get("digest") != args.expected_digest:
            raise ValueError("Selected model digest changed")
        if record["preflight"]["version"].get("version") != profiles["ollama_version"]:
            raise ValueError("Ollama version changed")
        _save(root / "run-inputs.json", record)
        for index, case in enumerate(CASES):
            result = await _run_case(root, args.endpoint, args.model, think, case, index)
            record["cases"].append(
                {
                    "name": result["case"],
                    "status": result["status"],
                    "gates": result.get("gates"),
                    "error": result.get("error"),
                    "cleanup": result.get("cleanup"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                }
            )
            _save(report_path, record)
            print(json.dumps(record["cases"][-1]), flush=True)
        record["model_ps_after"] = await _bounded_api(args.endpoint, "ps", None, 20)
        record["tags_after"] = await _bounded_api(args.endpoint, "tags", None, 20)
        after = next(row for row in record["tags_after"]["models"] if row["name"] == args.model)
        if after["digest"] != args.expected_digest:
            raise ValueError("Selected model digest changed during run")
        if any(_sha(p) != record["input_hashes"][p.relative_to(repo).as_posix()] for p in inputs):
            raise ValueError("Frozen inputs changed during run")
        record["status"] = (
            "passed"
            if len(record["cases"]) == 3 and all(r["status"] == "passed" for r in record["cases"])
            else "failed"
        )
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        _save(report_path, record)
    return 0 if record["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["qwen3.5:9b", "gpt-oss:20b"], required=True)
    parser.add_argument("--expected-digest", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--think", choices=["false", "low"], default="false")
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--repetitions", type=int, choices=[1], default=1)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
