"""Run the prospective native compact recall screen in one disposable root."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.native_hosts import compact_recall as fixture
from benchmarks.native_hosts import native_processes as native
from benchmarks.native_hosts import shared_trace as trace
from mnemos.config import MnemosConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.service import MnemosService, _protect_private_path
from mnemos.utils.embeddings import EmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore

PROTOCOL = "docs/plans/2026-11-07-native-compact-v1.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def checkpoint_helper() -> Any:
    spec = importlib.util.spec_from_file_location(
        "native_compact_checkpoint", ROOT / "scripts/check-native-explicit.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FixtureEmbedding(EmbeddingProvider):
    dim = 384

    def embed(self, text: str) -> list[float]:
        return [1.0] + [0.0] * 383

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def snapshot(service: MnemosService) -> list[dict[str, Any]]:
    dispatcher = service._dispatcher
    if dispatcher is None:
        raise RuntimeError("Cannot snapshot a service without a dispatcher")
    store = dispatcher.engine.store
    return [
        {
            "id": chunk.id,
            "version": chunk.version,
            "content": chunk.content,
            "embedding": chunk.embedding,
            "metadata": chunk.metadata,
            "evidence": [e.model_dump(mode="json") for e in store.list_fact_evidence(chunk.id)],
        }
        for chunk in sorted(store.get_all(), key=lambda item: item.id)
    ]


def prepare(root: Path) -> tuple[MnemosService, native.PhaseEnvironment, dict[str, Any]]:
    project, foreign = root / "project", root / "foreign"
    for directory in (project, foreign):
        directory.mkdir()
        subprocess.run(["git", "init", "--quiet", str(directory)], check=True, capture_output=True)
    registry = ScopeRegistry(profiles=[fixture.PROFILE])
    entry = registry.register_project(project, user_profile_id=fixture.PROFILE, enabled=True)
    old_id, entry.id = entry.id, fixture.PROJECT_ID
    registry.projects.pop(old_id)
    registry.projects[entry.id] = entry
    registry.register_project(foreign, user_profile_id=fixture.PROFILE, enabled=True)
    registry.save(root / "scopes.json")
    settings = root / "settings.toml"
    settings.write_text(
        '[judgment]\nprovider="none"\n[modules]\nsurprisal=false\nmutable_rag=false\n'
        'affective=false\nsleep=false\nspreading=false\n[retrieval]\nprofile="baseline"\n'
        'affective_enabled=false\nspreading_enabled=false\nquery_state_source="none"\n',
        encoding="utf-8",
    )
    (root / "plugin-snapshot").mkdir()
    config = MnemosConfig()
    config.retrieval.profile = "baseline"
    config.retrieval.affective_enabled = False
    config.retrieval.spreading_enabled = False
    config.retrieval.query_state_source = "none"
    engine = MnemosEngine(
        config=config,
        llm=UnavailableLLMProvider(),
        embedder=FixtureEmbedding(),
        store=SQLiteStore(str(root / "memory.sqlite")),
    )
    service = MnemosService(
        dispatcher=LocalMemoryService(engine, registry),
        descriptor_path=root / "runtime.json",
        lock_path=root / "runtime.lock",
    )
    service.start()
    try:
        context = registry.resolve(
            project, user_profile_id=fixture.PROFILE, session_id="fixture", host="manual"
        )
        foreign_context = registry.resolve(
            foreign, user_profile_id=fixture.PROFILE, session_id="foreign", host="manual"
        )
        assert (
            context.project_id == fixture.PROJECT_ID
            and foreign_context.project_id != context.project_id
        )
        for i, fact in enumerate((*fixture.FACTS, fixture.FOREIGN)):
            selected = context if i < len(fixture.FACTS) else foreign_context
            result = service.submit(
                "remember",
                {
                    "event_id": f"native-compact-fact-{i}",
                    "context": selected.model_dump(mode="json"),
                    "content": fact,
                    "role": "user",
                    "scope": "project",
                    "source_ref": f"fixture:fact-{i}",
                },
            )
            assert result["status"] == "committed"
        save(root / "seed-maintenance.json", service.submit("maintenance", {}))
        for chunk in engine.store.get_all():
            if chunk.content in fixture.FACTS[2:4]:
                chunk.metadata["assertion_status"] = "conflict"
                engine.store.update(chunk.id, chunk)
        request = {
            "query": fixture.arguments("full")["query"],
            "context": context.model_dump(mode="json"),
            "allowed_scopes": ["project"],
            "top_k": 5,
            "max_context_tokens": 1200,
            "view": "memory",
        }
        expected = service.submit("retrieve", request)
        save(root / "expected-retrieval.json", expected)
        assert {f["content"] for f in expected["facts"]} == set(fixture.FACTS)
        assert fixture.FOREIGN not in json.dumps(expected)
        save(root / "expected-retrieval.json", expected)
        save(root / "before.json", snapshot(service))
        save(
            root / "fixture.json",
            {
                "facts": fixture.FACTS,
                "foreign": fixture.FOREIGN,
                "expected_answer": fixture.EXPECTED,
                "configuration": config.model_dump(mode="json"),
            },
        )
        phase = native.PhaseEnvironment(
            root,
            project,
            root / "scopes.json",
            root / "runtime.json",
            settings,
            fixture.PROFILE,
            root / "plugin-snapshot",
        )
        return service, phase, expected
    except BaseException:
        descriptor = service.descriptor
        port = descriptor["port"] if descriptor is not None else None
        service.stop()
        if port is None:
            save(root / "prepare-cleanup.json", {"port": None, "closed": None})
        else:
            with socket.socket() as connection:
                save(
                    root / "prepare-cleanup.json",
                    {"port": port, "closed": connection.connect_ex(("127.0.0.1", port)) != 0},
                )
        raise


def observe(
    root: Path, process: dict[str, Any], form: str, expected: dict[str, Any]
) -> dict[str, Any]:
    source = Path(process["session_file"])
    provenance = trace.TraceProvenance(
        "codex", process["pid"], process["session_id"], str(source), sha(source)
    )
    traces = list(Path(process["rollout_trace_root"]).iterdir())
    assert len(traces) == 1 and traces[0].is_dir()
    calls, hashes = trace.parse_codex_code_mode_trace(
        traces[0], provenance=provenance, model=fixture.MODEL
    )
    trace.require_tool_sequence(calls, ["mnemos_retrieve"])
    save(root / "trace-hashes.json", hashes)
    call = calls[0]
    args, result = dict(call.arguments), dict(call.result)
    save(
        root / "normalized-call.json",
        {
            "arguments": args,
            "result": result,
            "model_call_correlated": call.model_call_correlated,
            "provider_call_id": call.provider_call_id,
            "call_id": call.call_id,
        },
    )
    args_match = args == fixture.arguments(form)
    assert result["context"] == expected["context"]
    assert result["evidence_context"] == expected["evidence_context"]
    records = result["facts"] if form == "full" else result["references"]
    assert [(f["id"], f["version"]) for f in records] == [
        (f["id"], f["version"]) for f in expected["facts"]
    ]
    rows = [json.loads(line) for line in source.read_bytes().splitlines() if line.strip()]
    assert "FORBIDDEN_FOREIGN_PROJECT" not in json.dumps(rows)
    assert "FORBIDDEN_FOREIGN_PROJECT" not in json.dumps(result)
    outputs = [
        row["payload"]
        for row in rows
        if row.get("type") == "response_item"
        and row.get("payload", {}).get("type") == "custom_tool_call_output"
    ]
    save(root / "model-visible-outputs.json", outputs)
    stream = [
        json.loads(line)
        for line in (root / "stdout.jsonl").read_bytes().splitlines()
        if line.strip()
    ]
    messages = [
        r["item"]["text"]
        for r in stream
        if r.get("type") == "item.completed" and r.get("item", {}).get("type") == "agent_message"
    ]
    answer = "\n".join(messages)
    save(root / "answer.json", {"messages": messages, "grade": fixture.grade_answer(answer)})
    raw_texts = []
    for relative in hashes:
        if not relative.startswith("payloads/"):
            continue
        value = json.loads((traces[0] / relative).read_bytes())
        if (
            isinstance(value, dict)
            and isinstance(value.get("result"), dict)
            and value["result"].get("Ok")
        ):
            content = value["result"]["Ok"].get("content", [])
            texts = [part["text"] for part in content if part.get("type") == "text"]
            if len(texts) == 1:
                try:
                    if json.loads(texts[0]) == result:
                        raw_texts.append(texts[0])
                except ValueError:
                    pass
    assert len(raw_texts) == 1
    save(root / "mcp-reply.json", {"text": raw_texts[0]})
    return {
        "format": form,
        "arguments_match": args_match,
        "facts": len(records),
        "context_equal": True,
        "reported_usage": fixture.validate_usage(process["reported_usage"]),
        "native_seconds": process["seconds"],
        "answer": fixture.grade_answer(answer),
        "mcp_text_bytes": len(raw_texts[0].encode()),
        "model_visible_output_count": len(outputs),
        "model_visible_output_json_bytes": len(
            json.dumps(outputs, ensure_ascii=False, separators=(",", ":")).encode()
        ),
        "model_call_correlated": call.model_call_correlated,
    }


def main(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    if root.exists() or root.is_relative_to(ROOT):
        raise ValueError("Output must be fresh and outside the checkout")
    checkpoint = checkpoint_helper()
    auth_hash = sha(args.auth) if not args.preflight else None
    if not args.preflight:
        checkpoint.codex_auth_preflight(args.auth)
        assert sha(args.codex) == checkpoint.CODEX_BINARY_SHA256
        assert (
            sha(args.codex.with_name("codex-code-mode-host.exe"))
            == checkpoint.CODE_MODE_BINARY_SHA256
        )
        version = subprocess.check_output([str(args.codex), "--version"], text=True).strip()
        assert version == checkpoint.CODEX_VERSION
    root.mkdir(parents=True)
    _protect_private_path(root)
    inputs = sorted((ROOT / "mnemos").rglob("*.py")) + [
        ROOT / PROTOCOL,
        Path(__file__),
        ROOT / "scripts/check-native-explicit.py",
        ROOT / "benchmarks/native_hosts/compact_recall.py",
        ROOT / "benchmarks/native_hosts/native_processes.py",
        ROOT / "benchmarks/native_hosts/shared_trace.py",
        ROOT / "tests/test_native_compact_recall.py",
    ]
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    report: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": bindings,
        "model": fixture.MODEL,
        "preflight_only": args.preflight,
        "phases": [],
    }
    save(root / "report.json", report)
    service = None
    port = None
    try:
        service, phase, expected = prepare(root)
        descriptor = service.descriptor
        if descriptor is None:
            raise RuntimeError("Prepared service has no runtime descriptor")
        port = descriptor["port"]
        report["service_port"] = port
        origin = subprocess.check_output(
            [sys.executable, "-I", "-c", "import mnemos; print(mnemos.__file__)"],
            cwd=phase.project,
            env=native.clean_environment({}),
            text=True,
        ).strip()
        assert Path(origin).resolve() == (ROOT / "mnemos/__init__.py").resolve()
        report["package_origin"] = origin
        report["host_executable_sha256"] = None if args.preflight else sha(args.codex)
        before = snapshot(service)
        for index, form in enumerate(()) if args.preflight else enumerate(fixture.ORDER):
            assert {name: sha(ROOT / name) for name in bindings} == bindings
            assert sha(args.auth) == auth_hash
            checkpoint.codex_auth_preflight(args.auth)
            name = f"session-{index + 1}-{form}"
            row: dict[str, Any] = {"phase": name, "format": form, "status": "running"}
            report["phases"].append(row)
            save(root / "report.json", report)
            save(root / f"{name}-prompt.json", {"prompt": fixture.prompt(form)})
            try:
                result = native.run_codex_exec_phase(
                    phase,
                    phase=name,
                    prompt=fixture.prompt(form),
                    executable=args.codex,
                    auth_source=args.auth,
                    model=fixture.MODEL,
                    allowed_tools={"mnemos_retrieve"},
                )
            except Exception as exc:
                row.update(status="native_failed", error_type=type(exc).__name__)
                retained = json.loads((root / name / "process.json").read_bytes())
                if retained.get("cleanup_errors") or not retained.get("auth_copy_removed"):
                    raise
            else:
                row.update(observe(root / name, result, form, expected), status="observed")
            assert snapshot(service) == before
            save(root / "report.json", report)
            print(
                json.dumps(
                    {
                        "phase": name,
                        "status": row["status"],
                        "answer_passed": row.get("answer", {}).get("passed"),
                    }
                ),
                flush=True,
            )
        save(root / "after.json", snapshot(service))
        assert snapshot(service) == before
        assert {name: sha(ROOT / name) for name in bindings} == bindings
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if service is not None:
            service.stop()
        if port:
            with socket.socket() as connection:
                report["service_port_closed"] = connection.connect_ex(("127.0.0.1", port)) != 0
        report["auth_source_unchanged"] = args.preflight or sha(args.auth) == auth_hash
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(root / "report.json", report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "service_port_closed": report.get("service_port_closed"),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--codex",
        type=Path,
        default=Path("C:/Users/Ant/AppData/Local/OpenAI/Codex/bin/2993767bd1dfb009/codex.exe"),
    )
    parser.add_argument("--auth", type=Path, default=Path("C:/Users/Ant/.codex/auth.json"))
    parser.add_argument("--preflight", action="store_true")
    main(parser.parse_args())
