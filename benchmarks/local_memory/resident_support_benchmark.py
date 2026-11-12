"""One frozen local recall-plus-encoder capacity attempt, not semantic grading."""

import asyncio
from copy import deepcopy
import hashlib
from importlib.metadata import distributions
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
import zipfile

REPO = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO / ".venv/bidirectional-support-capacity-v1"
MODEL_PYTHON = REPO / ".venv/reranker-runtime-v1/Scripts/python.exe"
SOURCE = Path(
    "C:/Users/Ant/AppData/Local/Temp/mnemos-improvement-checks-20261106/installed-pilot-command-timing-v1/candidate-0"
)
PROFILE = SOURCE.parents[1] / "profile-current-recall-v1"
WORKER = Path(__file__).with_name("resident_support_worker.py")
NAMES = (
    "empty",
    "one-short",
    "five-short",
    "token-one-source",
    "token-eight-sources",
    "byte-bound",
)


def helper(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


path_adapter = helper("resident_support_path")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def versions():
    return sorted(
        [dict(name=d.metadata["Name"], version=d.version) for d in distributions()],
        key=lambda r: r["name"].lower().replace("_", "-"),
    )


def check_bindings(files):
    errors = []
    for path, expected in files.items():
        try:
            if sha(path) != expected:
                errors.append(dict(path=path, error_type="sha256_mismatch"))
        except BaseException as exc:
            errors.append(dict(path=path, error_type=type(exc).__name__))
    return bool(files) and not errors, errors


def local_environment():
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.upper().startswith(("MNEMOS_", "GIT_"))
        and k.upper()
        not in {
            "PYTHONPATH",
            "PYTHONHOME",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "TYPESAFE_API_KEY",
            "JEV_API_KEY",
        }
    }
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
    return env


def memory_at(directory, embedder):
    from mnemos.config import MnemosConfig, ModuleConfig
    from mnemos.engine import MnemosEngine
    from mnemos.memory_service import LocalMemoryService
    from mnemos.scopes import ScopeRegistry
    from mnemos.utils.llm import UnavailableLLMProvider
    from mnemos.utils.storage import SQLiteStore

    registry = ScopeRegistry.load(directory / "scopes.json")
    store = SQLiteStore(str(directory / "memory.sqlite"))
    config = MnemosConfig(
        modules=ModuleConfig(
            surprisal=False, mutable_rag=False, affective=False, sleep=False, spreading=False
        )
    )
    engine = MnemosEngine(
        config=config,
        llm=UnavailableLLMProvider(),
        embedder=embedder,
        store=store,
        scope_registry=registry,
        require_scope_context=True,
    )
    return LocalMemoryService(
        engine, registry, registry_loader=lambda: ScopeRegistry.load(directory / "scopes.json")
    )


def prepare(root):
    from mnemos.service import _protect_private_path
    from mnemos.utils.storage import SQLiteStore

    root.mkdir(exist_ok=False)
    _protect_private_path(root)
    original = read(PROFILE / "result.json")
    base_request = json.loads((SOURCE / "observations.jsonl").read_text().splitlines()[0])[
        "request"
    ]
    base_cohort = dict(
        query=base_request["query"],
        reference_date="2026-11-11",
        candidates=[
            dict(
                id=f["id"],
                version=f["version"],
                content=f["content"],
                assertion_status=f["assertion_status"],
                sources=[
                    dict(
                        id=e["evidence_id"],
                        source_event_id=e["source_event_id"],
                        excerpt=e["excerpt"],
                    )
                    for e in sorted(f["sources"], key=lambda e: e["evidence_id"])
                ],
            )
            for f in original["facts"]
        ],
    )
    cases = []
    for name in NAMES:
        cohort, request = deepcopy(base_cohort), deepcopy(base_request)
        request["context"]["session_id"] = name
        if name == "empty":
            cohort["candidates"] = []
            request["context"] = dict(
                user_profile_id="capacity-empty",
                project_id="capacity-empty-project",
                session_id=name,
                host="manual",
            )
            request["allowed_scopes"] = ["global"]
        elif name == "one-short":
            cohort["candidates"] = cohort["candidates"][:1]
            request["top_k"] = 1
        elif name == "token-eight-sources":
            for fact in cohort["candidates"]:
                first = fact["sources"][0]
                for index in range(1, 8):
                    fact["sources"].append(
                        dict(
                            id=first["id"] + f":{index}",
                            source_event_id=first["source_event_id"] + f":{index}",
                            excerpt=f"Synthetic owned source {index}: " + first["excerpt"],
                        )
                    )
        cases.append(dict(name=name, request=request, cohort=cohort))
    save(root / "cases.json", cases)
    subprocess.run(
        [str(MODEL_PYTHON), "-I", str(WORKER), "prepare", str(root), str(MODEL_ROOT)],
        env=local_environment(),
        check=True,
        timeout=60,
    )
    cases = read(root / "cases.json")
    seed_bindings = {
        str(SOURCE / n): sha(SOURCE / n)
        for n in ("seed.sqlite", "scopes.json", "observations.jsonl")
    }
    assert not (SOURCE / "seed.sqlite-wal").exists()
    for case in cases:
        directory = root / "seeds" / case["name"]
        directory.mkdir(parents=True)
        shutil.copyfile(SOURCE / "seed.sqlite", directory / "memory.sqlite")
        registry = read(SOURCE / "scopes.json")
        registry["profiles"].append("capacity-empty")
        registry["projects"]["capacity-empty-project"] = dict(
            id="capacity-empty-project",
            root=str(root / "empty-project"),
            user_profile_id="capacity-empty",
            enabled=True,
            workspace_ids=[],
            aliases=[],
        )
        save(directory / "scopes.json", registry)
        store = SQLiteStore(str(directory / "memory.sqlite"))
        try:
            accepted = sum(
                json.loads(r[0]).get("source") != "captured_passthrough"
                for r in store._conn.execute("SELECT metadata FROM memory_chunks")
            )
            assert accepted == 10000
            for fact in case["cohort"]["candidates"]:
                original_evidence = store.list_fact_evidence(fact["id"])[0]
                original_event = store.get_episodic_event(original_evidence.source_event_id)
                for item in fact["sources"]:
                    if item["id"] == original_evidence.evidence_id:
                        store._conn.execute(
                            "UPDATE fact_evidence SET excerpt=? WHERE evidence_id=?",
                            (item["excerpt"], item["id"]),
                        )
                        store._conn.execute(
                            "UPDATE episodic_events SET content=?, normalized_content=? WHERE id=?",
                            (item["excerpt"], item["excerpt"], item["source_event_id"]),
                        )
                        store._conn.commit()
                    else:
                        metadata = {
                            **original_event.metadata,
                            "source_event_id": item["source_event_id"],
                            "session_id": item["source_event_id"],
                        }
                        event = original_event.model_copy(
                            update=dict(
                                id=item["source_event_id"],
                                event_fingerprint=path_adapter.digest(item),
                                content=item["excerpt"],
                                normalized_content=item["excerpt"],
                                metadata=metadata,
                            )
                        )
                        saved = store.append_episodic_event(event)
                        assert saved.id == event.id
                        store._conn.execute(
                            "INSERT INTO fact_evidence VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                item["id"],
                                fact["id"],
                                item["source_event_id"],
                                item["excerpt"],
                                original_evidence.source_ref,
                                original_evidence.created_at.isoformat(),
                            ),
                        )
                        store._conn.commit()
            assert store._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert not store._conn.execute("PRAGMA foreign_key_check").fetchall()
        finally:
            store.close()

    seal(root)


def seal(root):
    """Verify prepared stores and freeze them before the sole support attempt."""
    assert not (root / "freeze.json").exists() and not (root / "attempt.started").exists()
    cases = read(root / "cases.json")
    seed_bindings = {
        str(SOURCE / n): sha(SOURCE / n)
        for n in ("seed.sqlite", "scopes.json", "observations.jsonl")
    }

    # Prepare real BGE retrieval and tokenizer bindings without support inference.
    async def verify():
        from mnemos.contracts import RetrievalRequest
        from mnemos.utils.embeddings import FastembedProvider

        embedder = FastembedProvider()
        for case in cases:
            memory = memory_at(root / "seeds" / case["name"], embedder)
            try:
                request = RetrievalRequest.model_validate(case["request"])
                result = await memory.retrieve(request)
                assert not result.get("degraded")
                _, actual = path_adapter.snapshot(memory, request, result["facts"])
                assert actual == case["cohort"], case["name"]
            finally:
                await memory.close()

    asyncio.run(verify())
    package_path = REPO / ".venv/inspection-snapshot-package-v1/package.json"
    package = read(package_path)
    files = dict(seed_bindings)
    files.update(
        {str(Path(package["installed_target"]) / p): h for p, h in package["package_files"].items()}
    )
    old = read(MODEL_ROOT / "freeze.json")
    historical_plan = REPO / "docs/plans/2026-11-11-bidirectional-support-design.md"
    archive = REPO / ".venv/bidirectional-support-capacity-v1-evidence.zip"
    with zipfile.ZipFile(archive) as evidence:
        plan_bytes = evidence.read("frozen/docs/plans/2026-11-11-bidirectional-support-design.md")
    assert hashlib.sha256(plan_bytes).hexdigest() == old["files"][str(historical_plan)]
    (root / "historical-preregistration.md").write_bytes(plan_bytes)
    prior_bindings = {p: h for p, h in old["files"].items() if Path(p) != historical_plan}
    assert all(sha(p) == h for p, h in prior_bindings.items())
    files.update(prior_bindings)
    files.update(
        {
            str(p): sha(p)
            for p in (REPO / ".venv/support-runtime/embedding-cache").rglob("*")
            if p.is_file()
        }
    )
    paths = [
        Path(__file__),
        WORKER,
        Path(__file__).with_name("resident_support_path.py"),
        package_path,
        root / "cases.json",
        root / "frames.json",
        root / "historical-preregistration.md",
        historical_plan,
        archive,
        PROFILE / "profile.json",
        PROFILE / "result.json",
        REPO / "tests/test_resident_support_path.py",
        REPO / "docs/plans/2026-11-11-resident-support-boundary-amendment.md",
    ]
    paths += [p for p in (root / "seeds").rglob("*") if p.is_file()]
    files.update({str(p): sha(p) for p in paths})
    assert all(sha(p) == h for p, h in files.items())
    save(
        root / "freeze.json",
        dict(
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            files=files,
            native_versions=versions(),
            model_versions=old["versions"],
            blocks=2,
            warmups_per_case_per_block=3,
            measured_per_case_per_block=25,
            expected_measured=300,
            p95="inclusive linear interpolation",
            limit_ms=500,
            semantic_decisions=False,
            training=False,
        ),
    )
    print(json.dumps(dict(prepared=True, workloads=read(root / "frames.json"))), flush=True)


def run(root):
    from mnemos.client import MnemosClient, Outbox
    from mnemos.contracts import RetrievalRequest
    from mnemos.service import MnemosService
    from mnemos.utils.embeddings import FastembedProvider

    freeze = read(root / "freeze.json")
    with (root / "attempt.started").open("x") as stream:
        stream.write(str(time.time()))
    record = dict(
        status="failed",
        expected_measured=300,
        expected_warmups=36,
        semantic_decisions=False,
        training=False,
        freeze_sha256=sha(root / "freeze.json"),
        blocks=[],
    )
    rows, active = [], None
    try:
        assert versions() == freeze["native_versions"]
        assert all(sha(p) == h for p, h in freeze["files"].items())
        cases = read(root / "cases.json")
        frames = {f["name"]: f for f in read(root / "frames.json")}
        for block in range(2):
            directory = root / f"block-{block}"
            directory.mkdir()
            memories, process, service = {}, None, None
            summary = dict(block=block, worker_stopped=False, service_stopped=False)
            record["blocks"].append(summary)
            try:
                start = time.perf_counter()
                stderr = (directory / "worker-stderr.log").open("xb")
                process = subprocess.Popen(
                    [str(MODEL_PYTHON), "-I", str(WORKER), "serve", str(root), str(MODEL_ROOT)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    env=local_environment(),
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                token = secrets.token_urlsafe(32)
                process.stdin.write(json.dumps(dict(token=token)) + "\n")
                process.stdin.close()
                ready = json.loads(process.stdout.readline())
                assert ready["versions"] == freeze["model_versions"]
                summary.update(worker=ready, cold_worker_ms=(time.perf_counter() - start) * 1000)
                embedder = FastembedProvider()
                for case in cases:
                    target = directory / case["name"]
                    shutil.copytree(root / "seeds" / case["name"], target)
                    memories[case["name"]] = memory_at(target, embedder)
                by_name = {case["name"]: case for case in cases}
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

                class Dispatcher:
                    async def dispatch(self, operation, payload):
                        if operation != "retrieve":
                            raise ValueError("Capacity dispatcher permits retrieval only")
                        request = RetrievalRequest.model_validate(payload)
                        name = request.context.session_id
                        case = by_name[name]

                        async def infer(cohort):
                            data = json.dumps(dict(name=name, cohort=cohort)).encode()
                            rpc = urllib.request.Request(
                                f"http://127.0.0.1:{ready['port']}/infer",
                                data=data,
                                headers={
                                    "Authorization": "Bearer " + token,
                                    "Content-Type": "application/json",
                                },
                            )

                            def call():
                                with opener.open(rpc, timeout=30) as response:
                                    return json.load(response)

                            result = await asyncio.to_thread(call)
                            if result.get("frame_sha256") != frames[name]["frame_sha256"]:
                                raise ValueError("Unexpected frame binding")
                            return result

                        return await path_adapter.capacity_retrieve(
                            memories[name],
                            request,
                            infer,
                            [f["id"] for f in case["cohort"]["candidates"]],
                        )

                    async def close(self):
                        for memory in memories.values():
                            await memory.close()

                service = MnemosService(
                    dispatcher=Dispatcher(), descriptor_path=directory / "runtime.json"
                )
                service.start()
                summary["resident_setup_ms"] = (time.perf_counter() - start) * 1000

                async def measure():
                    nonlocal active
                    client = MnemosClient(
                        descriptor_path=directory / "runtime.json",
                        outbox=Outbox(directory / "outbox.json"),
                        auto_start=False,
                    )
                    try:
                        for warmup, count in ((True, 3), (False, 25)):
                            for index in range(count):
                                for case in cases:
                                    active = dict(
                                        block=block, warmup=warmup, index=index, name=case["name"]
                                    )
                                    started = time.perf_counter_ns()
                                    try:
                                        result = await client.retrieve(case["request"])
                                        finished = time.perf_counter_ns()
                                        assert not result.get("degraded") and "capacity" in result
                                        assert (
                                            result["capacity"]["tokens"]
                                            == frames[case["name"]]["tokens"]
                                        )
                                        row = dict(
                                            **active,
                                            status="completed",
                                            milliseconds=(finished - started) / 1e6,
                                            capacity=result["capacity"],
                                            returned_facts=len(result["facts"]),
                                            context_bytes=len(result["context"].encode()),
                                        )
                                    except BaseException as exc:
                                        row = dict(
                                            **active,
                                            status="failed",
                                            error_type=type(exc).__name__,
                                            elapsed_until_exception_ms=(
                                                time.perf_counter_ns() - started
                                            )
                                            / 1e6,
                                        )
                                        with (root / "samples.jsonl").open(
                                            "a", encoding="utf-8"
                                        ) as stream:
                                            stream.write(json.dumps(row) + "\n")
                                        raise
                                    with (root / "samples.jsonl").open(
                                        "a", encoding="utf-8"
                                    ) as stream:
                                        stream.write(json.dumps(row) + "\n")
                                    rows.append(row)
                                    active = None
                        summary["worker_calls"] = result["capacity"]["calls"]
                    finally:
                        await client.close()

                asyncio.run(measure())
                assert summary["worker_calls"] == dict(forward=140, requests=168)
            finally:
                errors = []
                try:
                    if service is not None:
                        service.stop()
                        summary["service_stopped"] = not (directory / "runtime.json").exists()
                except BaseException as exc:
                    errors.append("service:" + type(exc).__name__)
                try:
                    if process is not None:
                        process.terminate()
                        process.wait(timeout=15)
                        summary["worker_stopped"] = process.poll() is not None
                        process.stdout.close()
                        stderr.close()
                except BaseException as exc:
                    errors.append("worker:" + type(exc).__name__)
                summary["cleanup_errors"] = errors
                save(directory / "cleanup.json", summary)
                if errors:
                    raise RuntimeError("Resident cleanup failed")
            print(json.dumps(dict(block=block, completed=True)), flush=True)
        workloads = {}
        for name in NAMES:
            durations = sorted(
                r["milliseconds"] for r in rows if not r["warmup"] and r["name"] == name
            )
            assert len(durations) == 50
            workloads[name] = dict(
                samples=50,
                median_ms=statistics.median(durations),
                p95_ms=statistics.quantiles(durations, n=100, method="inclusive")[94],
            )
        assert len(rows) == 336
        record.update(
            status="passed" if all(w["p95_ms"] <= 500 for w in workloads.values()) else "failed",
            workloads=workloads,
        )
    except BaseException as exc:
        record.update(error_type=type(exc).__name__, error=str(exc), active=active)
    finally:
        unchanged, binding_errors = check_bindings(freeze["files"])
        try:
            native_versions_unchanged = versions() == freeze["native_versions"]
        except BaseException as exc:
            native_versions_unchanged = False
            record["inventory_error_type"] = type(exc).__name__
        record.update(
            completed_measured=sum(not r["warmup"] for r in rows),
            completed_warmups=sum(r["warmup"] for r in rows),
            bindings_unchanged=unchanged,
            binding_errors=binding_errors,
            native_versions_unchanged=native_versions_unchanged,
        )
        if (
            not record["bindings_unchanged"]
            or not record["native_versions_unchanged"]
            or not all(b["worker_stopped"] and b["service_stopped"] for b in record["blocks"])
        ):
            record["status"] = "failed"
        save(root / "result.json", record)
        print(json.dumps(record, indent=2), flush=True)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    os.environ.update(
        FASTEMBED_CACHE_PATH=str(REPO / ".venv/support-runtime/embedding-cache"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    destination = Path(sys.argv[2]).resolve()
    if sys.argv[1] == "prepare":
        prepare(destination)
    elif sys.argv[1] == "seal":
        seal(destination)
    elif sys.argv[1] == "run":
        sys.exit(run(destination))
    else:
        raise ValueError("Unknown operation")
