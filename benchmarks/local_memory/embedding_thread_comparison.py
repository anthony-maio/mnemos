"""One local embedding resource and complete ordinary-recall comparison."""

import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
CANDIDATE = Path("D:/Development/mnemos-delivery")
SOURCE = Path(
    "C:/Users/Ant/AppData/Local/Temp/mnemos-improvement-checks-20261106/installed-pilot-command-timing-v1/candidate-0"
)
PACKAGE = REPO / ".venv/recall-snapshot-package-v1/package.json"
CACHE = REPO / ".venv/support-runtime/embedding-cache"
PROTOCOL = REPO / "docs/plans/2026-11-11-embedding-thread-budget-v1.md"
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def head(repo):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def same_files(files):
    try:
        return bool(files) and all(sha(p) == h for p, h in files.items())
    except OSError:
        return False


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=CANDIDATE)
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)
    root.mkdir(exist_ok=False)
    package = read(PACKAGE)
    requests = [
        json.loads(line)["request"]
        for line in (SOURCE / "observations.jsonl").read_text().splitlines()
        if (lambda r: r["path"] == "recall" and not r["warmup"])(json.loads(line))
    ][:50]
    assert len(requests) == 50
    texts = [
        "",
        "Synthetic diagnostic: the notebook keeps release checklists locally.",
        "Juniper keeps audit logs for 30 days.",
        "Unit K must not run with its gate open.",
        "Assume permit Q is present. Is delivery allowed?",
        "Dana reported a temperature; no sensor reading was retained.",
        "Caf\u00e9 route \u6771",
        "synthetic long retained source " * 100,
    ]
    save(root / "inputs.json", dict(texts=texts, requests=requests))
    for arm in ("baseline", "candidate"):
        directory = root / arm
        directory.mkdir()
        shutil.copyfile(SOURCE / "seed.sqlite", directory / "memory.sqlite")
        shutil.copyfile(SOURCE / "scopes.json", directory / "scopes.json")
    files = {
        str(Path(package["installed_target"]) / p): h for p, h in package["package_files"].items()
    }
    files.update(
        {
            str(Path(package["installed_target"]) / p): h
            for p, h in read(PACKAGE.parent / "dependency-files.json").items()
        }
    )
    paths = [Path(__file__), PROTOCOL, PACKAGE, root / "inputs.json"]
    paths += [SOURCE / n for n in ("seed.sqlite", "scopes.json", "observations.jsonl")]
    paths += [p for p in CACHE.rglob("*") if p.is_file()]
    changed = subprocess.check_output(
        ["git", "diff", package["source_commit"], "HEAD", "--name-only", "--", "mnemos"],
        cwd=CANDIDATE,
        text=True,
    ).splitlines()
    assert set(changed) == {"mnemos/runtime.py", "mnemos/settings.py", "mnemos/utils/embeddings.py"}
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", "mnemos"], cwd=CANDIDATE, text=True
    ).splitlines()
    paths += [CANDIDATE / n for n in names]
    files.update({str(p): sha(p) for p in paths})
    assert all(sha(p) == h for p, h in files.items())
    save(
        root / "freeze.json",
        dict(
            baseline=package["source_commit"],
            candidate=head(CANDIDATE),
            runner=head(REPO),
            files=files,
            arms=["baseline", "candidate"],
            samples=50,
            warmups=3,
            max_absolute_error=1e-6,
            minimum_cosine=0.999999,
            maximum_candidate_idle_cpu_ms=50,
            minimum_idle_reduction=0.9,
            latency_gate_claim=False,
            semantic_gate_claim=False,
        ),
    )
    print(json.dumps(dict(status="prepared", bound_files=len(files))))


def idle(seconds):
    wall, cpu = time.perf_counter_ns(), time.process_time_ns()
    time.sleep(seconds)
    return dict(
        wall_ms=(time.perf_counter_ns() - wall) / 1e6, cpu_ms=(time.process_time_ns() - cpu) / 1e6
    )


def worker(root, arm):
    if arm == "candidate":
        sys.path.insert(0, str(CANDIDATE))
    from mnemos.client import MnemosClient, Outbox
    from mnemos.config import MnemosConfig, ModuleConfig
    from mnemos.engine import MnemosEngine
    from mnemos.memory_service import LocalMemoryService
    from mnemos.runtime import build_embedder_from_settings
    from mnemos.scopes import ScopeRegistry
    from mnemos.service import MnemosService, _protect_private_path
    from mnemos.settings import AppSettings
    from mnemos.utils.llm import UnavailableLLMProvider
    from mnemos.utils.storage import SQLiteStore
    import mnemos

    directory = root / arm
    _protect_private_path(directory)
    record = dict(status="failed", arm=arm, package_path=mnemos.__file__, rows=[], vectors=[])
    service, store = None, None
    stage = "embedding"
    try:
        inputs = read(root / "inputs.json")
        record["before_session"] = idle(0.3)
        embedder = build_embedder_from_settings(
            AppSettings.model_validate({"embedding": {"provider": "fastembed"}})
        )
        options = embedder._engine.model.model.get_session_options()
        record["threads"] = options.intra_op_num_threads
        assert record["threads"] == (0 if arm == "baseline" else 1)
        embedder.embed("Unrelated synthetic warmup: notebook housekeeping.")
        record["vectors"] = [
            *embedder.embed_batch(inputs["texts"]),
            *[embedder.embed(text) for text in inputs["texts"]],
        ]
        assert len(record["vectors"]) == 16 and all(
            len(v) == 384 and all(math.isfinite(x) for x in v) for v in record["vectors"]
        )
        record["idle_windows"] = [idle(0.1), idle(0.1), idle(0.3)]
        stage = "recall"
        registry = ScopeRegistry.load(directory / "scopes.json")
        store = SQLiteStore(str(directory / "memory.sqlite"))
        assert (
            sum(
                json.loads(r[0]).get("source") != "captured_passthrough"
                for r in store._conn.execute("SELECT metadata FROM memory_chunks")
            )
            == 10000
        )
        config = MnemosConfig(
            modules=ModuleConfig(
                surprisal=False, mutable_rag=False, affective=False, sleep=False, spreading=False
            )
        )
        engine = MnemosEngine(
            config=config,
            embedder=embedder,
            llm=UnavailableLLMProvider(),
            store=store,
            scope_registry=registry,
            require_scope_context=True,
        )
        memory = LocalMemoryService(
            engine, registry, registry_loader=lambda: ScopeRegistry.load(directory / "scopes.json")
        )
        service = MnemosService(dispatcher=memory, descriptor_path=directory / "runtime.json")
        service.start()

        async def measure():
            client = MnemosClient(
                descriptor_path=directory / "runtime.json",
                outbox=Outbox(directory / "outbox.json"),
                auto_start=False,
            )
            try:
                for index in range(3):
                    value = await client.retrieve(inputs["requests"][index])
                    assert value["facts"] and not value.get("degraded")
                for index, request in enumerate(inputs["requests"]):
                    started = time.perf_counter_ns()
                    value = await client.retrieve(request)
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    assert value["facts"] and not value.get("degraded")
                    row = dict(index=index, milliseconds=elapsed, result=value)
                    with (directory / "samples.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row) + "\n")
                    record["rows"].append(row)
            finally:
                await client.close()

        asyncio.run(measure())
        assert store._conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert not store._conn.execute("PRAGMA foreign_key_check").fetchall()
        times = [r["milliseconds"] for r in record["rows"]]
        record.update(
            status="completed",
            median_ms=statistics.median(times),
            p95_ms=statistics.quantiles(times, n=100, method="inclusive")[94],
        )
    except BaseException as exc:
        record.update(error_type=type(exc).__name__, error=str(exc), stage=stage)
    finally:
        try:
            if service is not None:
                service.stop()
            elif store is not None:
                store.close()
        except BaseException as exc:
            record.update(status="failed", cleanup_error_type=type(exc).__name__)
        record["descriptor_removed"] = not (directory / "runtime.json").exists()
        save(directory / "result.json", record)
    return 0 if record["status"] == "completed" and record["descriptor_removed"] else 1


def pair(root):
    freeze = read(root / "freeze.json")
    with (root / "attempt.started").open("x") as stream:
        stream.write(str(time.time()))
    result = dict(status="failed", quality_gate_claim=False, latency_gate_claim=False)
    try:
        assert all(sha(p) == h for p, h in freeze["files"].items())
        for arm in freeze["arms"]:
            with (root / arm / "worker.log").open("x") as log:
                completed = subprocess.run(
                    [sys.executable, "-I", str(Path(__file__).resolve()), "worker", str(root), arm],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=70,
                )
            assert completed.returncode == 0, arm
        baseline, candidate = [read(root / arm / "result.json") for arm in freeze["arms"]]
        assert all(
            len(r["rows"]) == 50 and r["status"] == "completed" for r in (baseline, candidate)
        )
        maximum, minimum = 0.0, 1.0
        for a, b in zip(baseline["vectors"], candidate["vectors"], strict=True):
            maximum = max(maximum, max(abs(x - y) for x, y in zip(a, b, strict=True)))
            cosine = sum(x * y for x, y in zip(a, b, strict=True)) / math.sqrt(
                sum(x * x for x in a) * sum(y * y for y in b)
            )
            minimum = min(minimum, cosine)

        def supported_data(value):
            return {
                **value,
                "facts": [{k: v for k, v in f.items() if k != "scores"} for f in value["facts"]],
            }

        matched = sum(
            supported_data(a["result"]) == supported_data(b["result"])
            for a, b in zip(baseline["rows"], candidate["rows"], strict=True)
        )
        baseline_cpu = sum(w["cpu_ms"] for w in baseline["idle_windows"])
        candidate_cpu = sum(w["cpu_ms"] for w in candidate["idle_windows"])
        checks = dict(
            numerical_parity=maximum <= freeze["max_absolute_error"]
            and minimum >= freeze["minimum_cosine"],
            idle_cpu=baseline_cpu > 0
            and candidate_cpu <= freeze["maximum_candidate_idle_cpu_ms"]
            and candidate_cpu <= baseline_cpu * (1 - freeze["minimum_idle_reduction"]),
            recall_pairs=matched == 50,
            recall_p95=candidate["p95_ms"] <= baseline["p95_ms"],
        )
        result.update(
            status="passed" if all(checks.values()) else "failed",
            checks=checks,
            max_absolute_error=maximum,
            min_cosine=minimum,
            bit_identical=baseline["vectors"] == candidate["vectors"],
            matched_recall_pairs=matched,
            baseline_idle_cpu_ms=baseline_cpu,
            candidate_idle_cpu_ms=candidate_cpu,
            recall={
                arm: {k: r[k] for k in ("median_ms", "p95_ms")}
                for arm, r in zip(freeze["arms"], (baseline, candidate))
            },
        )
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        result["bindings_unchanged"] = same_files(freeze["files"])
        result["expected_samples_per_arm"] = 50
        result["completed_samples"] = {
            arm: (
                len(read(root / arm / "result.json")["rows"])
                if (root / arm / "result.json").exists()
                else 0
            )
            for arm in freeze["arms"]
        }
        if not result["bindings_unchanged"]:
            result["status"] = "failed"
        save(root / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    os.environ.update(FASTEMBED_CACHE_PATH=str(CACHE), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    mode, directory = sys.argv[1], Path(sys.argv[2]).resolve()
    if mode == "prepare":
        prepare(directory)
    elif mode == "worker":
        sys.exit(worker(directory, sys.argv[3]))
    elif mode == "pair":
        sys.exit(pair(directory))
    else:
        raise ValueError("Unknown phase")
