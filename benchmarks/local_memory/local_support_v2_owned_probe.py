"""Run the frozen backward probe in the existing Windows-owned process boundary."""

import argparse
import json
from pathlib import Path
import runpy
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def run(root):
    root = root.resolve()
    if not root.is_relative_to((ROOT / ".venv").resolve()):
        raise ValueError("Probe artifacts must stay in this checkout private environment")
    worker = Path(__file__).with_name("local_support_v2_backward_probe.py")
    supervisor = ROOT / "benchmarks/native_hosts/windows_owned_job.py"
    loader = runpy.run_path(str(worker))
    learning = loader["learning"]
    freeze = loader["read"](root / "freeze.json")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=15
    ).strip()
    if head != freeze["source_commit"] or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, timeout=15
    ):
        raise ValueError("Expected the frozen clean source checkout")
    if (root / "execution-control.json").exists() or (root / "attempt.started").exists():
        raise ValueError("Refusing to overwrite an earlier execution")
    loader["verify"](freeze["files"])
    control_files = {
        str(p): learning.sha(p) for p in (Path(__file__), supervisor, root / "freeze.json")
    }
    record = dict(status="running", source_commit=head, timeout_seconds=180, bound=control_files)
    save = lambda: loader["save"](root / "execution-control.json", record)
    started = time.perf_counter()
    save()
    code = "import importlib.metadata as m,json;print(json.dumps(sorted([dict(name=d.metadata['Name'],version=d.version) for d in m.distributions()],key=lambda r:r['name'].lower().replace('_','-'))))"
    inventory = lambda: json.loads(
        subprocess.check_output(
            [str(loader["MODEL_PYTHON"]), "-I", "-c", code], text=True, timeout=15
        )
    )
    try:
        if inventory() != freeze["versions"]:
            raise ValueError("Frozen model dependency inventory changed")
        record["gpu_before"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=15,
        ).strip()
        save()
        command = [str(loader["MODEL_PYTHON"]), "-I", str(worker), "run", str(root)]
        record["command"] = command
        record["job"] = runpy.run_path(str(supervisor))["run_owned_job"](
            command, cwd=ROOT, env=loader["environment"](root), output=root / "job", timeout=180
        )
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        if not (root / "result.json").exists():
            try:
                progress = (
                    loader["read"](root / "progress.json")
                    if (root / "progress.json").exists()
                    else dict(expected_microbatches=4, microbatches=[])
                )
                loader["record_failure"](root, progress, error)
            except Exception as receipt_error:
                record["failure_receipt_error_type"] = type(receipt_error).__name__
    finally:
        checks = {
            "bindings_unchanged": lambda: loader["verify"]({**freeze["files"], **control_files})
            is None,
            "head_unchanged": lambda: head
            == subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=15
            ).strip(),
            "dependency_versions_unchanged": lambda: inventory() == freeze["versions"],
            "owned_processes_stopped": lambda: loader["read"](root / "job/process.json")[
                "active_after_cleanup"
            ]
            == 0,
            "owned_job_bound_satisfied": lambda: loader["read"](root / "job/process.json")[
                "elapsed_seconds"
            ]
            <= 190,
        }
        for name, check in checks.items():
            try:
                record[name] = bool(check())
            except BaseException as error:
                record[name] = False
                record[name + "_error_type"] = type(error).__name__
        if not all(record[name] for name in checks):
            record["status"] = "failed"
        loader["finalize_child_record"](
            record, root / "result.json", all(record[name] for name in checks)
        )
        record["elapsed_seconds"] = time.perf_counter() - started
        record["attempt_marker_present"] = (root / "attempt.started").exists()
        save()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    record = run(args.directory)
    print(json.dumps(record, indent=2), flush=True)
    raise SystemExit(
        0
        if record["status"] == "passed"
        else 2 if record["status"] == "not_started_resource_unavailable" else 1
    )
