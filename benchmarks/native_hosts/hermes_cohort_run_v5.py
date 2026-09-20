"""Run exactly three independent Hermes v7 lifecycles and preserve every outcome."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "native_hosts"

from .hermes_lifecycle_run_v7 import sha, verify_committed_inputs
from .native_processes import terminate_owned_process

ATTEMPT_TIMEOUT_SECONDS = 1800
CHILD_INPUTS = (
    "benchmarks/native_hosts/ollama_readiness_v1.py",
    "docs/plans/2026-09-20-hermes-native-lifecycle-protocol-v7.md",
    "docs/plans/2026-09-19-hermes-native-lifecycle-protocol-v3.md",
    "docs/plans/2026-09-19-hermes-native-lifecycle-protocol-v4.md",
    "benchmarks/native_hosts/native_processes.py",
    "benchmarks/native_hosts/hermes_lifecycle_run_v7.py",
    "benchmarks/native_hosts/hermes_native_bootstrap_v4.py",
    "benchmarks/native_hosts/hermes_trace.py",
    "benchmarks/native_hosts/hermes_trace_v2.py",
    "docs/plans/2026-09-19-hermes-native-lifecycle-protocol-v5.md",
    "docs/plans/2026-09-19-hermes-native-lifecycle-protocol-v6.md",
    "docs/validation/2026-09-19/native-model-profile-v1/profile.json",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _input_hashes(repo: Path, commit: str) -> dict[str, str]:
    paths = ("benchmarks/native_hosts/hermes_cohort_run_v5.py", *CHILD_INPUTS)
    verify_committed_inputs(repo, list(paths), commit)
    return {relative: sha(repo / relative) for relative in paths}


def _command(args: argparse.Namespace, directory: Path) -> list[str]:
    command = [
        sys.executable,
        "-I",
        "-u",
        str(Path(__file__).with_name("hermes_lifecycle_run_v7.py")),
        "--checks",
        str(args.checks.resolve()),
        "--installed-checks",
        str(args.installed_checks.resolve()),
        "--output",
        str(directory),
        "--prior-queue-state",
        str(args.prior_queue_state.resolve()),
    ]
    for pid in args.exclude_pid:
        command.extend(["--exclude-pid", str(pid)])
    return command


def _record_report(
    entry: dict[str, Any], report_path: Path, report: dict[str, Any], exit_code: int
) -> bool:
    entry.update(
        status=report.get("status"),
        exit_code=exit_code,
        report_sha256=sha(report_path),
        phases=report.get("phases"),
        source_commit=report.get("source_commit"),
        wheel_sha256=report.get("wheel_sha256"),
        error=report.get("error"),
        cleanup_error=report.get("cleanup_error"),
    )
    valid_status = report.get("status") in {"passed", "failed"}
    clean = (
        valid_status
        and bool(report.get("finished_at"))
        and not report.get("cleanup_error")
        and not (report_path.parent / "runtime.json").exists()
        and exit_code == (0 if report["status"] == "passed" else 1)
    )
    entry["cleanup_established"] = clean
    if not valid_status:
        entry["status"] = "invalid_report"
        entry["error"] = "Report status was not passed or failed"
    elif not clean:
        entry["status"] = "uncertain_cleanup"
        entry["error"] = entry.get("error") or "Attempt did not terminate with established cleanup"
    return clean


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--installed-checks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-queue-state", type=Path, required=True)
    parser.add_argument("--exclude-pid", type=int, action="append", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.is_relative_to(repo):
        raise ValueError("Cohort state must be outside the repository")
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    input_hashes = _input_hashes(repo, commit)
    output.mkdir(exist_ok=False)
    record: dict[str, Any] = {
        "status": "running",
        "started_at": _utcnow(),
        "planned_attempts": 3,
        "attempt_timeout_seconds": ATTEMPT_TIMEOUT_SECONDS,
        "runner_commit": commit,
        "input_hashes": input_hashes,
        "runner_sha256": sha(Path(__file__)),
        "attempts": [],
    }
    cohort_path = output / "cohort.json"
    _save(cohort_path, record)
    try:
        for number in (1, 2, 3):
            directory = output / f"attempt-{number}"
            command = _command(args, directory)
            entry: dict[str, Any] = {
                "attempt": number,
                "status": "running",
                "started_at": _utcnow(),
                "report": str(directory / "report.json"),
                "command": command,
            }
            record["attempts"].append(entry)
            _save(cohort_path, record)
            print(json.dumps({"attempt": number, "status": "starting"}), flush=True)
            try:
                directory.parent.mkdir(parents=True, exist_ok=True)
                with (
                    (output / f"attempt-{number}.stdout.log").open("wb") as stdout,
                    (output / f"attempt-{number}.stderr.log").open("wb") as stderr,
                ):
                    process = subprocess.Popen(
                        command,
                        cwd=output,
                        stdout=stdout,
                        stderr=stderr,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    try:
                        exit_code = process.wait(timeout=ATTEMPT_TIMEOUT_SECONDS)
                    except subprocess.TimeoutExpired:
                        entry.update(
                            status="timeout",
                            exit_code=None,
                            error=f"Attempt exceeded {ATTEMPT_TIMEOUT_SECONDS} seconds",
                            timeout=True,
                            cleanup_established=False,
                            uncertain_cleanup=True,
                        )
                        try:
                            terminate_owned_process(process)
                        except BaseException as exc:
                            entry["termination_error"] = f"{type(exc).__name__}: {exc}"
                        record.update(
                            status="stopped_uncertain_cleanup",
                            stop_reason="attempt_timeout",
                        )
                        _save(cohort_path, record)
                        return 1
            except BaseException as exc:
                entry.update(
                    status="launch_error",
                    exit_code=None,
                    error=f"{type(exc).__name__}: {exc}",
                    cleanup_established=False,
                    uncertain_cleanup=True,
                )
                record.update(status="stopped_uncertain_cleanup", stop_reason="launch_error")
                _save(cohort_path, record)
                return 1

            report_path = directory / "report.json"
            if not report_path.is_file():
                entry.update(
                    status="missing_report",
                    exit_code=exit_code,
                    error="Attempt exited without report.json",
                    cleanup_established=False,
                    uncertain_cleanup=True,
                )
                record.update(status="stopped_uncertain_cleanup", stop_reason="missing_report")
                _save(cohort_path, record)
                return 1
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if not isinstance(report, dict):
                    raise ValueError("report.json is not an object")
            except BaseException as exc:
                entry.update(
                    status="invalid_report",
                    exit_code=exit_code,
                    error=f"{type(exc).__name__}: {exc}",
                    cleanup_established=False,
                    uncertain_cleanup=True,
                )
                record.update(status="stopped_uncertain_cleanup", stop_reason="invalid_report")
                _save(cohort_path, record)
                return 1
            clean = _record_report(entry, report_path, report, exit_code)
            entry["finished_at"] = _utcnow()
            _save(cohort_path, record)
            print(json.dumps({"attempt": number, "status": entry["status"]}), flush=True)
            if not clean:
                record.update(status="stopped_uncertain_cleanup", stop_reason="cleanup")
                _save(cohort_path, record)
                return 1
        record["passed_attempts"] = sum(row["status"] == "passed" for row in record["attempts"])
        record["status"] = "passed" if record["passed_attempts"] == 3 else "completed_with_failures"
        return 0 if record["status"] == "passed" else 1
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record["finished_at"] = _utcnow()
        _save(cohort_path, record)


if __name__ == "__main__":
    raise SystemExit(main())
