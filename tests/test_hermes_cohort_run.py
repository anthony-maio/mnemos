"""The fixed native cohort retains failures and stops on uncertain cleanup."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.native_hosts import (
    hermes_cohort_run_v1,
    hermes_cohort_run_v2,
    hermes_cohort_run_v3,
    hermes_cohort_run_v4,
    hermes_cohort_run_v5,
    hermes_cohort_run_v6,
    hermes_cohort_run_v7,
)

cohort = hermes_cohort_run_v1


@pytest.fixture(
    autouse=True,
    params=[
        hermes_cohort_run_v1,
        hermes_cohort_run_v2,
        hermes_cohort_run_v3,
        hermes_cohort_run_v4,
        hermes_cohort_run_v5,
        hermes_cohort_run_v6,
        hermes_cohort_run_v7,
    ],
)
def cohort_version(request, monkeypatch):
    monkeypatch.setitem(globals(), "cohort", request.param)


def configure(monkeypatch, tmp_path, *, mode="three"):
    output = tmp_path / "cohort"
    monkeypatch.setattr(
        cohort.sys,
        "argv",
        [
            "cohort",
            "--checks",
            "checks.json",
            "--installed-checks",
            "installed.json",
            "--output",
            str(output),
            "--prior-queue-state",
            "queue.json",
            "--exclude-pid",
            "12",
        ],
    )
    monkeypatch.setattr(cohort.subprocess, "check_output", lambda *args, **kwargs: "frozen\n")
    monkeypatch.setattr(cohort, "verify_committed_inputs", lambda *args: None)
    monkeypatch.setattr(cohort, "sha", lambda path: "hash-" + Path(path).name)
    calls = []
    terminated = []

    class FakeProcess:
        def __init__(self, directory, number):
            self.directory = directory
            self.number = number
            self.pid = 1000 + number
            self.returncode = None

        def wait(self, timeout):
            calls.append(self.directory.name)
            self.directory.mkdir()
            if mode == "timeout":
                raise cohort.subprocess.TimeoutExpired(["fake"], timeout)
            if mode == "missing" and self.number == 1:
                self.returncode = 1
                return self.returncode
            failed = self.number == 1
            self.returncode = 1 if failed else 0
            report = {
                "status": "failed" if failed else "passed",
                "finished_at": "terminal",
                "phases": [],
            }
            if mode == "uncertain" and self.number == 1:
                (self.directory / "runtime.json").write_text("still-owned")
            (self.directory / "report.json").write_text(json.dumps(report))
            return self.returncode

    def popen(command, **kwargs):
        directory = Path(command[command.index("--output") + 1])
        number = int(directory.name.rsplit("-", 1)[1])
        return FakeProcess(directory, number)

    monkeypatch.setattr(cohort.subprocess, "Popen", popen)
    monkeypatch.setattr(
        cohort,
        "terminate_owned_process",
        lambda process: terminated.append(process.pid),
    )
    return output, calls, terminated


def test_fixed_cohort_retains_failed_attempt_and_runs_all_three(monkeypatch, tmp_path):
    output, calls, terminated = configure(monkeypatch, tmp_path)
    assert cohort.main() == 1
    report = json.loads((output / "cohort.json").read_text())
    assert calls == ["attempt-1", "attempt-2", "attempt-3"]
    assert terminated == []
    assert report["status"] == "completed_with_failures"
    assert report["passed_attempts"] == 2
    assert [row["status"] for row in report["attempts"]] == [
        "failed",
        "passed",
        "passed",
    ]
    assert report["attempts"][0]["cleanup_established"] is True
    assert report["attempt_timeout_seconds"] == 1800
    assert "benchmarks/native_hosts/native_processes.py" in report["input_hashes"]


def test_uncertain_cleanup_prevents_next_attempt(monkeypatch, tmp_path):
    output, calls, terminated = configure(monkeypatch, tmp_path, mode="uncertain")
    assert cohort.main() == 1
    assert calls == ["attempt-1"]
    report = json.loads((output / "cohort.json").read_text())
    assert report["status"] == "stopped_uncertain_cleanup"
    assert report["stop_reason"] == "cleanup"
    assert report["attempts"][0]["status"] == "uncertain_cleanup"
    assert report["attempts"][0]["cleanup_established"] is False


def test_missing_report_is_retained_and_stops_cohort(monkeypatch, tmp_path):
    output, calls, _ = configure(monkeypatch, tmp_path, mode="missing")
    assert cohort.main() == 1
    assert calls == ["attempt-1"]
    report = json.loads((output / "cohort.json").read_text())
    assert report["status"] == "stopped_uncertain_cleanup"
    assert report["stop_reason"] == "missing_report"
    assert report["attempts"][0]["status"] == "missing_report"
    assert report["attempts"][0]["exit_code"] == 1


def test_timeout_terminates_owned_process_and_stops_cohort(monkeypatch, tmp_path):
    output, calls, terminated = configure(monkeypatch, tmp_path, mode="timeout")
    assert cohort.main() == 1
    assert calls == ["attempt-1"]
    assert terminated == [1001]
    report = json.loads((output / "cohort.json").read_text())
    assert report["status"] == "stopped_uncertain_cleanup"
    assert report["stop_reason"] == "attempt_timeout"
    assert report["attempts"][0]["status"] == "timeout"
    assert report["attempts"][0]["uncertain_cleanup"] is True


def test_input_hash_failure_prevents_output_mutation(monkeypatch, tmp_path):
    output = tmp_path / "cohort"
    monkeypatch.setattr(
        cohort.sys,
        "argv",
        [
            "cohort",
            "--checks",
            "c",
            "--installed-checks",
            "i",
            "--output",
            str(output),
            "--prior-queue-state",
            "q",
            "--exclude-pid",
            "12",
        ],
    )
    monkeypatch.setattr(cohort.subprocess, "check_output", lambda *args, **kwargs: "frozen\n")
    monkeypatch.setattr(
        cohort,
        "verify_committed_inputs",
        lambda *args: (_ for _ in ()).throw(ValueError("changed")),
    )
    with pytest.raises(ValueError, match="changed"):
        cohort.main()
    assert not output.exists()
