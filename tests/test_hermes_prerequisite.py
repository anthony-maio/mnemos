"""Integrity checks for the retained three-attempt Hermes prerequisite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.native_hosts.hermes_prerequisite import validate_hermes_cohort

SOURCE = "source-commit"
WHEEL = "wheel-sha"


def _write_cohort(
    root: Path, statuses: tuple[str, str, str] = ("passed", "failed", "failed")
) -> Path:
    attempts: list[dict[str, object]] = []
    for number, status in enumerate(statuses, 1):
        directory = root / f"attempt-{number}"
        directory.mkdir(parents=True, exist_ok=True)
        report = {
            "status": status,
            "source_commit": SOURCE,
            "wheel_sha256": WHEEL,
            "finished_at": f"2026-09-20T00:0{number}:00Z",
            "cleanup_error": None,
        }
        report_path = directory / "report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        attempts.append(
            {
                "attempt": number,
                "status": status,
                "report": str(report_path),
                "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "cleanup_established": True,
            }
        )
    cohort = {
        "status": (
            "passed"
            if all(status == "passed" for status in statuses)
            else "completed_with_failures"
        ),
        "planned_attempts": 3,
        "finished_at": "2026-09-20T00:10:00Z",
        "attempts": attempts,
    }
    (root / "cohort.json").write_text(json.dumps(cohort), encoding="utf-8")
    return root / "attempt-1" / "report.json"


def test_validates_one_pass_and_two_failures_and_returns_all_hashes(tmp_path: Path) -> None:
    acceptance = _write_cohort(tmp_path)

    result = validate_hermes_cohort(acceptance, source_commit=SOURCE, wheel_sha256=WHEEL)

    assert result["acceptance_path"] == str(acceptance.resolve())
    assert set(result["report_hashes"]) == {"1", "2", "3"}


def test_accepts_three_passes(tmp_path: Path) -> None:
    acceptance = _write_cohort(tmp_path, ("passed", "passed", "passed"))

    validate_hermes_cohort(acceptance, source_commit=SOURCE, wheel_sha256=WHEEL)


def test_accepts_two_passes_and_one_failure(tmp_path: Path) -> None:
    acceptance = _write_cohort(tmp_path, ("passed", "passed", "failed"))

    validate_hermes_cohort(acceptance, source_commit=SOURCE, wheel_sha256=WHEEL)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "swapped",
        "edited",
        "cleanup",
        "runtime",
        "status",
        "source",
        "wheel",
    ],
)
def test_rejects_integrity_mutations(tmp_path: Path, mutation: str) -> None:
    acceptance = _write_cohort(tmp_path)
    cohort_path = tmp_path / "cohort.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    attempts = cohort["attempts"]
    if mutation == "duplicate":
        attempts[1]["attempt"] = 1
    elif mutation == "missing":
        (tmp_path / "attempt-2" / "report.json").unlink()
    elif mutation == "swapped":
        attempts[0]["report"] = str(tmp_path / "attempt-2" / "report.json")
    elif mutation == "edited":
        (tmp_path / "attempt-2" / "report.json").write_text("{}", encoding="utf-8")
    elif mutation == "cleanup":
        attempts[1]["cleanup_established"] = False
    elif mutation == "runtime":
        (tmp_path / "attempt-2" / "runtime.json").write_text("live", encoding="utf-8")
    elif mutation == "status":
        attempts[1]["status"] = "passed"
    elif mutation == "source":
        json_path = tmp_path / "attempt-2" / "report.json"
        report = json.loads(json_path.read_text(encoding="utf-8"))
        report["source_commit"] = "other"
        json_path.write_text(json.dumps(report), encoding="utf-8")
        attempts[1]["report_sha256"] = hashlib.sha256(json_path.read_bytes()).hexdigest()
    elif mutation == "wheel":
        json_path = tmp_path / "attempt-2" / "report.json"
        report = json.loads(json_path.read_text(encoding="utf-8"))
        report["wheel_sha256"] = "other"
        json_path.write_text(json.dumps(report), encoding="utf-8")
        attempts[1]["report_sha256"] = hashlib.sha256(json_path.read_bytes()).hexdigest()
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_hermes_cohort(acceptance, source_commit=SOURCE, wheel_sha256=WHEEL)


def test_rejects_non_string_child_status(tmp_path: Path) -> None:
    acceptance = _write_cohort(tmp_path)
    report_path = tmp_path / "attempt-2" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = []
    report_path.write_text(json.dumps(report), encoding="utf-8")
    cohort_path = tmp_path / "cohort.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort["attempts"][1]["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_hermes_cohort(acceptance, source_commit=SOURCE, wheel_sha256=WHEEL)


def test_rejects_non_string_cohort_status(tmp_path: Path) -> None:
    acceptance = _write_cohort(tmp_path)
    cohort_path = tmp_path / "cohort.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort["status"] = []
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_hermes_cohort(acceptance, source_commit=SOURCE, wheel_sha256=WHEEL)


def test_rejects_invalid_acceptance_path_and_selected_failed_child(tmp_path: Path) -> None:
    acceptance = _write_cohort(tmp_path)

    with pytest.raises(ValueError):
        validate_hermes_cohort(tmp_path / "missing.json", source_commit=SOURCE, wheel_sha256=WHEEL)
    with pytest.raises(ValueError):
        validate_hermes_cohort(
            tmp_path / "attempt-2" / "report.json", source_commit=SOURCE, wheel_sha256=WHEEL
        )
