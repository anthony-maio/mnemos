"""Read-only validation of a retained three-attempt Hermes cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON report: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON report is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"missing cohort report: {path}") from exc


def validate_hermes_cohort(
    acceptance_path: Path, *, source_commit: str, wheel_sha256: str
) -> dict[str, Any]:
    """Validate every retained attempt and return bound cohort evidence.

    The selected acceptance report is one child of the cohort and must be the
    sole selected report whose child status is ``passed``.  Failed attempts are
    retained and validated rather than discarded.
    """

    selected = acceptance_path.resolve()
    cohort_path = selected.parent.parent / "cohort.json"
    cohort = _read_object(cohort_path)
    attempts_value = cohort.get("attempts")
    cohort_status = cohort.get("status")
    if (
        not isinstance(cohort_status, str)
        or cohort_status not in {"passed", "completed_with_failures"}
        or not isinstance(cohort.get("finished_at"), str)
        or not cohort.get("finished_at")
        or cohort.get("planned_attempts") != 3
        or not isinstance(attempts_value, list)
        or len(attempts_value) != 3
    ):
        raise ValueError("Hermes cohort is not a terminal three-attempt cohort")

    attempts: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    report_hashes: dict[str, str] = {}
    passed_paths: list[Path] = []
    for row_value in attempts_value:
        if not isinstance(row_value, dict):
            raise ValueError("Hermes cohort attempt is not an object")
        attempt_id = row_value.get("attempt")
        if type(attempt_id) is not int or attempt_id not in {1, 2, 3} or attempt_id in seen_ids:
            raise ValueError("Hermes cohort attempt IDs must be unique 1, 2, and 3")
        seen_ids.add(attempt_id)
        if row_value.get("cleanup_established") is not True:
            raise ValueError("Hermes cohort attempt lacks cleanup evidence")
        expected_path = cohort_path.parent / f"attempt-{attempt_id}" / "report.json"
        row_path_value = row_value.get("report")
        if (
            not isinstance(row_path_value, str)
            or Path(row_path_value).resolve() != expected_path.resolve()
        ):
            raise ValueError("Hermes cohort attempt report path is not bound to its attempt")
        if not expected_path.is_file():
            raise ValueError("Hermes cohort attempt report is missing")
        if (expected_path.parent / "runtime.json").exists():
            raise ValueError("Hermes cohort attempt retains a live runtime descriptor")
        digest = _sha256(expected_path)
        if row_value.get("report_sha256") != digest:
            raise ValueError("Hermes cohort attempt report hash differs")
        report = _read_object(expected_path)
        status = report.get("status")
        if (
            not isinstance(status, str)
            or status not in {"passed", "failed"}
            or row_value.get("status") != status
        ):
            raise ValueError("Hermes cohort attempt status differs from its report")
        if (
            report.get("source_commit") != source_commit
            or report.get("wheel_sha256") != wheel_sha256
        ):
            raise ValueError("Hermes cohort attempt package binding differs")
        if not isinstance(report.get("finished_at"), str) or not report.get("finished_at"):
            raise ValueError("Hermes cohort attempt is unfinished")
        if report.get("cleanup_error"):
            raise ValueError("Hermes cohort attempt has a cleanup error")
        report_hashes[str(attempt_id)] = digest
        if status == "passed":
            passed_paths.append(expected_path.resolve())
        attempts.append(row_value)

    if seen_ids != {1, 2, 3} or not passed_paths:
        raise ValueError("Hermes cohort must contain at least one passed attempt")
    expected_cohort_status = "passed" if len(passed_paths) == 3 else "completed_with_failures"
    if cohort_status != expected_cohort_status:
        raise ValueError("Hermes cohort status does not match its child reports")
    if selected not in passed_paths:
        raise ValueError("Hermes acceptance report is not a passed cohort child")

    return {
        "acceptance_path": str(selected),
        "cohort_path": str(cohort_path.resolve()),
        "report_hashes": report_hashes,
        "cohort": cohort,
    }
