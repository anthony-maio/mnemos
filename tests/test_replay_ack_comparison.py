"""Reject incomplete delivery, corrupted durability and misleading timing subsets."""

import copy
import json
from pathlib import Path
import runpy
import sqlite3
import subprocess

import pytest

HELPERS = Path(__file__).resolve().parents[1] / "benchmarks/native_hosts"
RUNNER = runpy.run_path(str(HELPERS / "replay_ack_comparison.py"))
AUDIT = runpy.run_path(str(HELPERS / "replay_ack_audit.py"))


def records(tmp_path):
    schedule = RUNNER["schedule"](tmp_path / "project")
    ledger = [
        dict(
            key=row["key"],
            status="executed",
            returncode=0,
            stdout='{"continue":true}',
            stderr=AUDIT["EXPECTED_STDERR"],
            start_ns=(i + 1) * 1000000,
            end_ns=(i + 1) * 1000000 + 500000,
        )
        for i, row in enumerate(schedule)
    ]
    return schedule, ledger


@pytest.mark.parametrize("failure", ["cleanup", "binding", "postcheck", "none"])
def test_completed_schedule_does_not_hide_cleanup_or_binding_failure(failure):
    clean = dict(
        exit_code=0,
        owner_absent=True,
        ports_closed=dict(memory_port=True, worker_port=True),
        stopped=dict(stopped=True, cleanup_errors=[]),
    )
    execution = dict(
        schedule_completed=True,
        cleanup={"baseline": copy.deepcopy(clean), "candidate": copy.deepcopy(clean)},
        files_unchanged={"file": True},
        git_resolution_unchanged=True,
        freeze_unchanged=True,
        dependencies_unchanged=True,
    )
    if failure == "cleanup":
        execution["cleanup"]["candidate"]["error_type"] = "TimeoutError"
    elif failure == "binding":
        execution["files_unchanged"]["file"] = False
    elif failure == "postcheck":
        execution["postcheck_error"] = "OSError"
    RUNNER["finalize_status"](execution)
    assert execution["status"] == ("completed" if failure == "none" else "failed")


def test_invalid_initial_ledger_records_terminal_failure(tmp_path):
    (tmp_path / "freeze.json").write_text(json.dumps({"schedule": [], "files": {}}))
    (tmp_path / "ledger.json").write_text(json.dumps([{"key": "unexpected"}]))
    with pytest.raises(AssertionError):
        RUNNER["run"](tmp_path)
    assert json.loads((tmp_path / "execution.json").read_text())["status"] == "failed"
    assert json.loads((tmp_path / "ledger.json").read_text()) == [{"key": "unexpected"}]


def test_native_timeout_preserves_partial_output_and_does_not_retry(monkeypatch, tmp_path):
    calls = []

    def timeout(*args, **kwargs):
        calls.append(True)
        raise subprocess.TimeoutExpired(args[0], 30, output=b"partial", stderr=b"error")

    monkeypatch.setattr(RUNNER["subprocess"], "run", timeout)
    row = RUNNER["native_command"](["unused"], {}, {}, tmp_path)
    assert calls == [True] and row["status"] == "timeout"
    assert row["stdout"] == "partial" and row["stderr"] == "error"
    assert row["end_ns"] >= row["start_ns"]


def test_full_schedule_and_independent_reconstruction(tmp_path):
    schedule, ledger = records(tmp_path)
    freeze = dict(
        schedule=schedule,
        warmup_rounds=5,
        measured_rounds=50,
        context=dict(
            user_profile_id="ack-comparison",
            session_id="ack-comparison-session",
            host="codex",
            workspace_id=None,
            project_id="test-project",
        ),
    )
    assert AUDIT["validate_schedule"](freeze, tmp_path) == schedule
    assert len(schedule) == 220
    assert sum(not row["warmup"] for row in schedule) == 200
    for arm in ("baseline", "candidate"):
        assert len({row["payload"]["turn_id"] for row in schedule if row["arm"] == arm}) == 110
    AUDIT["validate_ledger"](schedule, ledger)


@pytest.mark.parametrize("change", ["missing", "duplicate", "timeout", "unknown_delivery"])
def test_ledger_cannot_drop_or_hide_bad_outcomes(tmp_path, change):
    schedule, ledger = records(tmp_path)
    if change == "missing":
        ledger.pop(33)
    elif change == "duplicate":
        ledger[33] = copy.deepcopy(ledger[32])
    elif change == "timeout":
        ledger[33]["status"] = "timeout"
    else:
        # The native unknown-outcome path returns 0 and continue:true too.
        ledger[33]["stderr"] = "mnemos hook diagnostic host=codex event=Stop status=error"
    with pytest.raises(AssertionError):
        AUDIT["validate_ledger"](schedule, ledger)


def test_failed_latency_stays_in_descriptive_distribution(tmp_path):
    schedule, ledger = records(tmp_path)
    row = next(row for row in ledger if row["key"] == "5-baseline-primer")
    row.update(status="timeout", end_ns=row["start_ns"] + 30_000_000_000)
    report = AUDIT["summarize"](schedule, ledger, {})
    group = report["arms"]["baseline"]["all"]
    assert group["command_ms"]["n"] == 100 and group["command_ms"]["max"] == 30000
    assert group["delivered"] == 99
    assert report["status_counts"]["timeout"] == 1


def test_nearest_rank_and_half_open_overlap(tmp_path):
    assert AUDIT["distribution"](list(range(1, 51)))["p95"] == 48
    schedule, ledger = records(tmp_path)
    sample = next(row for row in ledger if row["key"] == "5-baseline-primer")
    trace = {
        "baseline": {
            "ack_intervals": [dict(start_ns=sample["end_ns"], end_ns=sample["end_ns"] + 1)]
        }
    }
    assert (
        AUDIT["summarize"](schedule, ledger, trace)["arms"]["baseline"]["all"][
            "commands_overlapping_ack"
        ]
        == 0
    )


def test_malformed_trace_produces_failed_audit_without_invented_summary(tmp_path):
    schedule, ledger = records(tmp_path)
    (tmp_path / "freeze.json").write_text(json.dumps({"schedule": schedule}))
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))
    (tmp_path / "baseline").mkdir()
    (tmp_path / "baseline/final-trace.json").write_text(
        json.dumps({"ack_intervals": [{"start_ns": None}]})
    )
    report = AUDIT["audit"](tmp_path)
    assert report["status"] == "failed" and report["errors"]
    assert report["summary"] is None
    assert not any(report["synthetic_complete_command_p95_under_100ms"].values())


def test_own_ack_is_not_evidence_of_prior_event_contention(tmp_path):
    schedule, ledger = records(tmp_path)
    context = dict(
        user_profile_id="test", project_id="project", workspace_id=None, session_id="session"
    )
    planned = next(row for row in schedule if row["key"] == "5-baseline-successor")
    command = next(row for row in ledger if row["key"] == planned["key"])
    own_id = AUDIT["event_identity"](context, planned["payload"])[1]
    interval = dict(start_ns=command["start_ns"], end_ns=command["end_ns"], event_ids=[own_id])
    traces = {"baseline": {"ack_intervals": [interval]}}
    counts = AUDIT["prior_ack_overlap"](schedule, ledger, traces, context)
    assert counts["baseline"]["successor"] == 0
    primer = next(row for row in schedule if row["key"] == "5-baseline-primer")
    interval["event_ids"] = [AUDIT["event_identity"](context, primer["payload"])[1]]
    assert (
        AUDIT["prior_ack_overlap"](schedule, ledger, traces, context)["baseline"]["successor"] == 1
    )


def durable_fixture(folder):
    folder.mkdir()
    context = dict(
        user_profile_id="ack-preflight",
        project_id="b8268f16-0c07-4b4e-91df-b3fa3636374f",
        workspace_id=None,
        session_id="owner-preflight",
        host="codex",
    )
    payload = dict(turn_id="preflight-0", last_assistant_message="Retain owner preflight item 0.")
    original = "codex:5eddda44d544629a45938d2d4c075ad16b00b8842a16e1150fca98225ef31b4c"
    queued = "4474ae5632701ccfc86ed37b62b96bc9e91ab893e6ce95d82d0f6f68cf84d1a5"
    assert AUDIT["event_identity"](context, payload) == (original, queued)
    metadata = dict(
        context, source_event_id=original, source_ref="Stop:preflight-0", source_kind="codex_hook"
    )
    event = dict(
        id=queued,
        source_event_id=original,
        source="codex",
        role="assistant",
        content=payload["last_assistant_message"],
        scope="project",
        scope_id=context["project_id"],
        user_profile_id=context["user_profile_id"],
        session_id=context["session_id"],
        metadata=json.dumps(metadata),
    )
    db = sqlite3.connect(folder / "memory.sqlite")
    db.execute("CREATE TABLE episodic_events (" + ",".join(f"{key} TEXT" for key in event) + ")")
    db.execute(
        "INSERT INTO episodic_events VALUES (" + ",".join("?" for _ in event) + ")",
        list(event.values()),
    )
    db.execute("CREATE TABLE memory_jobs (status TEXT, kind TEXT, payload TEXT)")
    db.execute(
        "INSERT INTO memory_jobs VALUES (?,?,?)",
        (
            "completed",
            "extract",
            json.dumps(dict(event_id=queued, context=context, scope="project")),
        ),
    )
    db.commit()
    db.close()
    db = sqlite3.connect(folder / "outbox.sqlite")
    db.execute("CREATE TABLE outbox_entries (event_id TEXT)")
    db.commit()
    db.close()
    trace = dict(
        activity=dict(ack=0, replay=0),
        ack_intervals=[dict(start_ns=1, end_ns=2, event_ids=[queued], acknowledged=1)],
        responses=[
            dict(
                status="handled",
                dispatched=True,
                captured=1,
                queued=1,
                capture_state="queued",
                event_ids=[original],
                output={"continue": True},
                host="codex",
                environment_sha256="test",
                hook_protocol=1,
                schema_version=1,
                request_id="request-one",
            )
        ],
    )
    return [dict(payload=payload)], trace, context


def test_durable_receipt_fixture(tmp_path):
    folder = tmp_path / "arm"
    planned, trace, context = durable_fixture(folder)
    AUDIT["validate_durable"](folder, planned, trace, context, "test")


@pytest.mark.parametrize(
    "change", ["content", "identity", "missing", "job", "pending", "ack", "receipt"]
)
def test_durable_corruption_is_rejected(tmp_path, change):
    folder = tmp_path / "arm"
    planned, trace, context = durable_fixture(folder)
    if change in ("content", "identity", "missing", "job"):
        db = sqlite3.connect(folder / "memory.sqlite")
        sql = {
            "content": "UPDATE episodic_events SET content='wrong'",
            "identity": "UPDATE episodic_events SET source_event_id='wrong'",
            "missing": "DELETE FROM episodic_events",
            "job": "UPDATE memory_jobs SET status='pending'",
        }[change]
        db.execute(sql)
        db.commit()
        db.close()
    elif change == "pending":
        db = sqlite3.connect(folder / "outbox.sqlite")
        db.execute("INSERT INTO outbox_entries VALUES ('pending')")
        db.commit()
        db.close()
    elif change == "ack":
        trace["ack_intervals"][0]["acknowledged"] = 0
    else:
        trace["responses"][0]["event_ids"] = ["wrong"]
    with pytest.raises(AssertionError):
        AUDIT["validate_durable"](folder, planned, trace, context, "test")
