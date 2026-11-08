"""Independent, read-only audit of frozen replay comparison artifacts.

No product or runner imports. Reconstruct identities and reopen SQLite directly.
"""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import zipfile

ARMS = ("baseline", "candidate")
POSITIONS = ("primer", "successor")
EXPECTED_STDERR = "mnemos hook diagnostic host=codex event=Stop status=degraded"


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return dict(
        n=len(values),
        min=ordered[0],
        median=statistics.median(ordered),
        p95=ordered[math.ceil(0.95 * len(ordered)) - 1],
        max=ordered[-1],
    )


def delivered(row):
    try:
        return (
            row["status"] == "executed"
            and row["returncode"] == 0
            and json.loads(row["stdout"]) == {"continue": True}
            and row["stderr"].strip() == EXPECTED_STDERR
        )
    except (KeyError, ValueError, TypeError, AttributeError):
        return False


def event_identity(context, payload):
    parts = [
        context["user_profile_id"],
        context["project_id"],
        context["workspace_id"],
        context["session_id"],
        "Stop",
        payload["turn_id"],
        "assistant",
        payload["last_assistant_message"],
    ]
    original = (
        "codex:"
        + hashlib.sha256(
            json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
    )
    queued = hashlib.sha256(
        json.dumps([context["user_profile_id"], "codex", context["session_id"], original]).encode()
    ).hexdigest()
    return original, queued


def validate_schedule(freeze, root):
    expected = []
    for index in range(55):
        for arm in (ARMS if index % 2 == 0 else ARMS[::-1]):
            for position in POSITIONS:
                expected.append(
                    dict(
                        key=f"{index}-{arm}-{position}",
                        round=index,
                        arm=arm,
                        position=position,
                        warmup=index < 5,
                        payload=dict(
                            hook_event_name="Stop",
                            cwd=str(root / "project"),
                            session_id="ack-comparison-session",
                            turn_id=f"round-{index}-{position}",
                            timestamp="2026-11-08T00:00:00+00:00",
                            last_assistant_message=f"Retain acknowledgement comparison round {index} {position}.",
                        ),
                    )
                )
    assert freeze["schedule"] == expected, "Schedule differs from the fixed 55-round protocol"
    assert freeze["warmup_rounds"] == 5 and freeze["measured_rounds"] == 50
    context = freeze["context"]
    assert context["user_profile_id"] == "ack-comparison"
    assert context["session_id"] == "ack-comparison-session" and context["host"] == "codex"
    assert context["workspace_id"] is None and context["project_id"]
    return expected


def validate_ledger(schedule, ledger):
    assert [row["key"] for row in ledger] == [
        row["key"] for row in schedule
    ], "Missing, duplicate or reordered commands"
    previous = 0
    for row in ledger:
        assert row["status"] in {"unexecuted", "in_flight", "executed", "timeout", "failed"}
        if "start_ns" in row:
            assert type(row["start_ns"]) is int and type(row["end_ns"]) is int
            assert previous <= row["start_ns"] < row["end_ns"], "Invalid command interval"
            previous = row["end_ns"]
    assert all(
        delivered(row) for row in ledger
    ), "Not every scheduled native command has a successful delivery outcome"


def summarize(schedule, ledger, traces):
    by_key = {row.get("key"): row for row in ledger}
    result = dict(
        scheduled=len(schedule),
        recorded=len(ledger),
        status_counts=dict(Counter(row.get("status", "missing") for row in ledger)),
        arms={},
    )
    for arm in ARMS:
        selected = [row for row in schedule if row["arm"] == arm and not row["warmup"]]
        intervals = traces.get(arm, {}).get("ack_intervals", [])
        arm_result = {}
        for position in (*POSITIONS, "all"):
            group = [row for row in selected if position == "all" or row["position"] == position]
            values, overlap_count = [], 0
            for planned in group:
                row = by_key.get(planned["key"], {})
                start, end = row.get("start_ns"), row.get("end_ns")
                if type(start) is int and type(end) is int and end >= start:
                    values.append((end - start) / 1_000_000)
                    overlap_count += any(
                        a["start_ns"] < end and start < a["end_ns"] for a in intervals
                    )
            arm_result[position] = dict(
                command_ms=distribution(values),
                scheduled=len(group),
                delivered=sum(delivered(by_key.get(row["key"], {})) for row in group),
                missing_timing=len(group) - len(values),
                commands_overlapping_ack=overlap_count,
            )
        result["arms"][arm] = arm_result
    pairs = {position: [] for position in (*POSITIONS, "all")}
    for index in range(5, 55):
        durations = {}
        for arm in ARMS:
            durations[arm] = {}
            for position in POSITIONS:
                row = by_key.get(f"{index}-{arm}-{position}", {})
                if type(row.get("start_ns")) is int and type(row.get("end_ns")) is int:
                    durations[arm][position] = (row["end_ns"] - row["start_ns"]) / 1_000_000
        if all(len(durations[arm]) == 2 for arm in ARMS):
            for position in POSITIONS:
                pairs[position].append(
                    dict(
                        round=index,
                        candidate_minus_baseline_ms=durations["candidate"][position]
                        - durations["baseline"][position],
                    )
                )
            pairs["all"].append(
                dict(
                    round=index,
                    candidate_minus_baseline_ms=sum(durations["candidate"].values())
                    - sum(durations["baseline"].values()),
                )
            )
    result["paired_round_differences"] = pairs
    result["paired_difference_distributions_ms"] = {
        key: distribution([row["candidate_minus_baseline_ms"] for row in values])
        for key, values in pairs.items()
    }
    return result


def validate_durable(folder, planned, trace, context, environment_sha):
    assert trace["activity"] == {"ack": 0, "replay": 0}
    assert len(trace["responses"]) == len(planned)
    with closing(sqlite3.connect((folder / "memory.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        events = [dict(row) for row in db.execute("SELECT * FROM episodic_events")]
        jobs = [dict(row) for row in db.execute("SELECT * FROM memory_jobs")]
    assert len(events) == len(jobs) == len(planned), "Missing or duplicate durable event/job"
    event_map = {row["id"]: row for row in events}
    jobs_by_event = {}
    for job in jobs:
        payload = json.loads(job["payload"])
        assert job["status"] == "completed" and job["kind"] == "extract"
        assert payload["event_id"] not in jobs_by_event, "Duplicate extraction job"
        assert payload["context"] == context and payload["scope"] == "project"
        jobs_by_event[payload["event_id"]] = job
    expected_queue_ids = []
    request_ids = []
    for row, response in zip(planned, trace["responses"]):
        original_id, queued_id = event_identity(context, row["payload"])
        expected_queue_ids.append(queued_id)
        assert response["status"] == "handled" and response["dispatched"] is True
        assert (
            response["captured"] == response["queued"] == 1
            and response["capture_state"] == "queued"
        )
        assert response["event_ids"] == [original_id] and response["output"] == {"continue": True}
        assert response["host"] == "codex" and response["environment_sha256"] == environment_sha
        assert response["hook_protocol"] == response["schema_version"] == 1
        request_ids.append(response["request_id"])
        event = event_map[queued_id]
        assert event["source_event_id"] == original_id and event["source"] == "codex"
        assert (
            event["role"] == "assistant"
            and event["content"] == row["payload"]["last_assistant_message"]
        )
        assert event["scope"] == "project" and event["scope_id"] == context["project_id"]
        assert (
            event["user_profile_id"] == context["user_profile_id"]
            and event["session_id"] == context["session_id"]
        )
        metadata = json.loads(event["metadata"])
        assert all(metadata[key] == value for key, value in context.items())
        assert (
            metadata["source_event_id"] == original_id
            and metadata["source_ref"] == "Stop:" + row["payload"]["turn_id"]
        )
        assert metadata["source_kind"] == "codex_hook" and queued_id in jobs_by_event
    assert len(set(request_ids)) == len(planned)
    acknowledged = []
    for interval in trace["ack_intervals"]:
        assert type(interval["start_ns"]) is int and interval["start_ns"] < interval["end_ns"]
        assert "error_type" not in interval and interval["acknowledged"] == len(
            interval["event_ids"]
        )
        acknowledged.extend(interval["event_ids"])
    assert Counter(acknowledged) == Counter(
        expected_queue_ids
    ), "Acknowledgement trace does not match durable events"
    with closing(sqlite3.connect((folder / "outbox.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
        assert db.execute("SELECT count(*) FROM outbox_entries").fetchone()[0] == 0
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    if (folder / "outbox.json").exists():
        assert read(folder / "outbox.json") == []


def prior_ack_overlap(schedule, ledger, traces, context):
    """Distinguish prior-event contention from a command's own later acknowledgement."""
    by_key = {row["key"]: row for row in ledger}
    counts = {arm: {position: 0 for position in (*POSITIONS, "all")} for arm in ARMS}
    for arm in ARMS:
        earlier_ids = set()
        intervals = traces.get(arm, {}).get("ack_intervals", [])
        for planned in (row for row in schedule if row["arm"] == arm):
            row = by_key.get(planned["key"], {})
            if not planned["warmup"] and "start_ns" in row and "end_ns" in row:
                overlaps = any(
                    interval["start_ns"] < row["end_ns"]
                    and row["start_ns"] < interval["end_ns"]
                    and earlier_ids.intersection(interval["event_ids"])
                    for interval in intervals
                )
                counts[arm][planned["position"]] += bool(overlaps)
                counts[arm]["all"] += bool(overlaps)
            earlier_ids.add(event_identity(context, planned["payload"])[1])
    return counts


def audit(root):
    report = dict(status="failed", errors=[], summary=None)
    try:
        freeze, ledger = read(root / "freeze.json"), read(root / "ledger.json")
        traces = {
            arm: read(root / arm / "final-trace.json")
            for arm in ARMS
            if (root / arm / "final-trace.json").exists()
        }
        report["summary"] = summarize(freeze["schedule"], ledger, traces)
        schedule = validate_schedule(freeze, root)
        report["summary"]["commands_overlapping_prior_event_ack"] = prior_ack_overlap(
            schedule, ledger, traces, freeze["context"]
        )
        report["summary"][
            "paired_all_definition"
        ] = "Sum of primer and successor command milliseconds, candidate minus baseline"
        by_key = {row["key"]: row for row in ledger}
        validate_ledger(schedule, ledger)
        assert [freeze["arms"][a]["source_commit"] for a in ARMS] == [
            "7a77e40f4ec8dd5b2fbf47ff0b1e3f0862e0129d",
            "ec4a36b262be61c16d807a2b39006bcad7f8367c",
        ]
        assert (
            sha(freeze["native"])
            == "b154c81345e4a7473086b946599baf2535daafc627133b1131984d7091914683"
        )
        here = Path(__file__).resolve().parent
        required = [
            here / name
            for name in (
                "replay_ack_comparison.py",
                "replay_ack_owner.py",
                "replay_ack_audit.py",
                "windows_process_identity.py",
                "windows_owned_job.py",
            )
        ]
        required.extend(
            [
                Path(freeze["native"]),
                Path(freeze["protocol"]),
                Path(freeze["git_executable"]),
                Path(freeze["git_runtime_manifest"]),
                root / "scopes.json",
                root / "project/settings.toml",
                root / "project/.mnemos/mnemos.toml",
            ]
        )
        assert all(str(path.resolve()) in freeze["files"] for path in required)
        execution = read(root / "execution.json")
        assert execution["status"] == "completed"
        assert sha(root / "freeze.json") == execution["freeze_sha256"]
        assert execution["freeze_unchanged"] and execution["dependencies_unchanged"]
        assert execution["git_resolution_unchanged"]
        assert set(execution["files_unchanged"]) == set(freeze["files"])
        assert all(execution["files_unchanged"].values())
        assert all(
            Path(path).is_file() and sha(path) == digest for path, digest in freeze["files"].items()
        )
        assert freeze["prepared_ns"] < execution["started_ns"] <= ledger[0]["start_ns"]
        assert execution["finished_ns"] >= ledger[-1]["end_ns"]
        supervisor = read(root / "supervisor/process.json")
        assert supervisor["status"] == "passed" and supervisor["job_assigned"]
        assert supervisor["job_name"] == freeze["job_name"]
        assert supervisor["exit_code"] == supervisor["active_after_cleanup"] == 0
        registry = read(root / "scopes.json")
        project = registry["projects"][freeze["context"]["project_id"]]
        assert project["enabled"] and project["user_profile_id"] == "ack-comparison"
        assert Path(project["root"]).resolve() == (root / "project").resolve()
        wheel_members = {}
        for arm in ARMS:
            info = freeze["arms"][arm]
            wheels = [
                Path(p)
                for p in freeze["files"]
                if p.endswith(".whl") and Path(p).parent.parent.name == arm
            ]
            assert len(wheels) == 1
            with zipfile.ZipFile(wheels[0]) as archive:
                wheel_members[arm] = {name: archive.read(name) for name in archive.namelist()}
            for name, digest in info["package_files"].items():
                assert hashlib.sha256(wheel_members[arm][name]).hexdigest() == digest
                assert sha(Path(info["installed_target"]) / name) == digest
            cleanup = execution["cleanup"][arm]
            assert cleanup["exit_code"] == 0 and cleanup["owner_absent"]
            assert cleanup["ports_closed"] == {"memory_port": True, "worker_port": True}
            assert cleanup["stopped"]["stopped"] and not cleanup["stopped"]["cleanup_errors"]
            assert read(root / arm / "stopped.json") == cleanup["stopped"]
            owner = execution["owners"][arm]
            assert owner["proof"]["owner_pid"] == owner["ready"]["pid"]
            assert owner["ready"]["full_sync"] and owner["ready"]["integrity_check"] == "ok"
            assert owner["ready"]["environment_sha256"] == freeze["environment_sha256"][arm]
            planned = [row for row in schedule if row["arm"] == arm]
            validate_durable(
                root / arm,
                planned,
                traces[arm],
                freeze["context"],
                freeze["environment_sha256"][arm],
            )
            for index in range(55):
                for phase in ("after", "end"):
                    barrier = read(root / arm / f"barrier-{index}-{phase}.json")
                    count = 2 * (index + 1)
                    assert barrier["events"] == barrier["jobs"] == count
                    assert (
                        barrier["pending"] == barrier["ack"] == barrier["replay"] == 0
                        and barrier["pumps_idle"]
                    )
                    assert barrier["responses"] == traces[arm]["responses"][:count]
                    # The final trace must retain every earlier observation.
                    assert (
                        barrier["ack_intervals"]
                        == traces[arm]["ack_intervals"][: len(barrier["ack_intervals"])]
                    )
                    assert all(
                        a["end_ns"] <= barrier["observed_ns"] for a in barrier["ack_intervals"]
                    )
                    assert by_key[f"{index}-{arm}-successor"]["end_ns"] <= barrier["observed_ns"]
                    ordered_arms = ARMS if index % 2 == 0 else ARMS[::-1]
                    if phase == "after" and arm == ordered_arms[0]:
                        assert (
                            barrier["observed_ns"]
                            <= by_key[f"{index}-{ordered_arms[1]}-primer"]["start_ns"]
                        )
                    if index < 54:
                        first_next = "baseline" if (index + 1) % 2 == 0 else "candidate"
                        assert (
                            barrier["observed_ns"]
                            <= by_key[f"{index + 1}-{first_next}-primer"]["start_ns"]
                        )
        assert wheel_members["baseline"].keys() == wheel_members["candidate"].keys()
        changed = {
            name
            for name, value in wheel_members["baseline"].items()
            if wheel_members["candidate"][name] != value
        }
        assert changed == {"mnemos/client.py", "mnemos_memory-0.7.0b1.dist-info/RECORD"}
        report["status"] = "passed"
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["synthetic_complete_command_p95_under_100ms"] = {
        arm: report["status"] == "passed"
        and report["summary"]["arms"][arm]["all"]["command_ms"]["p95"] <= 100
        for arm in ARMS
    }
    report["limits"] = [
        "Synthetic simple-embedding transport comparison; original acceptance gates remain open.",
        "Acknowledgement intervals include lock waits, not only COMMIT.",
        "No or little overlap means the scheduling mechanism was insufficiently exercised; no rerun is authorized.",
        "All observed timings, including failed commands, remain in descriptive distributions; missing outcomes prevent a pass.",
    ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root.resolve())
    # Never replace an earlier audit result.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "errors": report["errors"]}), flush=True)
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
