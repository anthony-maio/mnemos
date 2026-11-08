"""Offline audit of preserved native evidence; never rewrites the original result."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import runpy
import sqlite3


def audit(root, job, control):
    helpers = runpy.run_path(str(Path(__file__).with_name("registry_claude_session.py")))
    common = runpy.run_path(str(Path(__file__).with_name("registry_mcp_session.py")))
    read = lambda name: json.loads((root / name).read_bytes())
    frozen, native, outer = (
        read("native-freeze.json"),
        read("native-result.json"),
        read("result.json"),
    )
    assert all(helpers["sha"](p) == h for p, h in frozen["files"].items())
    target = Path(outer["package"]["installed_target"])
    assert all(
        helpers["sha"](target / p) == h for p, h in outer["package"]["package_files"].items()
    )
    assert native["status"] == outer["status"] == "failed"
    assert native["error_type"] == outer["error_type"] == "AssertionError"
    assert not native["cleanup_errors"]
    assert all(
        native[k] for k in ("auth_copy_removed", "auth_source_unchanged", "inputs_unchanged")
    )
    assert all(outer[k] for k in ("owner_stopped", "port_closed", "package_unchanged"))
    supervised = json.loads((job / "process.json").read_bytes())
    assert supervised["job_assigned"] and supervised["active_after_cleanup"] == 0
    cleanup = json.loads(control.read_bytes())
    assert (
        cleanup["auth_copy_removed"]
        and cleanup["watched_unchanged"]
        and not cleanup["cleanup_errors"]
    )
    assert not (root / "claude-home/.credentials.json").exists()
    stderr = (job / "stderr.log").read_text(encoding="utf-8")
    assert 'assert len(init) == 1 and init[0]["mcp_servers"]' in stderr
    records = [
        json.loads(line) for line in (root / "native-stdout.jsonl").read_bytes().splitlines()
    ]
    assert len(helpers["parse_calls"](records)) == 12
    assert all(r.get("session_id", native["session_id"]) == native["session_id"] for r in records)
    expected = [
        ["mnemos_store", "mnemos_consolidate", "mnemos_job_status", "mnemos_inspect"],
        ["mnemos_retrieve", "mnemos_store", "mnemos_forget"],
        [
            "mnemos_inspect",
            "mnemos_retrieve",
            "mnemos_store",
            "mnemos_consolidate",
            "mnemos_job_status",
        ],
    ]
    phases, results = [], []
    start = 1
    for number, phase in enumerate(native["phases"]):
        assert phase["phase"] == number and phase["source_start_line"] == start
        segment = records[start - 1 : phase["source_end_line"]]
        start = phase["source_end_line"] + 1
        init = [r for r in segment if r.get("type") == "system" and r.get("subtype") == "init"]
        assert len(init) == 1
        assert init[0]["mcp_servers"] == [{"name": "mnemos", "status": "connected"}]
        assert init[0]["model"] == "claude-sonnet-4-6"
        result = [r for r in segment if r.get("type") == "result"]
        assert len(result) == 1 and result[0]["subtype"] == "success" and not result[0]["is_error"]
        assert not result[0]["permission_denials"]
        calls = helpers["parse_calls"](segment)
        assert calls == read(f"phase-{number}-calls.json")
        assert [c["name"] for c in calls] == expected[number]
        assert phase["owner_unchanged"] and phase["mcp_owner"] == read("mcp-owner.json")
        phases.append(calls)
        results.append(result[0])
    assert len(phases) == 3 and start == len(records) + 1
    original = phases[0][3]["result"]["fact"]
    project_id = original["metadata"]["scope_id"]
    assert original["content"] == helpers["MARKER"]
    for call in [phases[0][0], phases[2][2]]:
        assert call["arguments"]["scope"] == "project"
        helpers["require_project"](call["result"], project_id)
        assert call["result"]["status"] == "committed" and call["result"]["event_committed"]
    for call in phases[1]:
        common["require_scope_denial"](
            json.dumps(call["result"]),
            store_scope_id=project_id if call["name"] == "mnemos_store" else None,
        )
    assert phases[1][1]["arguments"]["content"] == helpers["BLOCKED"]
    assert phases[1][2]["arguments"] == dict(
        chunk_id=original["id"], expected_version=original["version"]
    )
    denied = read("native-revocation-state.json")
    assert denied["before"] == denied["after"]
    restored = phases[2][0]["result"]["fact"]
    assert restored == original
    for phase, index in ((1, 0), (2, 1)):
        assert phases[phase][index]["arguments"]["allowed_scopes"] == "project"
    recalled = phases[2][1]["result"]["facts"]
    assert len(recalled) == 1 and recalled[0]["id"] == original["id"]
    assert recalled[0]["content"] == helpers["MARKER"]
    helpers["require_project"](recalled[0], project_id)
    assert phases[2][2]["arguments"]["content"] == helpers["SECOND"]
    helpers["require_durable_project"](root / "memory.sqlite", project_id, 2)
    with closing(sqlite3.connect((root / "memory.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
        assert db.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        facts = dict(db.execute("SELECT id,content FROM memory_chunks"))
        assert set(facts.values()) == {helpers["MARKER"], helpers["SECOND"]}
        events = dict(db.execute("SELECT id,content FROM episodic_events"))
        assert set(events.values()) == set(facts.values())
        jobs = {
            row[0]: (row[1], json.loads(row[2]))
            for row in db.execute("SELECT id,status,result FROM memory_jobs")
        }
        assert len(jobs) == 2
        for store, completed in ((phases[0][0], phases[0][2]), (phases[2][2], phases[2][4])):
            payload = store["result"]
            assert events[payload["event_id"]] == store["arguments"]["content"]
            job_record = completed["result"]["job"]
            assert job_record["id"] == payload["job_id"]
            assert jobs[payload["job_id"]] == ("completed", job_record["result"])
            assert len(job_record["result"]["fact_ids"]) == 1
            assert facts[job_record["result"]["fact_ids"][0]] == store["arguments"]["content"]
    final = common["snapshot"](root / "memory.sqlite")
    assert final["memory_suppressions"]["count"] == 0
    return dict(
        status="passed",
        original_run_status="failed",
        amendment="Require one initialization report per turn, not per process; preserve all other checks.",
        native_calls=12,
        denied_calls=3,
        phases=3,
        final=final,
        cumulative_cost_usd=results[-1]["total_cost_usd"],
        evidence_sha256={
            name: helpers["sha"](root / name)
            for name in (
                "native-freeze.json",
                "native-stdout.jsonl",
                "native-result.json",
                "result.json",
                "memory.sqlite",
                "native-revocation-state.json",
                "mcp-owner.json",
                "phase-0-calls.json",
                "phase-1-calls.json",
                "phase-2-calls.json",
            )
        },
        supporting_sha256={
            str(path): helpers["sha"](path)
            for path in (Path(__file__).resolve(), job / "process.json", job / "stderr.log", control)
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("job", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.job.resolve(), args.control.resolve())
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "native_calls", "denied_calls", "phases")}))
