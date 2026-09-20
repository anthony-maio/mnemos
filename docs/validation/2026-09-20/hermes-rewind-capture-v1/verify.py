"""Verify retained bytes and both arms' actual capture acceptance results."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
index = json.loads((root / "artifact-index.json").read_text())
for item in index["files"]:
    data = (root / item["path"]).read_bytes()
    assert len(data) == item["bytes"], item["path"]
    assert hashlib.sha256(data).hexdigest() == item["sha256"], item["path"]

for arm in ("base", "candidate"):
    report = json.loads((root / f"{arm}-report.json").read_text())
    assert report["network_attempts"] == 0
    assert report["host_unchanged"] is True
    assert report["host_files_before"] == report["host_files_after"]
    assert report["host_commit"] == "14a346345486418aa85ed6526549412ee2665ade"
    assert report["wheel_sha256"] == "9af6667d86933570beecc940e55e637f70a9258e5339e948dc48ba9692782bf6"
    assert report["runner_sha256"] == hashlib.sha256((root / "hermes_rewind_capture_v1.py.txt").read_bytes()).hexdigest()
    assert [case["operation"] for case in report["cases"]] == ["undo", "retry"]
    expected_status = "failed" if arm == "base" else "passed"
    assert report["status"] == expected_status
    for case in report["cases"]:
        assert case["status"] == expected_status
        phases = case["phases"]
        counts = [len(phase["pending"]) for phase in phases]
        assert counts == ([4, 4, 8] if arm == "base" else [4, 4, 6, 6, 6])
        snapshots = [{p["event_id"]:p for p in phase["pending"]} for phase in phases]
        assert all(len(snapshot) == count for snapshot, count in zip(snapshots, counts))
        assert snapshots[0] == snapshots[1]
        assert all(snapshots[2].get(key) == value for key, value in snapshots[0].items())
        if arm == "base":
            assert case["error"] == "AssertionError: expected six observations, got 8"
            assert phases[2]["input_row_ids"] == [None, None, 5, 6]
            assert sum(p.get("identity_status") == "provisional_unassociated" for p in snapshots[2].values()) == 4
        else:
            assert phases[2]["input_row_ids"] == [1, 2, 5, 6]
            assert phases[4]["input_row_ids"] == [1, 2]
            assert snapshots[2] == snapshots[3] == snapshots[4]
            assert all(p.get("identity_status") != "provisional_unassociated" for p in snapshots[2].values())
            by_ref = {p["source_ref"]:p for p in snapshots[2].values()}
            assert set(by_ref) == {f"transcript:v2:{i}" for i in range(1, 7)}
            for old, new in ((3, 5), (4, 6)):
                a, b = by_ref[f"transcript:v2:{old}"], by_ref[f"transcript:v2:{new}"]
                assert (a["role"], a["content"]) == (b["role"], b["content"])
                assert a["event_id"] != b["event_id"]
print("Verified 19 indexed artifacts; base failed both cases, candidate passed both cases.")
