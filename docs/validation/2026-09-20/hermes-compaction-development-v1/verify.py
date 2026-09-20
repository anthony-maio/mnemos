import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parent
for item in json.loads((root / "artifact-index.json").read_text())["files"]:
    raw = (root / item["path"]).read_bytes()
    assert len(raw) == item["bytes"]
    assert hashlib.sha256(raw).hexdigest() == item["sha256"], item["path"]
expected = {
    "unit-red-v1": (10, 10), "unit-red-v2": (12, 12),
    "classification-red-v1": (2, 2), "unit-green-v1": (38, 0),
    "unit-green-v2": (40, 0), "unit-green-v3": (21, 1),
    "unit-green-v4": (76, 0),
}
for name, counts in expected.items():
    suite = ET.parse(root / "tests" / f"mnemos-compaction-{name}.xml").getroot().find("testsuite")
    assert suite is not None
    assert (int(suite.attrib["tests"]), int(suite.attrib["failures"])) == counts
    assert int(suite.attrib["errors"]) == 0
report = json.loads((root / "host-diagnostic/report.json").read_text())
assert report["status"] == "diagnostic_observed"
assert report["network_attempts"] == 0 and report["host_unchanged"] is True
assert report["host_files_before"] == report["host_files_after"]
assert report["runner_sha256"] == hashlib.sha256((root / "host-diagnostic/run.py.txt").read_bytes()).hexdigest()
assert report["database"]["original_parent_ids"] == [1, 2, 3, 4]
assert report["database"]["child_loaded_ids"] == [5, 6, 8, 9]
assert len(report["adapter"]["initial_payloads"]) == 4
assert len(report["adapter"]["child_payloads"]) == 4
assert all(p.get("identity_status", "normal") == "normal" for p in report["adapter"]["child_payloads"])
assert "Derived compaction summary:" in report["adapter"]["summary_payloads"][0]["content"]
assert not any(m.get("_compressed_summary") for m in report["adapter"]["child_loaded_messages_no_repair"])
print("Verified 27 indexed artifacts, original failures, 76-case green run, and bounded host diagnostic.")
