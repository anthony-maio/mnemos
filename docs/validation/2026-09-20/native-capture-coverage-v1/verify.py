"""Reproduce receipt coverage from the indexed native text projections."""

from pathlib import Path
import hashlib
import json
import sys

archive = Path(__file__).resolve().parent
root = archive.parents[3]
sys.path.insert(0, str(root))
from benchmarks.native_hosts.capture_coverage import expected_messages, reconcile_captures

index = json.loads((archive / "evidence-index.json").read_bytes())
for name, item in index["files"].items():
    raw = (archive / name).read_bytes()
    assert len(raw) == item["bytes"]
    assert hashlib.sha256(raw).hexdigest() == item["sha256"], name
bindings = json.loads((archive / "source-bindings.json").read_bytes())
verifier = (root / bindings["verifier_file"]).read_bytes()
assert hashlib.sha256(verifier).hexdigest() == bindings["verifier"]["sha256"]
expected = []
for source in bindings["sources"]:
    raw = (archive / source["selected_path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == source["selected"]["sha256"]
    rows = [json.loads(line) for line in raw.splitlines()]
    messages = expected_messages(source["host"], source["phase"], rows)
    assert len(messages) == source["full_source_expected_count"]
    expected.extend(messages)
dispatches = json.loads((archive / "dispatch-records.json").read_bytes())
state = json.loads((archive / "final-state.json").read_bytes())
actual = reconcile_captures(expected, dispatches, scope=bindings["scope"], jobs=state["jobs"])
assert actual == json.loads((archive / "report.json").read_bytes())
print(json.dumps({"status": actual["status"], "expected": actual["expected_messages"], "covered": actual["covered_messages"], "indexed_files": len(index["files"])}))
