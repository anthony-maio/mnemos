"""Replay the diagnostic against hashed saved inputs without model calls."""

from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import zipfile

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))
from benchmarks.local_memory.retrieval_miss_audit import analyze

directory = Path(__file__).parent
dataset_raw = (directory / "dataset-exact.jsonl").read_bytes()
assert hashlib.sha256(dataset_raw).hexdigest() == "036fa51ee13271a85a14a62a2fe4ef65d13eaeb290331fe325aa5bf32ce893fc"
with zipfile.ZipFile(root / "docs/validation/2026-09-19/retrieval-terminal-local-v1/retrieval-reports.zip") as archive:
    report_raw = archive.read("full.json")
assert hashlib.sha256(report_raw).hexdigest() == "9553827a7c5f8b14079607c7fb5b38a4c40457d8987ad41efac1b825a028b8de"
report = json.loads(report_raw)
sessions = [row for row in map(json.loads, dataset_raw.splitlines()) if row.get("record_type") == "session"]
before = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
result = analyze(report, sessions)
assert hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest() == before
assert len(result["observations"]) == 1156
assert len(result["partial_observations"]) == 44
expected = {}
for observation in report["raw_outputs"]:
    for comparison in observation["raw_output"]["retrieval_comparisons"]:
        if comparison["quality_kind"] != "answerable":
            continue
        for strategy, detail in comparison["strategies"].items():
            key = (observation["session_id"], observation["repetition"], comparison["query_id"], strategy)
            assert key not in expected
            expected[key] = detail["quality_hit"]
seen = set()
for row in result["observations"] + result["partial_observations"]:
    key = (row["session_id"], row["repetition"], row["query_id"], row["strategy"])
    assert key not in seen
    seen.add(key)
    assert row["original_quality_hit"] is expected[key]
assert seen == set(expected)
full = [row for row in result["observations"] if row["strategy"] == "full"]
assert len(full) == 289
assert Counter(row["original_quality_hit"] for row in full) == {True: 106, False: 183}
result["input_binding"] = {
    "report_sha256": hashlib.sha256(report_raw).hexdigest(),
    "dataset_sha256": hashlib.sha256(dataset_raw).hexdigest(),
    "diagnostic_source_sha256": hashlib.sha256((root / "benchmarks/local_memory/retrieval_miss_audit.py").read_bytes()).hexdigest(),
    "unchanged_strategy_verdicts_verified": len(seen),
    "report_not_mutated": True,
}
(directory / "all-strategy-boundaries.json").write_bytes((json.dumps(result, indent=2) + "\n").encode())
print(json.dumps({"complete_strategy_observations": len(result["observations"]), "partial_strategy_observations": len(result["partial_observations"]), "unchanged_verdicts": len(seen), "full_original_hits": 106, "full_original_misses": 183, "boundary_counts": result["failure_boundary_counts"]}, indent=2))
