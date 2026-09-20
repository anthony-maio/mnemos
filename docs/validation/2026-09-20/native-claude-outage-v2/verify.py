"""Recheck archived native recovery without contacting a host or service."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
index = json.loads((root / "evidence-index.json").read_bytes())["files"]
for name, expected in index.items():
    raw = (root / name).read_bytes()
    assert len(raw) == expected["bytes"]
    assert hashlib.sha256(raw).hexdigest() == expected["sha256"], name
report = json.loads((root / "report.json").read_bytes())
source_map = json.loads((root / "source-map.json").read_bytes())
for original, copied in source_map.items():
    assert hashlib.sha256((root / copied).read_bytes()).hexdigest() == report["input_hashes"][original]
sys.path.insert(0, str(root / "sources"))
from native_hosts.claude_service_outage_v1 import stable
from native_hosts.claude_service_outage_v2 import PROMPT
from native_hosts.capture_coverage import expected_messages, reconcile_captures
from native_hosts.outage_acceptance import (
    require_health_gate, require_native_transcript, require_pending_sources_v2,
)

def read(name):
    return json.loads((root / name).read_bytes())

native_bytes = (root / "native-transcript.jsonl").read_bytes()
assert hashlib.sha256(native_bytes).hexdigest() == report["native_transcript"]["sha256"]
native = [json.loads(line) for line in native_bytes.splitlines()]
require_native_transcript(native, submitted_marker=PROMPT,
                          expected_ack="Acknowledged birch-window-618.",
                          expected_model="claude-sonnet-4-6")
process = read("claude-offline/process.json")
assert process == report["native_process"]
assert process["status"] == "completed" and process["returncode"] == 0
assert process["auth_copy_removed"] and not process["timed_out"]
for row in native:
    if row.get("type") in {"user", "assistant"}:
        assert row["sessionId"] == process["session_id"]
for stream, filename in (("stdout", "stdout.jsonl"), ("stderr", "stderr.log")):
    assert hashlib.sha256((root / "claude-offline" / filename).read_bytes()).hexdigest() == process[stream + "_sha256"]
pending = read("outbox-before-recovery.json")
before, final = read("state-after-recovery.json"), read("final-state.json")
require_pending_sources_v2(before, pending)
require_pending_sources_v2(final, pending)
assert stable(before) == stable(final)
health, modules = read("health-responses.json"), read("injected-module-configs.json")
assert len(health) == len(modules) == 3
for response, config in zip(health, modules):
    require_health_gate(response, expected_database=report["isolation"]["resolved_database"],
                        module_config=config)
dispatches = read("dispatch-records.json")
captures = [i for i, row in enumerate(dispatches) if row["operation"] == "capture"]
assert len(captures) == 2 * len(pending)
for position, i in enumerate(captures):
    entry = pending[position % len(pending)]
    assert dispatches[i]["payload"] == entry["payload"]
    assert dispatches[i]["result"]["status"] == "committed"
    assert dispatches[i]["result"]["event_id"] == entry["event_id"]
expected = expected_messages("claude-code", "claude-offline", native)
context = pending[0]["payload"]["context"]
coverage = reconcile_captures(expected, dispatches[:captures[len(pending)-1]+1],
    scope={key: context[key] for key in ("user_profile_id", "project_id", "workspace_id")},
    jobs=before["jobs"])
assert coverage == read("capture-coverage.json")
assert coverage["status"] == "bounded_text_receipts_matched"
assert coverage["covered_messages"] == report["covered_messages"] == 2
assert len(pending) == report["queued_entries"] == 3
assert report["outage_verified"] == {"descriptor_absent": True, "former_port_closed": True}
assert report["repeat_replay_and_restart_preserved"] and report["cleanup_established"]
assert report["embedding_responses_observed"] == 0 and not report["cleanup_errors"]
assert report["status"] == "native_service_outage_raw_capture_observed"
print(json.dumps({"status": "verified", "indexed_files": len(index), "native_messages": 2,
                  "queued_entries": 3, "facts": 0, "pending_jobs": len(final["jobs"])}))
