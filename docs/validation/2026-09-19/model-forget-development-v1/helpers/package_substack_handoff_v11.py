"""Export a working writing packet without claiming engineering completion."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

repo = Path(__file__).resolve().parent.parent
output = Path("D:/Development/mnemos/deliverables/mnemos-substack-research-packet-2026-09-19-v11.zip")
assert not output.exists()
selected = [
    "benchmarks/native_hosts/model_forget_development_v1.py",
    "benchmarks/native_hosts/hermes_trace_v2.py",
    "tests/test_model_forget_development.py",
    "tests/test_model_forget_runner.py",
    "docs/plans/2026-09-19-local-model-forget-development-v1.md",
    "docs/plans/2026-09-19-local-model-forget-profiles-v1.json",
    "docs/plans/2026-09-19-historical-assertion-contract-review.md",
    "mnemos/modules/sleep.py",
    "mnemos/memory_service.py",
    "benchmarks/local_memory/ollama_adapter.py",
    "mnemos/client.py",
    "mnemos/service.py",
    "mnemos/utils/storage.py",
    "tests/test_related_forget_service_errors.py",
    "mnemos/errors.py",
    "tests/test_hermes_related_forget.py",
    "tests/test_hermes_related_forget_scope.py",
    "mnemos/adapters/hermes_tools.py",
    "mnemos/adapters/hermes.py",
    "benchmarks/local_memory/module_mechanism.py",
    "benchmarks/local_memory/datasets/module-mechanism-v1.json",
    "docs/plans/2026-09-16-module-evaluation-protocol.md",
    "docs/validation/2026-09-19/affective-fidelity-review.md",
    "tests/test_service_fact_errors.py",
    "tests/test_bridge.py",
    "tests/test_assistant_forgotten_id_capture.py",
    "tests/test_forgotten_id_lifecycle.py",
    "docs/handoffs/substack-local-memory.md",
    "docs/plans/2026-09-19-hermes-native-lifecycle-protocol-v2.md",
    "docs/plans/2026-09-19-three-host-continuity-protocol-v1.md",
    "docs/plans/2026-09-16-functional-local-memory.md",
    "docs/functional-local-memory.md",
    "docs/client-compatibility-matrix.md",
    "docs/integrations/hermes.md",
    "docs/integrations/hermes-cli-session-cwd.patch",
    "docs/paper-errata.md",
    "docs/validation/2026-09-16/implementation-status.md",
    "docs/validation/2026-09-16/progress.md",
    "docs/validation/2026-09-16/claim-evidence-ledger.md",
    "docs/validation/2026-09-16/evaluation-running.md",
    "docs/validation/2026-09-19/comparison-diagnostics.md",
    "docs/validation/2026-09-19/lifecycle-extraction-diagnosis.md",
    "docs/validation/2026-09-19/lifecycle-incomplete-local-v2/README.md",
    "docs/validation/2026-09-19/loopback-recovery-checkpoint-v1/README.md",
    "docs/validation/2026-09-19/retrieval-parity-checkpoint-v1/README.md",
    "tests/test_memory_service.py",
    "tests/test_service_job_errors.py",
    "tests/test_service_lost_ack.py",
    "tests/test_storage_retrieval_contract.py",
    "tests/test_client_stale_descriptor.py",
    "tests/test_service_competing_recovery.py",
    "tests/test_hermes_native_identity.py",
    "tests/test_hermes_adapter.py",
    "tests/test_hermes_tools.py",
    "tests/test_hermes_transcript_identity.py",
    "tests/test_hermes_provisional_capture.py",
    "tests/test_lifecycle_extraction_outcomes.py",
]
for folder in (
    "docs/validation/2026-09-19/job-not-found-checkpoint-v1",
    "docs/validation/2026-09-19/hermes-native-lifecycle-local-v2",
    "docs/validation/2026-09-19/encoding-terminal-local-v1",
    "docs/validation/2026-09-19/retrieval-terminal-local-v1",
    "docs/validation/2026-09-19/hermes-native-lifecycle-local-v1",
    "docs/validation/2026-09-19/hermes-native-discovery-preflight-v3",
    "docs/validation/2026-09-19/three-host-runner-offline-v1",
    "docs/validation/2026-09-19/hermes-native-identity-checkpoint-v1",
    "docs/validation/2026-09-19/hermes-native-rows-contract-v3",
    "docs/validation/2026-09-19/hermes-native-discovery-preflight-v1",
    "docs/validation/2026-09-19/hermes-rewind-diagnostic-v2",
    "docs/validation/2026-09-19/correction-retry-checkpoint-v1",
    "docs/validation/2026-09-19/hermes-api-recovery-v1",
    "docs/validation/2026-09-19/graph-coverage-diagnosis-v1",
    "docs/validation/2026-09-19/graph-degree-development-v1",
    "docs/validation/2026-09-19/graph-degree-development-v2",
    "docs/validation/2026-09-19/hermes-lifecycle-checkpoints-v1",
    "docs/validation/2026-09-19/supplemental-launch-v2",
    "docs/validation/2026-09-19/performance-recovery-v1",
    "docs/validation/2026-09-19/hermes-safety-checkpoint-v1",
    "docs/validation/2026-09-19/native-hermes-local-v1",
    "docs/validation/2026-09-19/native-hermes-local-v3",
    "docs/validation/2026-09-19/native-hermes-local-v4",
    "docs/validation/2026-09-19/hermes-continuity-checkpoint-v1",
):
    base = repo / folder
    assert base.is_dir(), folder
    selected.extend(str(p.relative_to(repo)).replace("\\", "/") for p in base.rglob("*")
                    if p.is_file() and p.suffix in {".md", ".json", ".patch"})

for folder in (
    "docs/validation/2026-09-19/encoding-terminal-local-v1",
    "docs/validation/2026-09-19/retrieval-terminal-local-v1",
    "docs/validation/2026-09-19/hermes-native-lifecycle-local-v1",
    "docs/validation/2026-09-19/hermes-native-lifecycle-local-v2",
):
    selected.extend(p.relative_to(repo).as_posix() for p in (repo/folder).rglob("*")
                    if p.is_file() and p.suffix in {".zip", ".jsonl", ".log", ".txt"})

for folder in ("model-forget-development-v1", "local-model-readiness-v1", "related-forget-error-checkpoint-v1", "hermes-native-cohort-v6", "hermes-native-discovery-preflight-v5", "hermes-reviewed-forget-checkpoint-v1", "hermes-native-cohort-v5", "mechanism-local-v1", "native-repairs-checkpoint-v1", "affective-guidance-checkpoint-v1", "affective-response-contract-v1", "hermes-native-cohort-v4", "hermes-native-cohort-v3", "writer-primary-sources-v1", "native-model-profile-v1", "hermes-native-discovery-preflight-v4"):
    base = repo / "docs/validation/2026-09-19" / folder
    assert base.is_dir(), folder
    selected.extend(p.relative_to(repo).as_posix() for p in base.rglob("*") if p.is_file())
selected.extend(["docs/plans/2026-09-19-hermes-native-lifecycle-protocol-v6.md", "docs/plans/2026-09-19-hermes-native-lifecycle-protocol-v5.md", "docs/plans/2026-09-19-hermes-native-lifecycle-protocol-v3.md", "docs/plans/2026-09-19-hermes-native-lifecycle-protocol-v4.md"])

members = {}
for relative in sorted(set(selected)):
    members["repository/" + relative] = (repo / relative).read_bytes()
voice = Path("C:/Users/Ant/.agents/skills/writing-in-anthonys-voice/SKILL.md")
members["writing-in-anthonys-voice/SKILL.md"] = voice.read_bytes()
handoff = "repository/docs/handoffs/substack-local-memory.md"
members[handoff] = members[handoff].replace(
    b"C:/Users/Ant/.agents/skills/writing-in-anthonys-voice/SKILL.md",
    b"../../../writing-in-anthonys-voice/SKILL.md",
)
if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=repo, text=True).strip():
    raise ValueError("Commit the final writing snapshot before packaging")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
members["START_HERE.md"] = f"""# Mnemos Substack research packet

Working snapshot, September 19, 2026. Start with
[the writing handoff](repository/docs/handoffs/substack-local-memory.md), then
read [Anthony's voice skill](writing-in-anthonys-voice/SKILL.md).

Assignment: a roughly 2,500-word Substack draft, three title options, and a
separate claim/source note. Do not publish. Engineering remains in progress;
this packet does not establish final acceptance or daily-use success.

Repository snapshot: `{commit}`. This packet contains selected research
documents, reports, and regression examples. It includes the compressed terminal
encoding/retrieval reports, all four fixed-cohort Hermes runs with every scheduled
attempt retained, the reviewed-copy tool and source-selection error checkpoints,
affective fidelity review, earlier
native failure traces, the six-case local model development comparison,
model readiness records, and dated website HTML and paper PDF snapshots. Some older
archives remain omitted. Follow source links in the checkout when deeper
evidence is needed; ask for the cited artifact if it is absent from this packet.
The original [website](https://mnemos.making-minds.ai/) and
[paper](https://making-minds.ai/papers/mnemos.pdf) remain primary research inputs.
Read the included dated copies and verify public sources before publication.
These current snapshots do not establish the sources original audit state.

Check `manifest.json` for included files and hashes. Some relative links point
to materials in the full repository. Only the voice-skill link in the copied
handoff has been adjusted for portability; the repository handoff is unchanged.
""".encode()
manifest = {"source_commit": commit, "status": "working_snapshot",
            "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                      for name, data in sorted(members.items())}}
members["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
output.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for name, data in members.items():
        archive.writestr(name, data)
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
    assert set(archive.namelist()) == set(members)
    for name, data in members.items():
        assert archive.read(name) == data
print(json.dumps({"path": str(output), "bytes": output.stat().st_size,
                  "files": len(members), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))
