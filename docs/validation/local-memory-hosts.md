# Local memory host validation

The [selection-rejection package](2026-09-20/selection-rejection-checkpoint-v1/README.md)
at `5155ebd` passes 1,638 source tests with four skips, all six source/package
checks, 655 isolated installed cases with four matching skips, and seven
replay assertions. Out-of-preview selected fact/source IDs receive a fixed,
noncommitting rejection while scope, digest, version, and deletion rules stay
unchanged. No native model has run this package. The failed v11 cohort below
belongs to preceding source `697b672`; its outcomes remain unchanged.

The [native Hermes v11 cohort](2026-09-20/hermes-native-cohort-v11/README.md)
finished zero of three complete lifecycles on `697b672`. Every attempt passed
teaching, restarted recall, correction, and corrected recall, then failed at
forgetting. Attempt 1 deleted the target and two selected copies but retained
two user captures with the labels. Attempt 2 supplied out-of-preview source
IDs and received HTTP 500. Attempt 3 obtained a preview without applying it.
All 50 model responses had verified 65,536-token context and positive GPU
residency; cleanup and independently inspected control state passed for all
three attempts. No source replay or final recall ran.

The Hermes-only setup does not install shared Codex AGENTS guidance, so this
cohort cannot measure that repair's effect. The planned three-host v6 run
remains NOT RUN because this package has no passing native prerequisite.
Earlier v10 remains one pass and two failures on its own package. Complete
capture, full-module quality, Desktop, and the operational pilot remain open.

The [three-host v5 attempt](2026-09-20/three-host-native-v5/README.md) passed
both teaching phases, then failed at Hermes recall on unsupported scope
arguments. Query-only retries retrieved both facts but do not clear the
failure. Both service starts reported a running worker; native correction
was not reached. Generated project guidance explicitly requests the rejected
arguments and appears in all three Hermes provider requests. That instruction
conflict is repaired in source `697b672`; native confirmation remains open.
Cleanup passed; full continuity remains open.

The [three-host v4 attempt](2026-09-20/three-host-native-v4/README.md) passed
Codex teaching, Claude teaching, and Hermes retrieval of both facts after a
service restart. Hermes correction then failed: an unsupported scope argument
was rejected, required inspection was omitted, and the committed correction
job remained pending. The final fact stayed at version 1. Later phases did
not run. Cleanup passed. The runner processes jobs after a successful native
phase; validation stopped before that maintenance step. Pending status in
this run does not establish a production worker defect.

The [v3 model-substitution failure](2026-09-20/three-host-native-v3/README.md) and
[v2 correlation failure](2026-09-20/three-host-native-v2/README.md) remain failed.
Full three-host continuity remains open. Hermes v10 separately retains one
complete bounded lifecycle pass and two failures.

Updated September 20, 2026 for the unreleased functional-local-memory worktree. This record separates adapter and transport checks from native host acceptance. The [compatibility matrix](../client-compatibility-matrix.md) records observed versions and source bindings.

## Current native evidence

| Host or workflow | Observed result | Boundary |
| --- | --- | --- |
| Codex CLI 0.146.0 and Claude Code 2.1.191 | [Shared lifecycle v9](2026-09-18/native-codex-claude-continuity-v9/README.md), installed `5af3d15`: 15 bounded criteria pass for teaching, corrected recall, reviewed deletion, replay suppression, retained facts, and foreign-project exclusion across restarts. | All three Codex processes exited zero, but two Stop hooks timed out. Fixture vectors and disabled cognition modules do not prove model quality or complete capture. |
| Concurrent Codex/Claude recovery | [Mixed-host v4](2026-09-18/native-crosshost-recovery-v4/README.md), installed `fc72cc2`: three same-event replay pairs committed to stable event/job identities; eight normal messages and two non-extractable provisional records survived owner restart. | One pairing timeout and degraded Claude identity diagnostics remain. The test instruments overlap and explicitly restores a fixture owner. |
| Claude Code lifecycle | [Native v18](2026-09-17/native-claude-lifecycle-v18/README.md), installed `dedfcfd`: 15 fact-lifecycle criteria pass, including correction, forgetting, restarts, and reconstructed replay. | Overall partial because capture diagnostics were degraded. An observed connection abort remains unexplained. |
| Codex Desktop | No isolated native conversation run. | NOT RUN; shared adapter tests are insufficient. |
| Hermes in WSL, isolated source `14a3463` with cwd-only patch | [Native v10](2026-09-20/hermes-native-cohort-v10/README.md), installed `f2f7414`, completed one of three full lifecycles. Attempt 3 passed correction, reviewed deletion, replay suppression, and restarted final recall. All 40 serving-context checks and three cleanup checks passed. | Attempts 1 and 2 failed at omitted recall and incomplete candidate inspection respectively. All five cognition modules were disabled. No stale-preview rejection occurred. Three-host continuity, complete capture, and daily-use readiness remain open; this is isolated patched-host evidence. |

The Hermes installation used by the recorded checks is `/home/anthony/.hermes/hermes-agent` in WSL. Initial API evidence used revision `97962358`; later wheel-discovery evidence used `36842e639a178741b70c821e0f4cdd129cd0dfe3`. Those are observation bindings, not a live installation inventory. The Windows Mnemos environment does not claim Hermes as a native dependency. See the [bridge record](2026-09-16/hermes-bridge-smoke.md) and [host smoke history](2026-09-16/native-host-smoke.md).

Claude's observed prompt payload lacks an occurrence ID and its transcript appears later. Mnemos preserves the prompt as provisional evidence, then captures native UUID-backed messages separately. Provisional evidence remains excluded from extraction. The resulting identity warnings are recorded limitations, not grounds to merge identical text or suppress diagnostics.

## Synthetic cross-host contract

`tests/test_cross_host_contract.py` starts an authenticated loopback `MnemosService` with a temporary SQLite database and registry. It uses real `MnemosClient` HTTP transport with Codex, Claude Code, and Hermes scope contexts. The [scenario fixture](../../benchmarks/scenarios/local-memory-continuity.jsonl) rotates the teaching context across all three adapters.

The test checks source evidence, recall after restart, correction preserving the fact ID while advancing its version, positive local/foreign retrieval controls, and cross-project exclusion. Forgetting through the Hermes context must survive source replay, maintenance, and another restart. Cognition is synthetic and disabled; the test does not launch three native conversations or prove extraction quality.

```text
.venv\Scripts\python.exe -m pytest -q tests/test_cross_host_contract.py tests/test_mcp_compatibility.py
```

Other service tests cover authentication, origin rejection, body limits, ownership, startup, retry, and sanitized outbox replay. Adapter tests cover transcript deltas, partial lines, rotation, checkpoints, scope, stable identity, and native context output. These contract checks do not replace the native observations above.

## Disposable adapter checks

Use the adapter regression suite for repeatable payload checks in temporary fixtures:

```text
.venv\Scripts\python.exe -m pytest -q tests/test_codex_adapter.py tests/test_claude_adapter.py tests/test_claude_provisional_capture.py
```

For a manual check, prepare a disposable service, enabled project registry, profile, database, runtime descriptor, outbox, and Claude checkpoint directory. Bind both adapters to that fixture and restore temporary environment settings after both checks. To prevent service autostart, construct `MnemosClient(auto_start=False)` and inject it through the adapter's `client` argument; the command-hook entry points do not read an `MNEMOS_AUTO_START` switch. The automated tests supply these dependencies directly.

Hook exit code zero is not a capture acknowledgment. Inspect the fixture's source events or outbox and its diagnostics. Without a native Claude occurrence ID a synthetic prompt is provisional; a payload-only check does not establish native transcript capture or a complete conversation record.

For Hermes, use its WSL environment and the explicit bridge configuration described in the [integration guide](../integrations/hermes.md). Do not guess a WSL address or share SQLite as a competing writer across the boundary. Installing the provider into an isolated Hermes environment is distinct from enabling it in a live profile.

## Remaining gates

A native acceptance run must bind the host version, disposable or approved configuration, package/plugin source, transport, profile, canonical project identity, provider configuration, and observed operations. Capture, retrieval, restart, correction, forgetting, outage, and concurrency results must be identified individually. Preserve failed attempts and remove secrets from selected records.

The bounded Codex/Claude lifecycle and replay observations are complete as described above. Complete capture, native Hermes continuity, Codex Desktop, configured-provider recovery, extraction quality, and the daily-use pilot remain open. The latest local 10,000-fact measurement fails both recall and capture latency targets; see the [September 19 performance record](2026-09-19/performance-local-v1/README.md). The [evaluation status](2026-09-16/evaluation-running.md) tracks the separate frozen real-model run. Its process being live does not establish accepted quality results. The two-week internal pilot and external readiness evidence remain NOT RUN.
