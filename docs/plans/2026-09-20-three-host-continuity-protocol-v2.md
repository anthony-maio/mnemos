# Three-host native continuity v2: current package and reviewed evidence

Status: prospective; NOT RUN. Commit this protocol, the driver, and all
dependencies before the import-only preflight and the single native attempt.
The original [v1 protocol](2026-09-19-three-host-continuity-protocol-v1.md)
retains its source binding and has no native result. Its phase order,
synthetic scenario, independent queries, controls, replay requirements,
budgets, failure rules, and acceptance boundaries remain in force here.

## Current bindings and prerequisite

Use verified source `f2f741443e60f838b392ae8ea1d1a0b4d8c6c1f2` and wheel SHA-256
`8552345d2c17c4c037347961eb3043d13b20f790f54a9ccc86a096add6197b0f`.
The source/package and installed verification reports must pass, agree on
the wheel, and bind the actual installed interpreter and module bytes.
Keep all five cognition modules disabled and real Nomic query embeddings.
Automatic durable-fact extraction remains unavailable in this baseline.

The prerequisite is [Hermes v10 attempt 3](../validation/2026-09-20/hermes-native-cohort-v10/README.md),
one passing complete lifecycle in a fixed cohort containing two failures.
Require the exact passing report, executed input hashes, every phase gate,
and its terminal three-attempt cohort with established cleanup. Preserve
both failed attempts. This requirement establishes a bounded demonstration;
it does not reinterpret the cohort as a reliability pass.

Use isolated Hermes source `14a346345486418aa85ed6526549412ee2665ade` with the
unchanged cwd-only patch. Bootstrap v7 wraps the unchanged v6 invocation,
adding an exit-time module-origin check without changing calls, prompts,
budgets, or results. Reject untracked host source and compare entry/exit
module origins as well as the interpreter inventory. The live interpreter installation
is `ebe3fcb2f85ebdbe0e469e9d295636ac930246b4`; its sorted dependency inventory
must retain SHA-256 `a175bfc46955bc72ebb1c854ce7646f3dad3fd22716d3219d3a42d8928a30f7f`.
Verify source/import origins and inventory before and after execution. The
live CLI's process-directory restoration does not establish explicit cwd
forwarding to its memory provider. Do not modify that installation.

Hermes uses local Ollama at `http://127.0.0.1:11434`, model
`mnemos-native-qwen35-9b:greedy-ctx64k-v2`, digest
`87bc5156d86918e0435431ab0286167a4c79c4a4a259b82fba5f3f75022006e2`.
Preserve temperature 0, seed 42, presence penalty 0, top-k 20, top-p 0.95,
thinking disabled, and model-defined 65,536 context tokens. Use readiness v2
through the actual OpenAI-compatible endpoint. Check effective context and
positive GPU residency after every observed model response. Verify Nomic
digest `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.
The retired remote Ollama endpoint is not used.

Codex CLI 0.146.0 and Claude Code 2.1.191 retain their existing native cloud
invocations and disposable homes. Codex uses `--disable code_mode_host`;
direct tool behavior must be observed. Claude uses `claude-sonnet-4-6`,
with the existing per-phase $0.50 cap. None of the Hermes model or context
settings apply to those cloud hosts. Record actual executable versions and
hashes. Never archive copied authentication or runtime descriptor tokens.

## Validation repairs before execution

The driver remains `benchmarks/native_hosts/three_host_run.py`. It now uses
the v10 package/profile bindings, bootstrap v7's observation wrapper, readiness v2, and an explicit
`--preflight-only` mode. Preflight verifies imports and eight-tool discovery
without model inference or service writes. It has its own fresh output
directory and cannot be called a native lifecycle result.

Keep exact native query-to-result binding: Hermes must recall each convention
with its corresponding query. A single result containing both facts under
one query cannot satisfy both requirements. Do not add an unrelated exact
tool-count restriction to otherwise valid retrieval.

Reviewed forgetting must prove the target inspection, complete matching
preview, matching digest and preview limit, inspection of every selected
additional fact at the previewed version/content, and required source
selection. Selected sources must be eligible and may not retain unselected
derived facts. Require the committed receipt's exact selection and persisted
erasure. The forget prompt explicitly requests those inspections and the
matching preview limit; the controller must never fill them in. This makes
the existing reviewed-selection requirement executable before any result.

Hermes dispatch traces match tool names and arguments, while provider call
IDs are recorded at the model boundary. Reject ambiguous identical pending
calls. Corroborate each accepted result against the subsequent outgoing HTTP
tool message under its real provider call ID, including matching function
name and arguments. Verify archived body bytes/hashes and reject altered or
missing messages. Label this evidence precisely rather than implying the
dispatch callback independently supplies the provider call ID.

In addition to foreign-label exclusion, reject foreign fact IDs and scope
IDs in structured tool results. Keep unrelated project facts and the
independent review convention intact. No controller action may teach,
correct, retrieve in place of a host, or delete the target.

## Schedule and limits

Run one fresh complete three-host attempt after preflight and offline checks.
Stop dependent phases on the first failure. Do not retry failed phases or
substitute another host. Preserve raw synthetic evidence, failures, host
exit state, callback diagnostics, and authentication/process cleanup.

The v1 schedule remains: Codex teaches the release convention; Claude teaches
the review convention; Hermes recalls both and corrects the release fact;
all three recall the correction; Claude reviews and forgets selected release
copies; scripted original-source replay tests suppression; all three recall
again while retaining the review convention. Restart boundaries stay fixed.

Codex and Claude keep 180-second phases. Hermes keeps eight turns, a
180-second model budget, a 240-second outer deadline, and five-second kill
grace. Import-only preflight has a 60-second limit. Maintenance retains its
60-second observation and 45-second dispatch bounds. An uncertain WSL exit
is a cleanup failure, not permission to start another model run.

Success means `bounded_lifecycle_observed`. Report complete capture,
Codex/Claude provider-wire inspection, Desktop, real-model module quality,
and the two-week pilot separately. Earlier failures, T18 negative results,
and the working Substack packet are not superseded by a transport pass.
