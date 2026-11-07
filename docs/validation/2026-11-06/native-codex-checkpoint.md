# Installed artifact and Codex checkpoint

The new installed artifact passed its CLI/MCP restart checks. Complete native
lifecycle acceptance is still open: all five Codex attempts stopped during the
first phase. V4 completed three MCP calls but failed trace normalization. V5
completed three MCP calls but also recorded a failed model script cell. None
completed the six-phase lifecycle. The older Claude authentication failure
remains preserved.

## Installed artifact

Product source `6d46aa1afda6ab48d914caf0a51b558bc4733438` built source and wheel
artifacts with isolated Hatchling 1.33.0. Both passed Twine. The wheel hash is
`e0874f18f3df5b9605bca19ba9bf0553ce56ddf8fbaba946fee2bc09e07b72ce`.

The [installed smoke receipt](installed-smoke-6d46aa1a.json) records a fresh
installation outside the checkout, all five experimental module defaults off,
real FastEmbed recall across service process restarts, CLI/MCP access to the
same fact ID and evidence, a negative foreign-project query, package resources
and entrypoints, and service cleanup. MCP was 1.30.0 and FastEmbed was 0.9.1.
Loading the Hermes entrypoint is not native Hermes acceptance.

## Attempt v1: older CLI

Runner `5741ffb1` used the installed npm Codex CLI 0.146.0 and `gpt-6-astra`
under the [v1 protocol](../../plans/2026-11-06-native-codex-explicit-v1.md).
The server rejected that CLI as too old for the model. The native stream ended
with `turn.failed`; no completed-turn usage receipt exists. Do not infer billing
from its absence. No memory tool ran.

Service cleanup, descriptor removal and copied-credential removal passed.
The normal credential file was byte-for-byte unchanged. The
[v1 receipt](native-codex-v1-receipt.json) identifies the failed attempt and its
verified private archive of 149 files. No normal CLI installation was changed.

## Attempt v2: app-bundled runtime

Runner `28e4ec92` used the already installed desktop runtime,
`codex-cli 0.164.0-alpha.11.1`, with its executable hash pinned by the
[runtime amendment](../../plans/2026-11-06-native-codex-runtime-v2.md).
This run authenticated and the native session recorded `gpt-6-astra`.
It returned a completed model turn, but no memory operation completed.

The model attempted tool discovery through `exec`. The host reported that Code
Mode was unavailable because the harness disabled its host. The model then
reported that it could not store the fact or verify completion. The native
usage receipt reports 33,457 input tokens, 24,448 cached input tokens, 154 output
tokens and 45 reasoning output tokens; these are reported fields, not a billing
statement.

The acceptance harness also rejected the prompt observation because it expected
the older `event_msg/user_message` record. This runtime records user messages as
`response_item` entries with per-content provenance. Its injected project
instructions and environment context are separate from the actual `user.text`
submission. The new parser checks those provenance fields and now recovers
exactly one submitted prompt with the original hash from the saved trace.
Unknown, mismatched or ambiguous provenance fails validation; an additional
user prompt remains visible and invalidates single-prompt acceptance.

That offline parser repair does not change v2's failed status. The run still
lacks completed native memory calls. The [v2 receipt](native-codex-v2-receipt.json)
records the failure and verified private archive of 144 files. Service,
descriptor and credential cleanup passed, and the normal credential file's
bytes remained unchanged. The archives retain raw session evidence privately;
no live memory database or host configuration was changed.

## Attempt v3: Code Mode and recorded dispatch

Runner `9dc0aa91` enabled the pinned Code Mode host and its private built-in
rollout trace. The model requested store and consolidate through Code Mode.
Both were rejected before dispatch because the host required MCP approval
while the phase's interactive approval policy was `never`. No memory operation
completed. The old session normalizer then rejected the missing MCP records;
those records now live in the separate native rollout trace.

The [v3 receipt](native-codex-v3-receipt.json) binds the failed attempt to a
verified private archive of 177 files, including the frozen runner inputs.
Reported usage was 85,213 input tokens, 75,264 cached input tokens, 266 output
tokens and 31 reasoning output tokens. These are host-reported fields, not a
billing statement. The owned host and service processes stopped, the runtime
descriptor and copied authentication were removed, and the normal credential
file's bytes remained unchanged.

## Attempt v4: successful dispatch, unrecognized correlation event

Runner `084ce3f7` used the documented per-tool permissions only for the current
phase. The native stream recorded completed store, consolidate and job-status
calls. The job-status result reported a completed remember job with a fact ID.
The parser then rejected `mcp_tool_call_correlation_assigned`, an event emitted
on successful dispatch that was absent from the pre-dispatch failure in v3.
No later lifecycle phase ran.

The [v4 receipt](native-codex-v4-receipt.json) binds the failed attempt to 188
verified private archive members. Service/process, descriptor and copied-auth
cleanup passed; normal authentication bytes were unchanged. Usage is retained
in the receipt without a billing claim.

The [v5 amendment](../../plans/2026-11-06-native-codex-dispatch-v5.md) requires
each dispatch assignment to match the native runtime call and carry a unique
MCP ID within that call's recorded interval. The repaired adapter now parses
all three saved v4 calls with native model correlation and hashes 41 evidence
files. This offline check does not relabel v4 or complete its missing phases.
All 118 focused checks passed after the new positive and missing-correlation
regressions failed before repair.

## Attempt v5: completed memory calls with a failed model script

Runner `27acb835` correctly parsed successful dispatch events. The native model
stored the fact, consolidated maintenance and observed its remember job as
completed. Its consolidation script also tried to save `data.job_id` from the
maintenance response. That field does not exist; the Code Mode cell failed
while trying to store an undefined value. The model recovered by checking the
job ID from the original store response, but the protocol requires completed
cells and the attempt failed that gate. No later lifecycle phase ran.

The [v5 receipt](native-codex-v5-receipt.json) records the failed attempt and
189 verified private archive members. Cleanup passed, owned host and service
processes stopped, and normal authentication bytes were unchanged. The failed
cell remains a failure; no acceptance rule was relaxed to admit it.

The production MCP descriptions previously omitted consolidation's response
contract and the correction workflow. They now state which call returns the
job ID, distinguish committed events from completed processing, explain that
consolidation returns aggregate counts, and require checking the original job
and inspecting a corrected fact. Both operating references carry the same
guidance. This documentation repair needs a new installed artifact and native
attempt before any claim that it improves host behavior.

The guidance change passed 24 existing MCP runtime, annotation, scope, review,
compatibility and provenance checks. The live tool catalog returned all four
updated descriptions without starting a memory service. Formatting, strict
mypy for the MCP server and the diff check passed. No new behavioral assertion
is inferred from these documentation checks.

## Validation and next work

The new runner first passed 86 focused checks for process isolation, timeouts,
failed-turn rejection, model/session/prompt identity, cleanup, trace normalization,
lifecycle contracts and installed artifacts. The prompt-provenance regression
failed before repair; all 88 checks then passed. Strict mypy and formatting
checks also passed. These are harness checks, not a native acceptance result or
a fresh full-suite run.

The [v4 amendment](../../plans/2026-11-06-native-codex-trace-v4.md) configures
only the authorized tools on the disposable server and validates the native
model-call, runtime-cell and MCP-call identities. It preserves the exact
single-prompt lifecycle checks, inspected versions, completed-job observations,
scope controls, original-source replay and cleanup requirements. V3 remains
failed regardless of subsequent parser repairs or native attempts.

Before v4, 114 focused tests passed, including 26 new Code Mode trace cases.
The permission regression failed before the per-tool configuration change.
The new adapter rejects the original v3 trace specifically at the failed native
MCP runtime. Strict mypy passed for all three helpers. The pinned executable
accepted the disposable permission configuration with `--strict-config` and
`--print-startup-config`, without starting an app server or using authentication.
These checks prepare a new attempt; they do not establish native acceptance.

The separate Claude validation login remains pending. Supported-answer quality,
fresh evaluation, extraction evaluation, native tokenizer compatibility and the
two-week/two-project pilot remain open. Nothing in this checkpoint establishes
publication or readiness to start that pilot.
