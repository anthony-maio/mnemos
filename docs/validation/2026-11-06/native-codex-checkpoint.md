# Installed artifact and Codex checkpoint

The new installed artifact passed its CLI/MCP restart checks. Complete native
lifecycle acceptance is still open: both Codex attempts stopped during the first
phase, before a memory tool completed. Neither is a successful teach, correct or
forget result. The older Claude authentication failure remains preserved.

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

## Validation and next work

The new runner first passed 86 focused checks for process isolation, timeouts,
failed-turn rejection, model/session/prompt identity, cleanup, trace normalization,
lifecycle contracts and installed artifacts. The prompt-provenance regression
failed before repair; all 88 checks then passed. Strict mypy and formatting
checks also passed. These are harness checks, not a native acceptance result or
a fresh full-suite run.

The next protocol must enable the transport required by the current native
model and verify the relationship between model requests and completed MCP
calls using recorded host evidence. Do not infer direct model-call correlation
from JavaScript text, timing or the assistant's final claim. Preserve the exact
single-prompt lifecycle checks, inspected versions, completed-job observations,
scope controls, original-source replay and cleanup requirements.

The separate Claude validation login remains pending. Supported-answer quality,
fresh evaluation, extraction evaluation, native tokenizer compatibility and the
two-week/two-project pilot remain open. Nothing in this checkpoint establishes
publication or readiness to start that pilot.
