# Three-host continuity v2: preflight passed, native attempt failed

The single scheduled native attempt stopped at its first Codex phase on
September 20, 2026. Codex stored the synthetic project fact, consolidated it,
and received a completed job with a fact ID. The runner then rejected the
trace: `Native call lacks model correlation`. No Claude or Hermes model phase,
restart, cross-host recall, correction, or forgetting ran in this attempt.
Three-host continuity remains unproved. The failed result is preserved.

## Binding and observed behavior

- Protocol: [prospective v2](../../../plans/2026-09-20-three-host-continuity-protocol-v2.md).
- Runner: `6e6513c42c7c8418e9c21901edf7e29c7ecf37c2`.
- Installed product source: `f2f741443e60f838b392ae8ea1d1a0b4d8c6c1f2`.
- Wheel SHA-256: `8552345d2c17c4c037347961eb3043d13b20f790f54a9ccc86a096add6197b0f`.
- Codex 0.146.0; native turn metadata reports `gpt-5.6-sol`.
- Command requested `--disable code_mode_host`. This records a requested
  setting, not observed direct-tool mode.
- All five cognition modules disabled; real local Nomic embeddings.
- Native run: 07:29:54 to 07:31:48 UTC. Codex exited normally in 106.293
  seconds, below its 180-second deadline. Cleanup passed; copied credentials
  and the runtime descriptor were removed.

The selected native records show model-authored `custom_tool_call` entries
named `exec`, with provider IDs beginning `call_`. The three MCP completion
records have separate IDs beginning `exec-`. There are no matching direct
model MCP call IDs. The existing strict parser correctly preserves tool
completion evidence while setting `model_call_correlated` to false. Reading
JavaScript source or using timing adjacency does not supply that missing ID
link. Provider wire input was not intercepted.

The stored fact ID is `6a919167-454d-4f09-81af-92163d23ff53`; its source event
is `56c28a542ea0c20ec14d7d726d62580ab2628bda1fe0577bb4af9813a9fdaf7e`.
The final state, service dispatch records, completed job receipt, and native
completion records support the write. They do not satisfy the frozen native
model-correlation requirement.

The controller accepted trust for the fresh disposable project and six
generated hooks, granted the requested store and consolidation permissions,
and submitted `/exit` after the completed turn. Those controller actions
are setup and shutdown, not model-issued memory operations.

## Separate preflight and offline checks

The preflight passed before the native attempt. It discovered eight installed
Hermes memory tools, recorded matching module origins at entry and exit,
and established cleanup without model inference or service writes. Its
report embeds earlier prerequisite evidence; that embedded evidence is not
new preflight activity.

The archived offline JUnit report contains 141 passing cases across nine
test files. Black passed ten implementation/test files and strict Python
3.10 mypy passed five benchmark modules in the preparation checkpoint.
These are benchmark checks, not a new product package validation count.
The prior product checkpoint remains 1,547 source passes with four skips,
650 isolated installed passes with four matching skips, and seven replay
assertions.

The prerequisite [Hermes v10 cohort](../hermes-native-cohort-v10/README.md)
retains one complete lifecycle pass and two failures. Its isolated patched
host result is unchanged by this three-host failure.

## Archive and next step

`evidence-index.json` binds 99 selected files by size and SHA-256. Reports,
service records, preflight traces, model readiness, and JUnit bytes are
preserved. Runner input copies match the recorded runtime hashes and the
committed inputs after CRLF/LF normalization. The README and index itself
are outside that file index.

The native export includes all response tool-call/output and MCP-completion
lines, preserving their original line bytes. `native-selection.json` maps
them to original source line numbers and hashes. The normalized evidence
copy explicitly removes full recorded host context and records its original
file hash and size. Unrelated instructions, account metadata, credentials,
host homes, and the database binary are not exported. This is a selected
evidence archive, not a complete native session or provider request capture.

Next, establish a supported direct-tool invocation or an explicit host
correlation record, validate that evidence path offline, and freeze a new
prospective attempt before running it. Do not reinterpret this attempt as
passed or relax correlation to accept matching prose, source text, or timing.
