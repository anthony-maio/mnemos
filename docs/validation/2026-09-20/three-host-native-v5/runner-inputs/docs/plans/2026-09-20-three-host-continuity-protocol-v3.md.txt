# Three-host native continuity v3: explicit Codex model binding

Status: prospective; NOT RUN. Commit the protocol, runner, diagnostic, and
dependencies before the separate import-only preflight and one native run.

The [v2 attempt](../validation/2026-09-20/three-host-native-v2/README.md)
remains failed at its first Codex phase. Its successful store and completed
job do not meet the missing model-to-MCP correlation requirement. No later
host phase ran. Do not retry or reinterpret that attempt.

## One changed host profile

The [read-only catalog diagnostic](../validation/2026-09-20/codex-tool-mode-diagnostic-v1/README.md)
found `tool_mode: code_mode_only` for the observed default `gpt-5.6-sol`,
while `gpt-5.5` has a null tool mode. All four feature flags were false under
the requested disable flag. Catalog metadata alone is not native evidence.

Pin Codex CLI 0.146.0 to `--model gpt-5.5 --disable code_mode_host` in every
fresh disposable phase. Require at least one native turn-context model
record and reject any record whose model differs. Record the requested and
observed models in process evidence. Keep the existing 180-second phase
deadline, terminal invocation, read-only sandbox, on-request approvals,
private copied authentication, and verified cleanup.

Require direct model `function_call` records linked by exact IDs and
arguments to successful MCP completion records. Neither model selection,
feature flags, catalog metadata, matching JavaScript, nor timing may replace
this evidence. A repeated wrapper path stops the attempt as a failure.

## Unchanged contracts and schedule

All [v2 contracts](2026-09-20-three-host-continuity-protocol-v2.md) and the
original v1 phase order remain in force. In particular:

- Product source `f2f741443e60f838b392ae8ea1d1a0b4d8c6c1f2`, wheel SHA-256
  `8552345d2c17c4c037347961eb3043d13b20f790f54a9ccc86a096add6197b0f`.
- Hermes v10 attempt 3 prerequisite, with its two cohort failures preserved;
  isolated patched Hermes and unchanged dependency inventory.
- Local Ollama 65,536-context Qwen alias, effective-context and GPU checks,
  real Nomic embeddings, all five cognition modules disabled.
- Claude Code 2.1.191, Sonnet 4.6, existing $0.50 and 180-second phase caps.
- Independent query/result matching, structured scope exclusion, restart
  boundaries, correction version checks, full reviewed selection before
  deletion, source replay suppression, and unrelated retained controls.
- Separate no-inference import preflight, then exactly one fresh native
  attempt. Stop at the first failed requirement; no selective retries.

Use profile ID `native-three-host-v3` and fresh `mnemos-three-host-preflight-v3`
and `mnemos-three-host-native-v3` outputs. Bind v1, v2, and v3 protocols and
the selected catalog diagnostic in the runner's committed inputs.

Success remains `bounded_lifecycle_observed`. Complete capture, provider
wire inspection, Desktop, real-model module quality, and the operational
pilot remain separate open gates. This profile change is not a comparison
of model quality or evidence that the Mnemos product changed.
