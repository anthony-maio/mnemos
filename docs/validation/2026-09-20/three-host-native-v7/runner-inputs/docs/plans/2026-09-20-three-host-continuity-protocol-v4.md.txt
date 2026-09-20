# Three-host native continuity v4: acknowledge the model migration notice

Status: prospective; NOT RUN. Commit the protocol and all runner inputs
before the separate preflight and one fresh native attempt.

The [v3 attempt](../validation/2026-09-20/three-host-native-v3/README.md)
remains failed. Its intended existing-model selection did not take effect,
both turns used `gpt-5.6-sol`, and the model gate correctly rejected them.
The controller also submitted `/exit` as ordinary conversation. Raw captures
committed, but the explicit teaching calls were cancelled and later hosts
did not run. Preserve that evidence and the earlier v2 failure.

## Setup change only

Retain `--model gpt-5.5 --disable code_mode_host` and add the command-scoped
Codex override:

```toml
notice.model_migrations = { "gpt-5.5" = "gpt-5.6-sol" }
```

This acknowledges the proposed migration notice. It does not select the
target model. The version-matched
[startup implementation](https://github.com/openai/codex/blob/rust-v0.146.0/codex-rs/tui/src/app/startup_prompts.rs)
uses this map to suppress that dialog; the same acknowledgment is written
when keeping the existing model. This affects only the disposable invocation.
Do not alter live configuration or suppress trust, hook, or tool approvals.

Require at least one native turn-context model observation, require every
observation to equal `gpt-5.5`, and retain exact model-to-MCP call ID and
argument correlation. Record the acknowledged migration pair in process
evidence. If the wrong model appears or a migration dialog still appears,
stop the attempt without accepting a lifecycle result.

Controller setup may accept the disposable project, verified generated
hooks, and only the requested memory operations. After each completed model
turn and hook completion, shut down with Ctrl+C twice; never submit `/exit`
as conversation. Any additional controller-authored conversation invalidates
the attempt and must be preserved.

## Unchanged schedule and limits

All [v3 bindings](2026-09-20-three-host-continuity-protocol-v3.md), inherited
v2/v1 memory operations, frozen prompts, phase order, controls, independent
queries, reviewed forgetting, source replay, model/context checks, and
resource limits remain unchanged.

Product source `f2f741443e60f838b392ae8ea1d1a0b4d8c6c1f2` and wheel SHA-256
`8552345d2c17c4c037347961eb3043d13b20f790f54a9ccc86a096add6197b0f`
remain fixed. Keep all five cognition modules disabled and the local Ollama
profile unchanged. This is a setup repair, not a model-quality comparison.

Use profile `native-three-host-v4`, fresh outputs
`mnemos-three-host-preflight-v4` and `mnemos-three-host-native-v4`, one
no-inference import preflight, and exactly one complete native attempt.
Stop dependent phases at the first failure. No selective retries.
Bind v1-v4 protocols and the catalog diagnostic as committed runner inputs.
Success remains `bounded_lifecycle_observed`; broader acceptance stays separate.
