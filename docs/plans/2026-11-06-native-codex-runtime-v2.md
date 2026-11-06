# Native Codex runtime amendment v2

The first Codex attempt at runner `5741ffb1` remains failed. CLI 0.146.0 exited
before memory tools when the server rejected it as too old for `gpt-6-astra`.
The raw stream contains `turn.failed` and no completed-turn usage receipt.
Do not infer billing from the absence of that receipt. Service and credential
cleanup passed; the normal credential file's bytes were unchanged.

A newer runtime is already installed with the desktop app. This attempt uses
that existing executable without replacing the normal CLI or app:

- Version: `codex-cli 0.164.0-alpha.11.1`.
- Binary SHA-256:
  `1103e9e09c71fa2cd0afd6d8f3b9a2d63174fe8efd7cb5bd09f1cb14ef58a098`.
- Executable location is recorded privately by the phase command receipt.

Reject any version or binary mismatch before inference. This is acceptance for
the named app-bundled alpha runtime, not a claim about all Codex versions or the
older installed npm CLI. The host's own model observation must still match the
pinned model; installed-version evidence alone does not establish compatibility.

Keep the product source, wheel, model, high reasoning effort, tool allowlists,
six single-prompt phases, 180-second deadlines, lifecycle checks and cleanup
requirements from [v1](2026-11-06-native-codex-explicit-v1.md). Use a new output
directory. Close stdin explicitly so the exec command cannot append inherited
input, and disable shell snapshots along with shell tools. These are process
isolation changes; the synthetic phase prompts remain unchanged.

Freeze the amendment and helpers before execution. Preserve both attempts,
including another failure if one occurs. No retry or corrective prompt is
permitted within either attempt. Normal credentials and live memory remain
outside the disposable configuration. The pilot is still unstarted.
