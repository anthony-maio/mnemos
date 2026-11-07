# Native Codex Code Mode amendment v3

The v1 and v2 attempts remain failed. V2 authenticated and recorded the requested
model, but the harness had disabled the model's required Code Mode transport.
It also expected an older user-message event format. The latter is now parsed
from native per-content provenance and verified offline against v2's original
submitted-prompt hash. That repair does not relabel the v2 attempt.

## Runtime and recording

Keep the product, wheel, model and app-bundled Codex executable from the
[runtime amendment](2026-11-06-native-codex-runtime-v2.md). Enable
`features.code_mode_host` explicitly. Its sibling executable is also pinned:

- Component: `codex-code-mode-host.exe`.
- SHA-256: `2727abaa2b30eb4afbe64b1da51a26cb594dd18dda82c6292b5373b15a758257`.

Place that runtime directory on the child process's PATH only. Keep shell,
apps, additional agents, hooks, browser and other unrelated tools disabled,
retain the per-phase MCP tool allowlist, and close stdin. No installed runtime,
live configuration or normal credential is changed.

Enable the runtime's built-in local rollout trace with
`CODEX_ROLLOUT_TRACE_ROOT` pointing to a protected directory inside each fresh
phase. Keep those records private: they can include host context and model
requests. Record their byte hashes along with the ordinary native stdout and
retained session. This is additional observation of the installed host, not a
patched host implementation or a substitute tool runner.

## Acceptance boundary

Use a new output directory and commit the amendment and helpers before
inference. Retain the same six single-prompt phases, 180-second phase limits,
installed-artifact binding, job-completion observations, scoped identity/version
checks, evidence inspection, reviewed forgetting, original-source suppression
checks and cleanup from [v1](2026-11-06-native-codex-explicit-v1.md).

Code Mode can dispatch MCP calls inside a model-requested exec cell. Successful
MCP completion alone is still insufficient to establish the missing link to
that model request. Require exact host-recorded request/parent/call identities
and matching invocation/result data. Do not reconstruct that link from
JavaScript text, temporal adjacency or an assistant's final message.

The current strict normalizer remains the acceptance gate for this attempt.
If it cannot prove the required linkage, stop the attempt as failed and retain
the new local trace for offline adapter work. Do not bypass the gate to advance
to later lifecycle phases. Any subsequent parser change or native attempt must
be independently frozen; this attempt's original status and records remain.

The native trace must also prove the pinned model and exactly one user prompt,
distinguishing host-injected context through its native provenance metadata.
Unknown or ambiguous provenance fails acceptance. Cleanup includes the owned
service, native process tree, descriptor and copied authentication. The daily-use
pilot remains unstarted, and no release or live-installation claim follows.
