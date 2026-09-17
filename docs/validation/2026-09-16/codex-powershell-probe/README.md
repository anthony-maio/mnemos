# Offline Codex PowerShell hook probe

This is a shell parser reproduction. It does not run Codex, call a model, read
authentication, or alter live configuration. The runner used the existing
wheel-smoke environment and a temporary protected fixture whose path is in
`result.json`.

The selected executable was `C:\\Program Files\\PowerShell\\7\\pwsh.exe`,
matching the `pwsh`-before-`powershell` lookup order in the tagged Codex
detector. The wheel was `mnemos-memory 0.7.0`, built from the recorded d17
checkpoint, with SHA-256
`a078ce7392bd8fc6181148cf270928805b23a191cc710b5da716fb2291b88614`.

The exact command returned by Mnemos' current `_codex_hook_command` failed
under `pwsh -NoProfile -Command` with exit code 1 and a PowerShell parser error
at `-c`. Prefixing that same command with PowerShell's call operator, `&`,
returned valid hook JSON with the seeded local-memory context and exit code 0.
The corrected invocation captured its distinct synthetic prompt as a durable
fact, and the fact survived a SQLite service restart. The malformed invocation
did not create a fact. The project and fixture paths included spaces, so the
allowlisted bootstrap values were also exercised through quoted path values.

This isolates a real PowerShell command-rendering defect in the current
generated command. It does not prove that native Codex reaches the hook, and
it does not prove that adding `&` is portable to Cmd, Bash, or other host shell
choices. The native v4 report still remains `not_observed` because it had zero
wrapper invocations. A future native turn must separately prove discovery,
trust, shell dispatch, adapter output, capture, and retrieval.

Evidence: [result.json](result.json) and the retained exact runner
[runner.py.txt](runner.py.txt).

Tagged source references:

- [`shell_detect.rs`](https://raw.githubusercontent.com/openai/codex/rust-v0.146.0/codex-rs/shell-command/src/shell_detect.rs)
- [`shell.rs`](https://raw.githubusercontent.com/openai/codex/rust-v0.146.0/codex-rs/core/src/shell.rs)
- [`command_runner.rs`](https://raw.githubusercontent.com/openai/codex/rust-v0.146.0/codex-rs/hooks/src/engine/command_runner.rs)
