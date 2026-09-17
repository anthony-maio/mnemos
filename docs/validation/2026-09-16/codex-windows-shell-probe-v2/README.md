# Codex Windows shell boundary probe v2

This is the corrected wheel-isolated follow-up to v1. It used the disposable
`D:\\WindowsTemp\\wheel-smoke` interpreter with no inherited `PYTHONPATH`;
the result confirms `mnemos` loaded from that interpreter's site-packages.
The wheel was built from the recorded source checkpoint; all 48 packaged
Python files byte-match the build source tree. A separate comparison against
the frozen `d17dd24` checkout verifies all 55 packaged files after normalizing
only CRLF line endings to LF; installed files match the wheel bytes exactly.
See [package-comparison.json](package-comparison.json). The wheel and runner
SHA256 values are in [result.json](result.json), with the exact runner preserved
as [runner.py.txt](runner.py.txt).

The fixture started one authenticated loopback Mnemos service with SQLite,
simple deterministic embeddings, and `UnavailableLLMProvider`. It used a
registered temporary project and retained the fixture under
`D:\\WindowsTemp\\mnemos-codex-shell-probe-v2-*` for review. No Codex process,
model, credential, or live configuration was used.

An explicit seed was committed through the service and processed by its real
maintenance path. Two different synthetic `UserPromptSubmit` payloads then
used different sessions and turn IDs:

- direct `python -m mnemos.adapters.codex`
- the current `_codex_hook_command` passed through the tagged Windows raw
  `cmd.exe /d /c` command form

Both subprocesses exited zero and returned valid `hookSpecificOutput` containing
the consolidated seed fact. The runner asserted that the exact direct prompt
and exact raw-shell prompt were both durable before restart and both remained
durable after the service reopened against the retained SQLite database. All
three facts remained available after restart. All biomimetic modules were off,
so these prompt facts reflect the baseline storage path, not model extraction
quality. This isolates Python bootstrap
and Windows shell quoting from native Codex lifecycle dispatch.

The tagged runner constructs Windows commands with a raw quoted hook argument
when `/c` is present. The probe records that exact generated command line and
the source hashes in [result.json](result.json):

- [Codex command runner at rust-v0.146.0](https://raw.githubusercontent.com/openai/codex/rust-v0.146.0/codex-rs/hooks/src/engine/command_runner.rs)
- [Codex hook discovery at rust-v0.146.0](https://raw.githubusercontent.com/openai/codex/rust-v0.146.0/codex-rs/hooks/src/engine/discovery.rs)

The service fixture is retained intentionally. This evidence covers the
adapter subprocess and shell boundary only; it does not prove native Codex
hook dispatch or cancellation behavior.
