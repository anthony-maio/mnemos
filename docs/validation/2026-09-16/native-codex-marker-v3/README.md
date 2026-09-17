# Native Codex marker probe v3

This disposable probe tested the installed Windows Codex CLI 0.146.0 with a
user-level `CODEX_HOME/hooks.json` containing one `SessionStart` command hook.
The command was a native `cmd.exe /d /c` invocation of a temporary `.cmd`
file. The file wrote two local markers and returned
`{"continue":false,"stopReason":"marker probe"}`.

The command reached the temporary hook: `codex-hook-event.txt` was observed.
The expected stop was not observed. The sanitized Codex stream contains
`thread.started`, `turn.started`, and `turn.completed`. The runner's
`model_usage_observed` field was only a heuristic based on those event shapes;
numeric usage was discarded, so model execution is recorded as NOT VERIFIED.
`marker_seen` was sampled during the safety loop before the launcher was
killed. `event_marker_seen` was checked after child output collection. Those
sampling points do not establish that one marker command ran while the other
did not.
The process was killed for safety after the hook marker check did not settle,
but the outer Windows launcher left a child running long enough to complete the
turn. This run therefore does not qualify as a no-model validation of
cancellation semantics.

The useful result is narrower. `hooks.json` discovery and native Windows
command dispatch reached the probe, so the previous absence of Mnemos Python
callbacks cannot be attributed only to Codex never discovering user hooks.
The stop-output path and the Python adapter command remain unproven. The
continued turn lifecycle also means this artifact must not be used as evidence of
safe fail-closed cancellation.

The tested shape follows the tagged Codex 0.146.0 discovery loader and
`SessionStart` output parser:

- [Hook discovery](https://raw.githubusercontent.com/openai/codex/rust-v0.146.0/codex-rs/hooks/src/engine/discovery.rs)
- [SessionStart output parsing](https://raw.githubusercontent.com/openai/codex/rust-v0.146.0/codex-rs/hooks/src/events/session_start.rs)

See [result.json](result.json) for the sanitized event summary. No auth data,
conversation text, or raw model output is stored here.

The generated fixture representations are preserved in
[generated_hooks.json](generated_hooks.json) and
[generated_probe.cmd.txt](generated_probe.cmd.txt); the disposable project
path is redacted. The runner removed the disposable root and its copied auth
file. A follow-up process check found zero matching non-PowerShell probe
processes and no disposable directories. The runner did not apply an explicit
ACL before copying the auth file, so this run is not evidence of ACL
enforcement.
