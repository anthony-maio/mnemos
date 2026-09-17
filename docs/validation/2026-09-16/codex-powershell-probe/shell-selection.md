# Codex 0.146.0 native hook shell selection

This note records the tagged Codex source behavior that matters for Mnemos
native hooks. It does not turn the failed v4 conversation into a shell proof.

## Finding

Codex does not guarantee that a hook command is run by `cmd.exe` on Windows.
In the `rust-v0.146.0` source, session hook construction receives the current
turn environment shell and converts it to `CommandShell` arguments. The
default Windows shell detector prefers PowerShell and falls back to `cmd.exe`
only when PowerShell cannot be found. An explicit turn environment can still
select `cmd`, `bash`, `sh`, or another supported shell.

The relevant tagged sources are:

- [`build_hooks_for_config`](https://raw.githubusercontent.com/openai/codex/rust-v0.146.0/codex-rs/core/src/session/mod.rs), around lines 3898-3926: the current environment shell is converted with `derive_exec_args` and passed to the hook engine.
- [`shell_detect.rs`](https://raw.githubusercontent.com/openai/codex/rust-v0.146.0/codex-rs/shell-command/src/shell_detect.rs), around lines 221-269: Windows `default_user_shell_from_path` asks for PowerShell first, then uses the `cmd.exe` ultimate fallback.
- [`shell.rs`](https://raw.githubusercontent.com/openai/codex/rust-v0.146.0/codex-rs/core/src/shell.rs), around lines 17-43: PowerShell receives `-NoProfile -Command`, while Cmd receives `/c`; Bash, Sh, and Zsh receive `-c` (or `-lc` when login mode is requested).
- [`command_runner.rs`](https://raw.githubusercontent.com/openai/codex/rust-v0.146.0/codex-rs/hooks/src/engine/command_runner.rs), around lines 155-189: the hook runner uses the configured shell and only applies Windows `raw_arg` quoting when the configured arguments contain `/c`. PowerShell therefore follows the ordinary `.arg(handler.command)` path.
- [`registry.rs`](https://raw.githubusercontent.com/openai/codex/rust-v0.146.0/codex-rs/hooks/src/registry.rs), around lines 28-37 and 63-73: `shell_program` and `shell_args` are explicit engine inputs.

The installed local CLI reports `codex-cli 0.146.0` and `hooks` as a stable
feature. Its feature listing exposes no Mnemos-relevant hook shell selector.
That listing does not reveal which shell a particular noninteractive turn
environment selected.

## What the experiments establish

The v4 report at
`docs/validation/2026-09-16/native-codex-capture-v4/report.json` records zero
wrapper invocations and zero hook episodes. The Codex process returned an
answer unrelated to the seeded SQLite fact. This establishes that the v4
capture and recall acceptance conditions were not observed. It does not
identify whether hooks were undiscovered, untrusted, disabled for that
session, or dispatched through an incompatible shell.

The earlier marker probe used an explicit `cmd.exe /d /c wrapper.cmd` path and
did dispatch. The offline v2 probe also passed the exact raw `cmd.exe /d /c`
shape through direct and raw-Cmd subprocesses. Those results show that the
Mnemos adapter and Cmd command shape work when Cmd is explicitly selected.
They do not show that native Codex selected Cmd.

The current host renderer in `mnemos/hosts.py` emits a quoted Windows Python
command. It does not currently emit a shell selector or a separate
`commandWindows` field. With Codex's default Windows PowerShell path, the
command is parsed by PowerShell using `-Command`; with Cmd it is parsed using
`/c`. The quoting that passed the raw-Cmd probe is therefore insufficient as a
portable native-host proof. A path with spaces and the inline `-c` bootstrap
also needs a shell-specific test before being treated as supported.

## Bounded next test

The next native test should use a disposable Codex home and a hook command
that writes a shell marker before invoking Mnemos, then inspect the marker and
the hook output. Keep the command registered in the actual discovered hook
file and do one short turn. The marker should record only a fixed shell label,
event name, exit code, and command digest. The result distinguishes:

1. no marker: discovery, trust, feature, or lifecycle dispatch remains the
   first blocker;
2. marker says PowerShell and adapter fails: Windows PowerShell quoting or
   bootstrap portability is the blocker;
3. marker says Cmd and adapter fails: the command or adapter itself is the
   blocker; and
4. marker and adapter succeed: the earlier v4 failure was configuration,
   trust, lifecycle timing, or another host setup condition.

Do not label native Codex capture supported from the v4 run. The safe product
contract is to render the host capability as pending until a real Codex turn
proves hook discovery, shell dispatch, adapter output, durable capture, and
retrieval separately.
