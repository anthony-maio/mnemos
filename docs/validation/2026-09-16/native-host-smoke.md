# Native host smoke

Date: 2026-09-16

This record contains the initial failed model attempt and one bounded retry
using a disposable SQLite store,
scope registry, runtime descriptor, project directory, settings file, and
checkpoint directory. The temporary fixture was removed after the run. No
live host configuration or persistent memory data was used.

Installed host versions were read without changing them:

- Codex CLI: `0.146.0`
- Claude Code: `2.1.191`
- Hermes: no Windows executable found; the previously recorded WSL package
  remains a separate adapter check

Both runs used Claude's `--settings` file to install SessionStart,
UserPromptSubmit, and Stop command hooks for that invocation only. They also
used `--no-session-persistence`, an isolated project, no tools, plan mode, and
a 0.05 dollar maximum budget. The fixture seeded one scoped memory through the
local service before starting Claude.

The first native hook path executed and created the disposable session
checkpoint file, but the conversation stopped before model generation. The
Claude API returned HTTP 400 with this installed-version requirement:

```text
Claude Code 2.1.191 does not support this model; version 2.1.251 or newer is required.
```

That process returned code 1, with zero cost recorded. The seeded event was
committed, but the post-run list contained no additional durable fact from the
failed conversation.

The retry selected the explicit `haiku` alias, which the host reported as
`claude-sonnet-4-6`. It completed one turn with return code 0 and recorded a
cost of `$0.0414423`. Claude returned exactly:

```text
The project uses SQLite for durable memory storage.
```

The stream output showed native SessionStart and Stop lifecycle events, and the
temporary adapter checkpoint was created. The scoped list after the run still
contained the seeded fact only. The fixture intentionally used
`UnavailableLLMProvider`, so hook capture could acknowledge events while no
new cognition-derived durable fact was expected. This proves one real host
recall path against a seeded fact, not extraction quality or complete
conversation capture.

Acceptance state:

- Native Claude hook invocation: observed, checkpoint created.
- Native Claude model conversation and seeded recall: OBSERVED on the explicit
  model retry.
- Native Codex conversation: NOT RUN.
- Native Hermes CLI conversation: NOT OBSERVED in the bounded WSL attempt. The
  installed provider API and explicit WSL-to-Windows bridge were exercised, but
  local model validation or timeout prevented a completed agent turn. See
  [native-hermes-smoke.md](native-hermes-smoke.md).
- Real model extraction quality: NOT RUN.

The next real-host gate needs a fresh disposable run that records scoped
capture, restart recovery, correction, forget, and outage buffering separately.
The installed Claude process also loaded other user-installed lifecycle hooks;
strict MCP isolation did not remove those host hooks. A production acceptance
run should isolate the host hook set as well and preserve only the Mnemos hook
events under test.

## Codex CLI attempt

Codex CLI 0.146.0 was invoked twice through `codex exec --json` with a
disposable `CODEX_HOME`. The runner copied only the existing authentication file
into that temporary home, generated and TOML-parsed the Mnemos hook config, and
used a read-only sandbox plus an isolated project. The generated config
contained `SessionStart`, `UserPromptSubmit`, `PreCompact`, `PostCompact`,
`Stop`, and `SessionEnd` hooks. The sanitized result is
[native-codex-smoke.json](native-codex-smoke.json).

Both model turns completed, but the Codex JSON event stream reported no
`hook_started` or `hook_completed` lifecycle events. The model answered that the
supplied memory context did not specify a SQLite convention, even though the
temporary store contained one seeded fact. The result therefore provides
config parsing and native CLI execution evidence, but no Codex hook capture or
recall evidence. The output also reported that the bypass trust option was
enabled and that helper binaries could not be created under the temporary
directory.

The installed CLI reports `hooks` as a stable feature and its `exec --help`
output includes both `--dangerously-bypass-hook-trust` and
`--ignore-user-config`. `codex doctor --summary --ascii` loaded the temporary
config successfully. The generated `SessionStart` matcher originally included
`fork`, but the current hooks contract lists only `startup`, `resume`, `clear`,
and `compact`; Mnemos now emits the documented matcher. That cleanup did not
change this run's result, so it is not the cause of the missing callbacks.

The bounded evidence narrows the failure to the host execution path. The
config parses and the model turn runs, but `codex exec --json` produced no hook
lifecycle markers and the Mnemos hook produced no callback or checkpoint. The
available output does not distinguish hooks being skipped by noninteractive
execution from hook events being omitted by the JSON stream. A follow-up must
run an interactive Codex session with a disposable home and trusted hook, or
use a later CLI version after confirming the same command hook fires. Until
then, Mnemos must report Codex native capture as unproven and retain its MCP or
manual fallback.

One interactive follow-up was then run with `codex -C <disposable-project>` and
the same temporary `CODEX_HOME`. It did not use `--ignore-user-config`. Codex
displayed its real hook review screen; `Trust all and continue` was selected.
The hook command used a valid TOML basic string with absolute Windows paths to
the Python interpreter and an independent marker script. After sandbox setup,
the marker file was still absent, so the session was stopped before model
generation. This removes the earlier test harness quoting mistake as an
explanation for the missing callback. It still cannot distinguish a host hook
dispatch problem from startup timing or Windows command execution.

The [official Codex hooks contract](https://learn.chatgpt.com/docs/hooks) says
inline `config.toml` hooks are supported and that bypassing hook trust runs
vetted hooks for one-off automation. This installed CLI run did not exercise
that path in `codex exec`; an interactive Codex session or a version-specific
trust flow remains required before claiming native Codex hook parity.

## Isolated Claude capture follow-up on September 17

A fresh native Claude 2.1.191 turn used the installed `b7a9d9d` wheel, a
private disposable fixture, `--setting-sources project`, explicit settings,
no tools, no MCP servers, and `claude-sonnet-4-6`. The command used
`--no-session-persistence` and an $0.08 budget. These invocation flags are
documented in the [Claude CLI reference](https://code.claude.com/docs/en/cli-reference).
The event stream recorded exactly one SessionStart, UserPromptSubmit, and
Stop invocation. The actual cost was $0.009033.

The user prompt was durably captured and survived service restart. The native
UserPromptSubmit hook supplied the seeded SQLite fact to Claude, which
answered with that convention. This supplies native prompt-capture and
seeded-recall evidence under the isolated hook set.

The Stop hook exposed a production adapter defect: its payload contained both
a nonexistent transcript path and a valid `last_assistant_message`. Reading
the transcript raised `FileNotFoundError` before the adapter reached the
last-message fallback. The host returned success while the adapter emitted
its error diagnostic; no assistant event was stored. This is partial capture,
not a successful complete conversation-capture check. The failed result is
preserved in [native-claude-capture-v2/report.json](native-claude-capture-v2/report.json).

The fixture used simple embeddings and unavailable cognition. Durable event
capture is separate from accepted-fact extraction, which this run did not
measure. Its private database and hook traces remain at
`D:\WindowsTemp\mnemos-native-claude-capture-o36uj8e0`.

## Codex native command marker follow-up

The installed Codex CLI 0.146.0 reached a native Windows command configured
through a disposable `hooks.json`. Its event marker was observed. This narrows
the earlier failures: the installed host can discover a user hook and dispatch
a command. It does not yet prove native Mnemos capture or recall.

The probe's cancellation controls were inconclusive. Killing the launcher left
a child running, and the retained stream records turn lifecycle events without
numeric usage. Actual model execution is NOT VERIFIED. The two marker checks
occurred at different times, so their differing observations do not establish
a shell failure. The copied authentication file and disposable project were
removed; no matching probe process remained. The run did not explicitly protect
the temporary directory with an ACL before copying authentication, and supplies
no ACL verification. See the [preserved marker evidence](native-codex-marker-v3/README.md).

## Claude missing-transcript repair checkpoint

Commit `d17dd24` allows Stop to capture `last_assistant_message` when the
transcript is absent or unreadable. It reports degraded transcript coverage,
retains fallback deduplication until transcript data arrives, and advances a
transcript checkpoint only after acknowledged capture. Rejected prompt,
transcript, and fallback captures report degraded status. Corrupt checkpoints
remain distinct from unavailable transcript files.

The focused adapter suite passes 27 tests. Full frozen-source verification is
running. The installed-wheel native repeat passed: Claude 2.1.191 invoked
SessionStart, UserPromptSubmit, and Stop; its user prompt and assistant reply
were both captured and survived service restart. UserPromptSubmit supplied the
SQLite seed and Claude answered with it. Stop emitted a degraded diagnostic for
the unavailable transcript while preserving the assistant fallback. The actual
`claude-sonnet-4-6` cost was $0.009036 under the same $0.08 bound.

The [native v3 report](native-claude-capture-v3/report.json) preserves the
result and exact hook payloads. Its installed package matches all 55 files in
the frozen `d17dd24` source after normalizing CRLF line endings, with exact
wheel-to-installed byte matches. The runner, comparison, and artifact hashes
are retained alongside the report. Its private fixture is
`D:\WindowsTemp\mnemos-native-claude-capture-xbngd1p9`; the service was stopped.
As before, simple embeddings and unavailable cognition establish capture and
seeded recall only. When the host
provides neither a transcript nor distinct event IDs, identical repeated
messages remain ambiguous; this repair does not establish complete transcript
coverage.

An offline real-service probe also reproduced an independent identity defect:
the same Claude session, native event ID, role, and content in two registered
projects generated the same adapter ID. Project A committed; project B was
rejected by the service's payload check. No cross-project read leak or duplicate
was observed. Checkpoint filenames also omit scope. The [reproduction and repair
design](claude-scope-collision/repair-design.md) retain this open issue. A repair
must preserve forgotten-source tombstones when IDs change; canonical fact text
alone does not suppress every replay of its original conversation.

The [corrected offline Codex shell probe](codex-windows-shell-probe-v2/README.md)
also used the installed `d17dd24` package. Direct Python and the tagged Windows
shell command each captured a distinct prompt and returned seeded context;
both events survived service restart. This supplies isolated adapter and shell
transport evidence. It does not launch a native Codex conversation.
