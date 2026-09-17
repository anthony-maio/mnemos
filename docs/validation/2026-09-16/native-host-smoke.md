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

The focused adapter suite passes 27 tests. The first full run had one service
startup deadline failure; its isolated retry passed, and a second full run is
pending. See the [verification record](native-host-repair-verification.md).
The installed-wheel native repeat passed: Claude 2.1.191 invoked
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

## Direct executable native Codex capture follow-up

A fresh native CLI 0.146.0 run used the direct executable, an ACL-protected
temporary home, `hooks.json`, and the verified installed `d17dd24` wheel. It
completed in 30.693 seconds, below its 60-second limit, with exit code zero.
No wrapper trace, prompt event, assistant event, or supplied memory context
was observed. The model answered "The project uses PostgreSQL" although the
fixture held the SQLite seed. Native Mnemos capture and recall therefore
remain unproven.

Unlike the earlier marker run, this result retained numeric usage: 18,810
input tokens, including 10,624 cached input tokens, and 10 output tokens.
The stream did not identify the model. Three items labeled `error` contained
hook-trust and shortened-skill-description notices, rather than a useful hook
failure explanation. Global skills were still discovered despite the temporary
home, so this was not an empty host context.

The [v4 report](native-codex-capture-v4/report.json) preserves invocation,
package provenance, numeric usage, and the negative outcome. Its [reviewed
observations](native-codex-capture-v4/observations.json) retain the synthetic
answer and notices. Exact streams remain in the private fixture
`D:\WindowsTemp\mnemos-native-codex-capture-v4-czc0pgog`. The temporary auth
copy and service descriptor were removed, and no matching Codex, Python, or
cmd process remained. The next investigation compares the host's actual shell
selection with the CMD transport already tested offline; no shell diagnosis
is claimed from the missing callback alone.

The next offline probe reproduced a real renderer defect. Tagged Codex shell
detection selected `C:\Program Files\PowerShell\7\pwsh.exe`. The generated
Mnemos command failed under `-NoProfile -Command` with a parser error at `-c`.
Adding PowerShell's `&` call operator to the same command invoked the adapter,
returned the seeded context, and captured the distinct prompt across restart.
The fixture paths included spaces. See the [PowerShell probe](codex-powershell-probe/README.md).

This explains why the prior CMD-only test could pass while leaving the
default Windows command invalid. The production renderer repair and another
native-host check remain pending. The offline result does not retroactively
prove which shell or dispatch path caused every earlier native failure.

## Codex native capture v5

After the offline PowerShell renderer repair, a fresh Codex 0.146.0 run used
the installed d17 wheel, a disposable `CODEX_HOME`, `hooks.json`, and the
same unavailable-cognition Mnemos fixture. The host process returned code 0 in
47.611 seconds. The wrapper recorded three native hook calls: `SessionStart`
timed out, while `UserPromptSubmit` and `Stop` completed. The prompt hook
returned the seeded SQLite context. Both user and assistant events were
captured and survived the SQLite service restart.

The acceptance result remains `not_observed` because Codex answered
`The project uses PostgreSQL.` despite the SQLite seed. This is a failed native
seeded-recall outcome. The run proves partial native dispatch and prompt
context delivery only; it does not prove complete lifecycle capture, correct
host recall, or extraction quality. The three host error items were trust
notices and a shortened skill-description notice, not a Mnemos diagnosis.

The [v5 report](native-codex-capture-v5/report.json) is preserved unchanged.
Its exact runner, reviewed host output, and artifact hashes are indexed in
[the evidence index](native-codex-capture-v5/evidence-index.json). The private
fixture remains at
`D:\WindowsTemp\mnemos-native-codex-capture-v5-yedmb2ex`; the copied auth file
and service descriptor were removed, and no matching probe process remained.
Native Codex support therefore stays pending, with MCP and manual fallback
retained.

## Codex native capture v6

A second repaired run used the installed d17 wheel in a disposable project with
simple embeddings and cognition unavailable. The native command returned 0 and
the wrapper recorded three hook calls: `SessionStart`, `UserPromptSubmit`, and
`Stop`. Each returned 0. The prompt hook returned the seeded SQLite context,
and both user and assistant capture rows survived the SQLite service restart.
The hook used the manual PowerShell wrapper from the probe, so this run does
not exercise the later production renderer package.

Codex answered `The project uses SQLite.` for the seeded prompt, so the v6
seeded recall criteria are recorded as observed. That sentence is an observed
host answer, not a model-quality result. The run does not measure whether the
answer was caused by the retrieved context, and it does not establish a
performance bound. The host did not report a model name. The warm direct
adapter preflight was excluded from native invocation counts.

The [v6 report](native-codex-capture-v6/report.json) is unchanged. Its exact
runner, reviewed synthetic stdout observations, and hashes are indexed in the
[v6 evidence index](native-codex-capture-v6/evidence-index.json). The private
fixture remains preserved at
`D:\WindowsTemp\mnemos-native-codex-capture-v6-ftt13i5g`; its copied auth file
and runtime descriptor were removed, and no matching probe process remained at
cleanup. No live Codex configuration, host memory, or Ollama service changed.

## Codex generated configuration on source 20f271d

Native v8 exercised the actual installed package's generated TOML hook
commands, with no wrapper or hooks.json. Codex returned a random canary that
was absent from the prompt, made no tool call, and captured both conversation
roles across service restart. The fixture used constant unit vectors and
unavailable cognition, so it establishes hook delivery and persistence only.
Individual native hook traces were unavailable. See the [v8 record](native-codex-generated-v8/README.md)
for exact package provenance, observed usage, limitations, and cleanup.

The preceding [v7 preflight failure](native-codex-generated-v7/README.md) remains
preserved: its fallback embedding scored the generic query below the unchanged
relevance floor, and no native model turn started. This motivated separating
the transport fixture from semantic-quality evaluation.
