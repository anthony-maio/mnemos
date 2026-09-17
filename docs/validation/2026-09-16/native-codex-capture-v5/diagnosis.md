# Native Codex v5 diagnosis

This is a read-only diagnosis of the preserved v5 smoke. It does not change the
private fixture, Codex configuration, authentication, or evidence report.

## Observed result

The report is honestly marked not_observed. The Mnemos side did complete two
durable role captures, the seeded fact survived restart, and the UserPromptSubmit
adapter returned the seeded SQLite context. Native Codex then answered
"PostgreSQL", so the host model did not produce a correct seeded-memory answer.

The private fixture is
D:\WindowsTemp\mnemos-native-codex-capture-v5-yedmb2ex. The report is
docs/validation/2026-09-16/native-codex-capture-v5/report.json.

The callback order was not shown to be reversed. NTFS write times, which are
wrapper file completion times rather than hook execution start times, are:

- SessionStart: 2026-09-17T06:16:11.9752449Z, return code 124.
- UserPromptSubmit: 2026-09-17T06:16:24.2067462Z, return code 0, seeded context present.
- Stop: 2026-09-17T06:16:24.2791085Z, return code 0.

The report listed files lexicographically, which placed Stop before
UserPromptSubmit. The completion times are consistent with UserPromptSubmit
then Stop for the same turn ID, but they do not independently prove hook start
order.

## Concrete timeout mechanism

The v5 runner configures every command hook with timeout 15 in
.venv/native_codex_capture_v5.py lines 238-248. Its generated wrapper starts
a second Python process with subprocess.run(..., timeout=20) at lines 199-215.
The command itself is run through the native Windows shell.

The tagged Codex v0.146.0 hook command runner confirms that it waits with the
handler timeout and uses kill_on_drop(true). When the timeout expires, the
shell child is dropped and killed, but that does not establish that a nested
Python descendant has exited. This is consistent with the trace:
the native hook budget can expire first while the nested wrapper later reaches
its own 20 second timeout and writes a private record with returncode 124 and
stderr TimeoutExpired.

Primary tagged sources:

- https://github.com/openai/codex/blob/rust-v0.146.0/codex-rs/hooks/src/engine/command_runner.rs
- https://github.com/openai/codex/blob/rust-v0.146.0/codex-rs/config/src/hook_config.rs

The timeout mechanism is established. The exact stage inside the Mnemos
SessionStart call that consumed the budget is not established by this run,
because the wrapper recorded no monotonic start/end data for its adapter
subprocess after subprocess.run timed out.

## Supporting timing evidence

The native process ran for 47,610.7 ms and exited with code 0. Native stderr
recorded a 5.4123442 second sqlx SQLite initialization statement and a
5.4128481 second pool acquisition warning before the process reported that it
was reading prompt input. Because this stderr belongs to the Codex process,
the database cannot be identified as Mnemos storage from this artifact. It is
evidence of host-side SQLite or startup work in the same run, not proof of
Mnemos service latency. It also does not prove that the SessionStart retrieve
itself took 15 seconds.

## UserPromptSubmit output contract

The preserved UserPromptSubmit stdout is structurally valid for the tagged
Codex parser:

{"continue":true,"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"..."}}

The tagged parser reads additionalContext from hookSpecificOutput and requires
hookEventName when that object is present. The preserved output supplies both,
with the exact UserPromptSubmit value, and uses continue true. The tagged
parser does not treat that shape as a blocking or invalid result. See the
tagged parser at
https://github.com/openai/codex/blob/rust-v0.146.0/codex-rs/hooks/src/engine/output_parser.rs
and the tagged wire schema at
https://github.com/openai/codex/blob/rust-v0.146.0/codex-rs/hooks/src/schema.rs.

The tagged wire types make the boundary precise. The outer
UserPromptSubmit output and its hookSpecificOutput object reject unknown JSON
fields. hookEventName is required inside hookSpecificOutput, while
additionalContext is optional. continue defaults to true when omitted and the
explicit true value in this trace is nonblocking. Therefore the v5 output has
no missing required field, unknown-field, or continue:false explanation for
the absent seeded answer.

The tagged core handoff also accepts this context for model use. Its
inspect_pending_input path invokes the UserPromptSubmit hooks, and the
context-injecting hook outcome carries the collected additional_contexts. The
core then converts each context to a HookAdditionalContext conversation item
and calls record_conversation_items before continuing the turn. See the tagged
core path at
https://github.com/openai/codex/blob/rust-v0.146.0/codex-rs/core/src/hook_runtime.rs.
This proves that the host is designed to put the parsed context into the
session conversation history. It does not expose the final provider request,
attention, or model-use decision, so the v5 JSONL still proves only parseable
context output and a native final answer, not successful context consumption.

This establishes that the adapter emitted context in the host's accepted
format. It does not establish that the model used the context. The native
answer PostgreSQL is direct negative evidence for a correct seeded-memory
answer, while the host's internal context assembly is not exposed by this
JSONL stream.

UserPromptSubmit succeeded after the service was warm, returned
hookSpecificOutput.additionalContext, and produced the seeded SQLite context.
Stop also succeeded and captured the assistant role. Thus the observed
difference is consistent with a cold or contended SessionStart path, followed
by a warm prompt path, while leaving other scheduling or process startup
causes open.

## Smallest next test

Keep the native model invocation unchanged and run one bounded diagnostic
smoke after the service is already warm:

1. Add private monotonic timestamps around wrapper start, adapter subprocess
   start/end, and trace write.
2. Record only durations, exit codes, and digests in the report. Keep full
   payloads and streams in the private fixture.
3. Invoke the adapter directly once against the already-running service before
   the native turn, then compare that duration with the native SessionStart
   invocation.
4. Set the nested wrapper timeout below the native 15 second hook timeout,
   for example 10 seconds, so a wrapper failure is recorded before Codex kills
   its shell and leaves no ambiguous descendant.
5. If the warm direct adapter call is fast while SessionStart remains slow,
   inspect native command startup and shell process lifetime. If both are slow,
   inspect the service retrieve path and separately instrument the Mnemos store.

The current evidence supports increasing diagnostic visibility and aligning
nested and native deadlines. It does not support calling SessionStart an
adapter logic failure, claiming native Codex seeded recall success, or
attributing the latency to SQLite alone.

