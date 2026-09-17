# Persistent Claude transcript attempt

This attempt did not complete. The native process reached its 60-second bound
and was terminated by PID tree. No Stop hook or final result record was
observed, so transcript checkpoint and replay acceptance were not exercised.

The installed `20f271d` wheel was verified against the source and archive.
The fixture used a private `CLAUDE_CONFIG_DIR`, a protected temporary copy of
only the existing credential file, invocation-specific hooks, project-only
settings, no tools or MCP servers, Sonnet 4-6, and an $0.08 bound. Session
persistence was enabled. The wrapper rejected transcript paths outside the
private configuration directory before invoking the adapter.

SessionStart completed and supplied the memory canary. UserPromptSubmit's
nested adapter exceeded its 10-second deadline. The resulting wrapper error
is preserved in the native hook response. The SQLite fixture contains the
user prompt, but no checkpoint file or assistant capture was present after
shutdown. That narrows the observed state; it does not establish which step
caused the delay.

The assistant stream did contain the correct canary, but the host emitted no
final result, Stop invocation, or final cost before termination. The immutable
report's answer criterion reads final result records and therefore remains
false. `observations.json` preserves the partial assistant message and its
partial usage fields separately. Actual billed cost is unknown; zero cost is
not claimed. The reason the native stream did not finish is unresolved.

The private transcript exists under the intended configuration root. It has
eight JSONL records, including a user record and attachments, but no assistant
record. Its hash and shape are recorded in `post-run-state.json`. No successful
persistent-transcript, assistant capture, or replay result follows from this
attempt. Constant vectors and unavailable cognition also preclude a semantic
quality claim.

The exact runner, original report, reviewed synthetic stream fragments, and
post-run state are preserved. The credential copy and runtime descriptor are
absent, and no matching fixture process remained at cleanup. No live host
configuration or memory data changed. The separate completed no-persistence
v4 result remains valid within its stated limits.
