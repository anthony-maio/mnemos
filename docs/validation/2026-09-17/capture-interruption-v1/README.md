# Capture interruption durability

Before the repair, a host terminated during service startup or HTTP dispatch
could lose capture because buffering happened only after a transport failure.
The repaired client persists sanitized capture and its stable event identity
before those waits. It removes the entry after commit or explicit rejection,
and retains it on interruption or an ambiguous response.

Two regressions kill an actual child Python process at request boundaries:
before sending, and after a real SQLite service commit but before the client
handles its response. Service/database restart and replay produce exactly one
event and job. Four more cases check pre-dispatch durability and sanitation,
terminal commit/rejection removal, and retention for ambiguous responses.

All six fail against the prior installed wheel from source6442044. Its exact
test copy, package origin, report, raw log, and XML are included as red-v3.
The repaired source passes all six and 84 related service, lost-acknowledgment,
startup, Codex, and Claude cases.

Earlier iterations remain preserved: red-v1 had two failures; red-v2 had six.
Green-v1 had 87 passes and three fixture errors because its final sanitation
check tried reading an outbox file already deleted on acknowledgement. The
check now inspects the persisted snapshot captured during dispatch. Green-v2
has 90 passes. No durability or replay assertion was removed.

This does not fix the native three-second hook deadline or host shutdown.
It protects interruption after the outbox write, not termination before it.
Provider quality and performance are separate gates.

A seventh case commits, forgets, kills the client before acknowledgement,
then restarts and replays. The outbox clears without restoring the fact or
source event. The job remains completed with a forgotten marker and no
captured text; forced maintenance does not recreate the fact. All seven
cases pass in forget-v3. Earlier forget-v1 omitted expected_version;
forget-v2 incorrectly expected deletion of the completed job marker. Those
fixture failures are preserved. Product code did not change between them.
