# Native Codex pilot startup outage

One native Codex turn preserved both its user prompt and assistant reply while
the disposable service could not start. The installed pilot launcher retained
its normal auto-start setting; the controller held the runtime ownership lock
for that turn. No memory tool was available, and the reply acknowledged only
the supplied marker. The existing database stayed unchanged while both hook
messages entered the durable outbox with their physical project, profile,
session and native turn identities intact.

After the controller released the lock, an ordinary launcher health request
started the installed service and drained both messages into pending jobs.
Maintenance completed those jobs as two unasserted raw chunks. The original
assistant event stayed verbatim; its processed chunk carried the existing
"Assistant reported:" qualification. Neither record became an accepted fact.
The explicit control remained unchanged. Replaying both original payloads
and restarting again left events, jobs, chunks, evidence and revisions equal
to the completed snapshot. SQLite integrity and foreign keys passed.

This is a simulated startup outage on installed product `6d0e5ba6`, with
runner `e523e50a`, Codex CLI 0.164.0-alpha.11.1 and `gpt-6-astra`. The native
turn took 24.91 seconds. All 95 focused checks and the runner's formatting and
strict types passed before the attempt. The [receipt](native-pilot-codex-outage-v1-receipt.json)
binds the wheel, launcher, native executables and reported usage. All 141
members of the private archive verified; its SHA-256 is
`be8a7f4e9469bacfe57c4e441b83557dd1fefc5160ff7e4ead8352250dd6ffd8`.
The [prospective protocol](../../plans/2026-11-07-native-pilot-codex-outage-v1.md)
and original native streams remain attached to this single attempt.

The independent audit initially expected the processed assistant chunk to be
verbatim, overlooking its existing role qualification. A later PID-only check
also failed transiently; subsequent inspection found none of those PIDs
present. The final audit checks creation time as well as PID, and found no
remaining owned process or descriptor-specific service. Both failed helpers
and their outputs are preserved. Neither audit correction reran the host.
Copied authentication was removed, source credentials and inactive pilot
files stayed unchanged, and the real pilot still has no runtime or database.

The one-run hook trust override and disabled Code Mode notices are retained.
This run proves durable capture and recovery under the stated startup fault.
It does not establish completion within every hook deadline, provider outage
recovery, mid-write crash recovery, ordinary interactive trust, repeated
reliability, extraction quality, live activation or actual pilot days.
