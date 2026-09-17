# Direct checkpoint diagnostic

The native persistent Claude v5 timeout did not reproduce in this bounded
direct adapter check. Both subprocesses completed within the original
10-second limit: UserPromptSubmit in 1,291.0 ms and Stop in 1,050.3 ms.

The installed package matches all 53 tracked package files at `20f271d` after
line-ending normalization and all 56 wheel package members byte for byte.
The wheel digest is recorded in `report.json`. The fixture uses constant
vectors, unavailable cognition, and an authenticated local service. No Claude
process, host credentials, model call, or external provider was used.

The user event was captured, its pending checkpoint was completed, and Stop
captured the assistant and advanced the transcript checkpoint to byte 475.
The store contains the explicit seed, one prompt, and one assistant event.
UserPromptSubmit reports degraded identity coverage because the fixture lacks
a native prompt ID; Stop reports success. Checkpoint fsync took 160.655 ms
and 180.207 ms respectively; replacement took less than 1 ms in both cases.
No delayed stack dump fired, and the service descriptor was absent after
shutdown.

Trace records are flushed without an extra fsync, so recording does not add
durable commits around the operations under investigation. The active frozen
real-model evaluation and focused verification shared this machine. These
observations are diagnostic traces, not isolated performance measurements.

The original native failure remains unexplained: its incomplete temporary
checkpoint localizes the interruption window but does not identify a blocking
syscall. This successful direct check neither fixes that failure nor proves
native persistent transcript acceptance. Stop replay is a separate known
defect at this source revision and was not exercised here.

The exact runner, raw operation traces, and original report are indexed in
`evidence-index.json`. The private synthetic fixture remains on disk; no
runtime token or descriptor is copied into this evidence directory.
