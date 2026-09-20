# Installed provider outage, all module flags enabled: v1

The probe failed on its incorrect expectation of exactly one attempt per maintenance call. Both jobs remained pending after three attempts and no facts were committed. Restart was NOT RUN. The preserved failure is a harness expectation defect, not proof of lost memory.

The installed product is d937892. The input is an exact replay of the two
synthetic native Codex events preserved in native-codex-outage-v1. This is
not a new native host session. The service uses authenticated loopback HTTP,
SQLite and actual Ollama provider adapters pointed at a reserved non-listening
local socket. No live Ollama endpoint or host configuration was changed.

All five flags use enabled defaults, but prediction failure prevents downstream
module execution. No mock model responses were used. This does not prove
real-model extraction recovery, full-module quality or the cross-host pilot.
Real-provider recovery is NOT RUN and must be recorded separately. The default
scheduler's exponential backoff permits several attempts during one batch.
A maintenance call is not an exactly-once-attempt boundary.

The archive includes the frozen inputs, runner, terminal outcome and an
integrity/FK-checked SQLite backup. Runtime descriptors and authentication are
not included. No selection or semantic acceptance criterion was changed.
