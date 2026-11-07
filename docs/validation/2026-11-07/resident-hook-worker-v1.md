# Resident hook worker: implementation and package checks

The experimental hook path can now keep the existing Codex or Claude adapter
resident. A small command process sends the event to that worker, which performs
the normal scope resolution, capture, retrieval and host-output work. The command
path uses the standard library; it does not import settings, Pydantic, HTTPX,
the service, scope models or either full adapter when the worker handles the event.
No host configuration selects this path by default.

The worker reuses the existing authenticated loopback runtime with a separate
private descriptor and owner lock. It is bound to one working directory and
environment. Every event rechecks effective settings and reloads scope membership.
Serialized events reuse the HTTP connection pool, while each new event receives
a fresh bounded autostart allowance and checks the shared outbox again.

The ordinary client still sanitizes and fsyncs the outbox before memory-service
startup or transport. Raw input crosses the new authenticated local boundary in
memory, without a second raw-input spool. Missing or unreachable workers permit
the ordinary one-shot adapter fallback before request bytes are sent. A bound
refusal before adapter dispatch also permits fallback. After dispatch may have
begun, timeout, transport failure or an invalid reply is an unknown outcome;
the command emits a sanitized diagnostic and does not resend the capture.

## Verified behavior

Twenty-one new checks pass. They cover durable outage capture across a worker
restart, replay of records queued by another client, renewed per-event startup
attempts, changed membership and settings, environment and directory mismatch,
secret refusal, authentication, descriptor/lock separation, Claude's existing
provisional capture, fallback, and lost acknowledgement after a durable write.
Misbound replies and a refusal after dispatch never authorize a retry.

Two actual command subprocesses complete separate captures through a resident
worker while an import guard forbids the ordinary adapter runtime. The test memory
dispatcher verifies that each event already exists in the durable outbox when
it receives the request. These are synthetic protocol checks using a recording
dispatcher, not native-host acceptance, real retrieval or latency evidence.

All 153 related client, Codex/Claude adapter, service, Windows private-path,
UTF-8 and import-boundary checks pass. Black and strict mypy pass for the changed
source. The initial new-test fixture called a nonexistent profile helper; correcting
it to use project registration removed the twelve setup errors. The original
failed log and corrected runs are retained.

## Built candidate and remaining validation

Source `feb6ac3b3eebd2e178ec584b3be1e72cca1228fc` builds with isolated Hatchling
1.33.0 and passes Twine. Wheel SHA-256:
`51ba20ba811eaa33cd18bde9665c49d21feaa10e7874bffdf878e3b9762b550f`.
All 94 installed package files match the wheel, and all 83 packaged Python files
match the checkout. An isolated installed import check verifies the lightweight
command boundary and that the worker, client and both adapters load from the
installed target. Compared with the earlier import-repair wheel, no package
file was removed; the support composition and two hook modules were added.

The existing environment has no pip module. The successful installation used
uv with no dependencies into a new private target. Normal packages, profiles,
databases and host configuration were not upgraded.

The 100 ms complete-hook requirement remains open. The next comparison must use
the installed candidate, the accepted-fact fixture, real embeddings and the full
prompt response. It must include command startup, durable capture, worker dispatch,
retrieval and output, while preserving the private launcher's configuration checks.
The original failed measurements remain unchanged. Current installed lifecycle,
native-host acceptance and actual pilot use have not run for this candidate.

The subsequent [installed comparison](resident-hook-performance-v1.md) records
complete command measurements with real retrieval and restart durability. It
reduces median latency but still fails the 100 ms requirement. Native-host and
pilot acceptance remain separate.

- [Worker and performance protocol](../../plans/2026-11-07-resident-hook-worker-v1.md)
- [Verification receipt](resident-hook-worker-v1-receipt.json)
- [Original hook latency diagnosis](hook-startup-diagnostics-v1.md)
- [Prior installed performance result](installed-performance-v3.md)
