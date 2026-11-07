# Resident hook worker and complete-command latency

The installed hook baseline misses the 100 ms requirement. Its fresh Python
process imports and configures the adapter on every event; the import repair
left roughly 690 ms of empty-command startup in the diagnostic comparison.
The next implementation keeps that adapter resident and makes the command
boundary small. The original full-command timer and threshold stay unchanged.

The experimental `mnemos.hook_worker` runs the existing Codex or Claude adapter
inside a separate authenticated loopback runtime. It reuses `MnemosService` for
private descriptors, owner locks, bearer authentication and bounded requests.
Its descriptor and lock must differ from the memory service's. The worker owns
only its adapter process; closing its client does not stop the memory service.

The command client, `mnemos.hook_client`, imports only the standard library and
lightweight host input/diagnostic helpers on its resident path. A request binds
the host, protocol, invocation ID, environment fingerprint, working directory,
event and original payload. Replies must identify that invocation. Neither
environment values nor captured text belong in diagnostics or descriptors.

Each worker is currently bound to one working directory and the environment
present at startup. The fingerprint covers all environment values except the
worker-descriptor routing variable. This avoids silently reinterpreting relative
paths, home-directory defaults or provider variables in a different process.
At every event, reload effective settings and scope membership. Changed settings,
environment or directory decline before adapter dispatch. No process-wide chdir
or environment mutation is allowed while handling a request.

Serialize host events. Keep the authenticated HTTP pool between events, but reset
the one-autostart-attempt allowance and outbox-drained marker for each new event.
Another command process may have queued a capture since the preceding event.
Run the existing client path: sanitize, persist and fsync its outbox before any
memory-service startup or HTTP wait. Raw host input crosses the new authenticated
local boundary in memory; it is never written into a new raw-input spool.

If no worker request has been sent, use the ordinary adapter. A bound refusal
that explicitly says dispatch did not begin also permits fallback. Once request
bytes may have been sent, a timeout, lost reply, error response or malformed
acknowledgement is an unknown outcome. Emit a sanitized diagnostic and preserve
the host's nonblocking output. Do not retry the capture or claim it was queued.
Any accepted capture already follows the existing durable outbox path.

Keep both host adapters' output, source identity and checkpoint behavior. Native
Claude's existing provisional-identity limits remain visible. The worker does
not turn a queued event into a completed fact or enable automatic extraction.
No generated host configuration or live profile selects this path by default.

Validate transport correlation, authentication, changed membership/configuration,
outage capture, worker restart, external outbox replay, renewed startup attempts,
lost acknowledgement and actual command subprocesses that cannot import the
ordinary runtime. Run the related client, adapter and service regressions.

Then build an isolated artifact and measure complete commands against the same
accepted-fact snapshot with real embeddings, original scope/configuration checks
and full prompt context. Include process startup, worker dispatch, durable capture,
retrieval and host output in the timer. Keep every sample and failure. Report
worker startup separately; do not replace the full hook requirement with an IPC
microbenchmark or disable retrieval to obtain a passing prompt time. Native-host
acceptance and pilot activation remain separate after the installed comparison.
