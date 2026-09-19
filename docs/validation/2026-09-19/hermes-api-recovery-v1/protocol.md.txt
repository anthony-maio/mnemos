# Installed Hermes API recovery protocol

This T17 probe uses the installed Hermes MemoryManager API, the verified
`5e4b057b16906c2ec7dd302ae80748ef1ac5eddc` Mnemos wheel, and the actual
Windows/WSL bridge. It runs disposable host homes, scope registries, outboxes,
and SQLite storage. Embeddings are fixtures, cognition is unavailable, and
all optional modules are disabled. No model call or live host configuration
change is part of this probe.

The earlier API lifecycle check establishes ordinary tool dispatch and provider
recreation while the service is available. This check adds a service outage,
a new host process, durable queue recovery, and replay after forgetting.
The controller restores the fixture service explicitly. Success cannot be
described as automatic daemon recovery or a native model conversation.

Freeze this protocol and the executed helper before collecting results.
Preserve failures and create a new output directory for any later attempt.
Bind the report to the source commit, wheel digest, installed Hermes revision,
protocol digest, and helper digest. Check the host source remains unchanged.

## Phases and required evidence

1. Start a fixture service and seed separate project and foreign-project
   controls. Stop the service cleanly. Bind attempted automatic starts to an
   isolated malformed configuration so failure cannot launch a live runtime.
2. In the first WSL host process, initialize MemoryManager with explicit
   project identity. Submit a two-message transcript through its normal
   callback path. Request a required pre-compression checkpoint and exit the
   host. Record callback, checkpoint, and shutdown outcomes without treating
   a host-level flush alone as evidence of durable capture.
3. Inspect storage after host exit. The stopped service must not have accepted
   those events, and the isolated outbox must contain both sanitized payloads
   with distinct stable event IDs and occurrence timestamps. Any successful
   checkpoint must be supported by those durable payloads. Report a failed
   checkpoint truthfully even if raw capture survived in the outbox.
4. Explicitly restore the fixture service. A new WSL MemoryManager process
   makes an ordinary provider request; its transport must drain the outbox
   without the coordinator calling replay. Repeat the original transcript
   through that provider and record the resulting acknowledgements.
5. After maintenance, require one raw event and one completed extraction job
   per original occurrence, with matching event IDs and timestamps. The
   outbox must be empty. Each accepted fixture fact must have scoped evidence;
   the foreign control must stay unchanged and unavailable in project recall.
6. Forget a selected captured fact using its exact ID and current version.
   Close and restart the service and use a new host process to replay the same
   transcript. After maintenance, the fact must remain absent, its source
   suppression must remain effective, and unrelated controls must survive.

Record phase durations and timeouts. Do not increase an observation deadline
to turn a failed attempt into a passing result. A process timeout does not
justify restarting a still-live process; inspect the same handle first.
Capture traces and selected synthetic database rows, but omit runtime tokens,
credentials, and live user content from the repository archive.

## Claim boundary

Passing these phases establishes the observed installed API, bridge, queue,
restart, and suppression behavior. It does not establish hard-kill recovery,
full native CLI capture, missing-transcript association, real-model extraction,
cross-host model-driven correction, two-week reliability, or latency targets.
The CLI cwd propagation gap retains its separate patch and acceptance record.
