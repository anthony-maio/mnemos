# Installed pilot command timing

Freeze this protocol before measurement. Keep product `f5a944ea`, its existing
wheel, the inactive pilot launcher, settings and dependency versions unchanged.
Compare the proposed Python hook command with the existing experimental native
v3 command using that same installed environment. The native launcher remains
experimental. This run cannot change the pilot configuration or promote it.

Use disposable copies of the preserved 10,000-fact seed and fixed queries.
Copy the pilot launcher byte for byte. Relocate only the database/runtime paths
and synthetic project registration; verify all other settings match the pilot.
Run Python-0, native-0, native-1, Python-1 sequentially. Each block gets a fresh
service process, three recall warmups and 25 timed recalls, then one warmup and
25 timed commands for each of Stop and UserPromptSubmit. Start a worker only
for native blocks. Preserve all 320 observations, including 300 timed results
and 160 output pairs. Do not retry, discard or trim measurements.

Measure these distinct boundaries:

- Complete command: start immediately before launching the executable; stop
  after stdout/stderr have closed and the process has exited. The Python arm
  uses the pilot interpreter with `-I launch.py hook codex`. This excludes
  Codex dispatch and any outer PowerShell startup. It is a command subprocess
  measurement, not a native-host acknowledgment measurement.
- Transaction component: a private service wrapper records monotonic timestamps
  immediately before and after the installed `capture_event_and_job` call.
  Record event identity, outcome and transaction state. Keep spans in memory
  and flush after service shutdown; add no per-capture disk logging. The
  package bytes remain unchanged. Correlate each span with its command and
  separately report dispatch-to-transaction-return time. These component
  measurements are not a host-visible acknowledgment or an outbox timing.
- Complete retrieval: time the client retrieve operation, including transport,
  local embedding, retrieval and context construction. Keep service/worker
  startup-to-health measurements separate, including ownership-check overhead.

Use interpolated p95, retaining the original 100 ms complete-command and 500 ms
warm-recall limits. Report transaction and dispatch-to-commit distributions as
diagnostics without substituting them for those gates. Neither command exposes
an early durable acknowledgment. Stop stdout and exit zero are insufficient
evidence of successful capture.

Require exact decoded output and full context equality for matched commands.
For native commands require one handled, nondegraded worker dispatch with
matching request/output hashes and no fallback. For every capture verify the
event and job, record any pending outbox entries, and check recovery after a
service restart. Separately report committed versus completed jobs. Verify
accepted facts, evidence, sources and embeddings remain unchanged, both stores
use the evidence index, package/assets retain their hashes and all owned
processes stop. Recompute metrics and correlations in an independent audit.

Bind the private driver, instrumentation, installed package, launcher/settings,
model assets, seed and queries before measurement. Archive every attempt and
the audit. Preserve all previous timing FAIL results. No semantic model run,
native-host turn, live configuration change or pilot day is part of this run.
The earlier bounded quality comparison remains closed and failed.
