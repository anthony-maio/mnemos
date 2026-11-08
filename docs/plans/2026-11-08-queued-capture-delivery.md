# Queued capture delivery candidate

Status: prepared for review, not integrated or published. The candidate is in `wt/mnemos-capture-ack` at 0684c983. It has focused installed validation, a three-pair backlog comparison and source process-exit recovery coverage. Full source validation is running separately. The existing explicit-memory pilot remains unchanged.

## Runtime scope

The proposed integration consists of six files: `mnemos/adapters/codex.py`, `mnemos/client.py`, `mnemos/hook_worker.py`, `mnemos/outbox_replay.py`, `mnemos/service.py` and `mnemos/sqlite_outbox.py`. Include the nine corresponding queue, replay, response-completion and process-exit test files. Exclude `project_identity_cache.py` and its tests: that prototype still requires external Git-policy invalidation and is not connected to runtime routing.

Queued capture is opt-in and currently supports Codex only. Synchronous capture remains the default. The worker accepts `--queue-captures` and, with that flag, `--outbox-backend json` or `--outbox-backend sqlite`. JSON remains the default backend. SQLite without queued capture is rejected. These options do not install hooks or activate the explicit-memory pilot.

## Operation and receipts

Start the worker from its bound project directory, using the same environment as its callers. Supply the existing explicit user profile, enabled project registry and settings. Give the hook worker its own descriptor path, distinct from the memory service descriptor and lock. Callers select the hook descriptor through `MNEMOS_HOOK_WORKER_DESCRIPTOR`. Changing the working directory, settings or relevant environment requires a new matching worker; requests that do not match the owner's binding are declined.

`durable_queued=true` means the sanitized capture has been committed to the local outbox. It does not mean the memory service has accepted it or completed extraction. The queued receipt has `committed=false`; worker output reports `capture_state=queued` and the event IDs. A failed append cannot produce a durable queued receipt. Source identity conflicts remain errors.

The replay owner starts with the worker and retries while it remains running, including when no new hook event arrives. Its service startup behavior follows the client's `MNEMOS_AUTO_START` setting. With autostart disabled and no service available, captures remain queued. Stopping the worker stops replay; it does not delete pending entries. HTTP handling wakes replay after the response write attempt, including when that write fails after durable capture.

SQLite uses a separate sibling outbox file, WAL journaling and FULL synchronous transactions. Capture appends commit individually. Replay removes accepted entries in batches of at most 32. Process exit before an acknowledgment commit can redeliver previously accepted entries with the same IDs. The service deduplicates those identities. Never interpret the queue's acknowledgment count as a count of completed extraction jobs.

## Backend rollback

When SQLite is selected, a compatibility replay owner also drains JSON entries, including fallback entries added after startup. To return to JSON, stop the SQLite worker and start a matching worker with `--queue-captures --outbox-backend json`. Keep queued capture enabled during this transition. If the sibling SQLite queue exists, the JSON worker also replays its remaining entries. There is no automatic migration or deletion of the old database.

Simply removing `--queue-captures` disables the replay owners and can leave a SQLite backlog waiting on disk. Verify both queues have drained before switching to synchronous-only operation. Preserve pending queue files if the service is unavailable; do not delete them to make a status check appear clean. Existing worker descriptors are bound to capture mode and backend, so a different mode requires stopping the old owner first.

## Evidence and remaining requirements

- [Installed v7 checks](../validation/2026-11-08/sqlite-outbox-installed-v7.md): 42 passed, one expected Windows skip.
- [Backlog comparison](../validation/2026-11-08/replay-batch-benchmark-v2.md): all three paired replay times improved, with exact event/job identity checks. This is not a capture-latency measurement.
- [Process-exit recovery](../validation/2026-11-08/replay-process-exit-v2.md): two source cases passed across the acknowledgment boundary and service restart.

Before integration, inspect the complete source-suite outcome and resolve relevant failures. After integration, validate the actual delivery tree and build an artifact from that tree; candidate evidence alone does not establish delivery-package correctness. The 100 ms complete durable-capture requirement, answer-support quality, fresh acceptance and original two-project daily-use pilot remain outstanding. No release or live activation is implied by this document.
