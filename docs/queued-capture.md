# Experimental queued capture

The local delivery branch supports opt-in queued capture for a resident Codex worker. Synchronous capture remains the default. This does not install hooks or change the explicit-memory pilot.

Run the worker from its bound project directory, with the explicit profile, enabled project registry and matching caller environment. Its descriptor must differ from the memory service descriptor and lock. Add `--queue-captures` to the worker command. Select `--outbox-backend sqlite` for SQLite or `--outbox-backend json` for the default JSON backend. SQLite requires queued mode. Claude Code queued mode is not supported. Callers select the worker through `MNEMOS_HOOK_WORKER_DESCRIPTOR`.

A queued receipt means the sanitized capture is durable in the local outbox. It has `durable_queued=true` and `committed=false`; worker output reports `capture_state=queued`. It does not establish service acceptance or completed extraction. Failed appends cannot return a durable queued receipt. The replay owner retries while the worker runs and follows the configured service autostart policy.

SQLite uses a separate sibling queue, WAL journaling and FULL synchronous commits. Capture appends commit individually. Replay acknowledges accepted entries in batches of at most 32. Interruption before acknowledgment can redeliver the same IDs, which the service deduplicates. Stopping the worker preserves pending entries.

To switch back to JSON, stop the SQLite worker and restart with `--queue-captures --outbox-backend json`. Keep queued mode enabled until both backlogs drain. SQLite mode also drains JSON fallback entries; JSON queued mode drains an existing sibling SQLite backlog. There is no automatic migration or deletion. Removing `--queue-captures` disables replay owners and can leave a SQLite backlog waiting on disk.

The integrated delivery tree passed 181 focused queue, service, native-launcher, identity and documentation checks, with one Windows open-SQLite rename skip. The source candidate separately passed 3,761 tests and 87 configured native/documentation follow-up checks. The [rebuilt delivery package](validation/2026-11-08/canonical-queue-package-v1.md) separately passed 250 selected installed checks with one Windows skip. A three-pair candidate backlog comparison improved replay time, but complete durable-capture latency still fails the 100 ms requirement. This feature remains experimental, disabled by default and unpublished.
