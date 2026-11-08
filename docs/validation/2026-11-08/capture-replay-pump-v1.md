# Owned capture replay lifecycle

The isolated capture candidate at `3b659f2a` adds an explicitly started `OutboxReplayPump` for a dedicated client. Startup schedules a drain of existing pending entries. Notifications coalesce, including notifications received during an active delivery. Periodic retries use a positive finite interval. Exceptions remain visible as an error type, without a successful result. Closing cancels the owned task before closing its client; durable queue entries remain authoritative.

Eight new cases and the five queue-receipt cases passed together. They cover startup, coalesced notifications during delivery, cancellation order, an actual disk queue surviving two pump/client restarts with the service unavailable, exception retry, and invalid intervals. The new module passed mypy and both new Python files were formatted with Black. No live worker or model was started.

The restart case does not start a new operating-system process, and the availability test does not prove real memory-service job completion. The pump is not yet wired to the resident hook worker. Hook acknowledgment, full process restart recovery, installed timing, and the unchanged 100 ms requirement remain outstanding. No live configuration or production default changed.

Next integrate the pump with explicit resident-worker ownership and a separately selected queued capture path. Preserve the existing host stdout schema and expose queue state through the authenticated internal worker response. Ensure startup and shutdown call the pump lifecycle, use a dedicated replay client with matching descriptor/outbox policy, and test worker shutdown while delivery is pending before any timing claim.
