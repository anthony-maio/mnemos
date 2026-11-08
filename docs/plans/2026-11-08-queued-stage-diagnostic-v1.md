# Queued command stage diagnostic

Use the same installed package and native binary as the paired command timing study, in a fresh isolated store with deterministic embeddings and no cognition. Preserve five warmups and 50 alternating pairs. Keep all prior failures unchanged.

Instrument the resident dispatch using a context-local timer for enqueue, sanitization, outbox write, permission checks, lock acquisition and fsync. Record counts and inclusive durations in the observed worker receipt. Each dispatch has its own context; background replay without that context is excluded. Nested stage durations overlap and must not be added. Wrapper and timer overhead remain included; these are diagnostic timings, not a new acceptance result.

Correlate each diagnostic with the full command timer, queued receipt, original event ID, persisted event and completed job. Subtract only the same sample's dispatch duration from its command duration. The remainder includes process launch and exit, transport, scheduling, response handling and observation work; it cannot identify one of those costs on its own. Freeze the helper, wrapper, dependency identities and schedule before execution. Verify cleanup and artifact hashes, then use the result to select the next bounded repair or measurement.
