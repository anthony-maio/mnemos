# Queued capture response completion

The capture candidate now wakes replay after the HTTP response write attempt, instead of during hook dispatch. The acknowledgement still follows the existing sanitized, permission-protected, fsynced outbox write. This changes notification timing, not the durability boundary.

The HTTP service has a separate dispatch entry point and a response-finished callback. Ordinary services retain their existing submit behavior and a no-op callback. The queued hook worker uses the later callback to schedule its replay notification on the owner loop. Direct worker submit calls also schedule replay, without requiring an HTTP response. Standalone dispatcher callers keep their existing notification behavior.

A failed response write still runs the callback because the capture may already be durable. The client continues to report an unknown outcome when its response is lost; no new retry or fallback is added. Timer-based replay and startup replay remain available when a notification is lost, a response never finishes, or shutdown races the callback. A late callback after shutdown does not create a new owner.

The focused suite passed 52 tests covering response ordering, failed writes, direct submit, lost-notification recovery, pump cancellation, worker restart, real SQLite persistence and lifecycle behavior, normal service operation, and resident hook behavior. Strict type checking passed for both changed runtime modules. The first run had 51 passes and one failed test expectation: the synthetic broken response correctly raised the existing `HookOutcomeUnknown`, which the new test initially did not catch. The corrected test requires that exception and then verifies replay still completes.

An additional 31 tests passed for lost acknowledgements, missing jobs/facts, and related-forget error handling. Total passing coverage for this change is 83 tests.

This implementation has not yet been built into a new isolated package or measured at the installed command boundary. The diagnostic's deferred-until-command-exit arm is not identical to notification after a server write attempt, so its timings cannot be claimed for this implementation. Rebuild, verify installed source parity, and run a fresh frozen comparison before making a performance claim. No live pilot configuration, database, published package or canonical delivery source changed.
