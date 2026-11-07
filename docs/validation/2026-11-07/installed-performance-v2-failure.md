# Installed performance V2: Windows ownership check failure

V2 fixed the embedding-identity setup error and created the same 10,000 real
embeddings and source-backed fixture facts. The service wrote its runtime
descriptor, but the harness stopped before any timed requests because it
required that descriptor's PID to equal `subprocess.Popen.pid`.

That assumption does not hold for this Windows virtual environment. A separate
model-free probe kept a child interpreter alive, verified its parent through
the operating system, and observed a different PID for the launcher and the
interpreter. Gracefully ending the child let the launcher exit successfully.
The probe does not recover the failed attempt's unrecorded live process tree,
but establishes the ownership relationship the next harness must handle.

V2's fallback cleanup terminated its launcher. Subsequent OS inspection found
the driver, launcher and recorded service interpreter absent, with no listener
on the recorded port. A stale descriptor remains in the failed runtime and is
preserved as evidence. No timed retrieval, hook, native-host or latency-gate
result was obtained. Fixture embedding and loading took 30.86 seconds, separate
from model construction.

The complete failed attempt, bound inputs and identity/ownership probes are
preserved in a verified 117-member private archive, SHA-256
`04e359f103abc306d3971af1d5f0b5a969bd9b4d344a1308c0df053cd6fd02f4`.
V1 remains preserved separately. Neither failure changed product source or live
configuration, and the pilot remains inactive.

The next harness must verify the service interpreter's parent and exact private
command before treating it as owned. It must register that owner for graceful
authenticated shutdown, then confirm both processes exit and the descriptor
is removed. A different PID alone is insufficient authority to stop a process.
The original workload, complete command timers and latency thresholds remain.

The subsequent [V3 result](installed-performance-v3.md) verifies that ownership
repair and completes the workload. Warm recall meets its target; both complete
hook commands miss the unchanged 100 ms requirement.

- [Failure receipt](installed-performance-v2-failure-receipt.json)
- [V2 fixture identity amendment](../../plans/2026-11-07-installed-performance-v2.md)
- [V1 failure](installed-performance-v1-failure.md)
- [Original performance protocol](../../plans/2026-11-07-installed-performance-v1.md)
