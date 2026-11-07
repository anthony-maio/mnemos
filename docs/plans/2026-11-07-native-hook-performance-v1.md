# Complete installed Windows HTTP transport comparison

Compare baseline `5735c502` with candidate `788766aa` on Windows. Both include
the scoped ranking index and use the resident adapter. The candidate uses
WinHTTP for the existing loopback endpoint when its required controls are
available. It preserves the ordinary transport if native setup fails before
sending. This is an installed command comparison, not native Codex/Claude
acceptance, fresh semantic quality evidence or actual pilot use.

Use the same four-block design as the completed scoped ranking comparison:
baseline-0, candidate-0, candidate-1, baseline-1. Each block starts from a new
copy of the pristine 10,000-fact seed. Half 0 uses query indices 0 through 24;
half 1 uses 25 through 49. Each block runs three recall warmups and 25 timed
recalls, followed by one warmup and 25 timed commands for each of Stop and
UserPromptSubmit. Retain all 320 observations, including 300 timed samples.
Only one service/worker pair runs at a time.

Use the same isolated interpreter, 57 dependency versions, BGE model assets,
accepted facts, source evidence, scope, retrieval profile and context budgets.
Verify each installed package against its wheel. Freeze the protocol, drivers,
wrappers, package files, wheel, Windows HTTP DLL, dependencies, model assets,
manifest, settings, registry and prepared seed before timing. Record effective
settings and environment only as hashes. Preserve all earlier evidence.

Keep ordinary site initialization and the original private launcher's policy.
Run all three policy preflights in each block. Preserve the worker's current
settings, environment, directory and membership checks, durable outbox before
service IO, real query embeddings, full retrieval and host output. Verify each
candidate command imports the native transport and does not import Python's
HTTP parser. A candidate that silently falls back to that parser does not prove
the intended performance change.

Measure recall through the normal client response and each complete command
from before process launch through exit. Keep the 500 ms recall and 100 ms
complete-hook p95 requirements. Do not subtract components or omit work.
Compare decoded responses exactly and require identical full context strings.
Retain failures, warmups, raw responses and paired timings privately.

Use the existing Windows process ownership proof. After each block, stop the
worker and service, restart the service, and verify every source/job pair.
Record completed and pending jobs separately. Verify unchanged accepted facts,
evidence and vectors, SQLite integrity, frozen files and model assets, process
absence and descriptor removal. Forced termination fails cleanup. Independently
recompute sample counts, equality, medians, interpolated p95, maxima and paired
differences. Report per-block and pooled results with their timing limitations.

The native transport requires direct connections, disables redirects, cookies,
automatic authentication, keep-alive, pooling and failed-connection resends,
and validates the complete bounded response. Unsupported controls permit only
the ordinary transport before sending. These controls follow Microsoft's
[option documentation](https://learn.microsoft.com/en-us/windows/win32/winhttp/option-flags)
and [failed-connection retry contract](https://learn.microsoft.com/en-us/windows/win32/api/winhttp/ns-winhttp-winhttp_failed_connection_retries).
