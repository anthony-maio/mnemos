# Native launcher removes most measured command-boundary cost

The experimental native client reduces median command-plus-loopback time from
171.40 to 31.86 ms in this development probe. All 44 authenticated dispatches
return the same full decoded output. This supports a complete installed
comparison; it does not establish complete hook performance or native-host
acceptance.

## What changed

Source `1facb81acdb3e53bc1af64829ba02bf4780939e1` adds a separate Rust executable
under `benchmarks/native_hook`. It is not included in the Python package or
enabled in host configuration. Production files still match `3d1866d5`.

The executable uses the original resident worker protocol and environment hash.
It validates the descriptor and bound acknowledgement, sends once to literal
loopback and requires a bounded Content-Length response. Missing or mismatched
workers, a connection refusal before sending, or a valid worker decline permit
the ordinary Python adapter fallback. Any uncertain result after sending ends
without another dispatch. UTF-8 input, arbitrary-precision integers and decoded
host output are preserved. Diagnostics contain fixed metadata only.

The first build used ordinary numeric parsing. A stronger input test showed
that it rounded `2**90 + 1`; both host variants failed. Enabling arbitrary
precision fixes the issue. The original binary and failed checks are retained.
The earlier 44-check pass is also retained; its power-of-two control did not
expose rounding and is not used as evidence for the corrected integer contract.

There are 50 native subprocess checks and 62 related hook/transport checks, all
passing. They cover environment hash parity, Unicode and integer preservation,
invalid descriptors, response binding, lost acknowledgements, redirects,
authentication challenges, malformed/oversized framing, safe fallback and
sanitized errors. A real disposable worker check retains one queued capture
across worker restart and repeated delivery of the same event. Its memory
service is unavailable, so this is outbox persistence, not a completed-job claim.
Two Rust tests prove that slow header/body delivery cannot extend the total
deadline. Clippy passes with warnings denied; Rustfmt and Black pass.
All 12 documentation consistency checks also pass.

## Frozen comparison

The Python arm uses the installed `3d1866d5` wheel and Python 3.12.11. All 96
installed package files match the artifact, SHA-256
`253ca806dba5aa817bd7ddcba1af123535b29853aa211fee61a14f71799a62b8`.
The native arm was built with Rust 1.95.0 from the locked dependency graph.
Executable SHA-256:
`65d920e4d57c3b07148ebda0f159bea9546129bb2467185fc438395bb7dda82a`.

Four command arms ran in rotating order, with two warmup rounds and 20 measured
rounds. Each arm occupies each position five times during measurement. The
complete timer includes process launch, command execution, output collection
and exit. No startup cost is subtracted. There are 88 commands, of which 80
are measured, and exactly 44 authenticated requests including warmups.

| Command boundary | Measured samples | Median ms | Nearest-rank p95 ms | Maximum ms |
| --- | ---: | ---: | ---: | ---: |
| Python, empty event | 20 | 149.22 | 262.85 | 263.93 |
| Native, empty event | 20 | 29.04 | 140.08 | 140.49 |
| Python, loopback dispatch | 20 | 171.40 | 214.47 | 763.73 |
| Native, loopback dispatch | 20 | 31.86 | 36.46 | 36.89 |

Native is faster in all 20 matched empty-command pairs and all 20 dispatch
pairs. Median paired changes are -118.44 ms and -138.53 ms, respectively.
The empty native arm still has a high tail. These small, rotating samples do
not establish a platform-wide startup bound or identify the cause of outliers.

The responder validates the complete envelope, authentication, unique request
ID, environment, event, working directory and exact synthetic payload. It then
returns a fixed context. It is a protocol fixture: no memory worker, Git scope
lookup, capture transaction, query embedding or retrieval runs. Empty commands
make no request. Every dispatch produces identical decoded host output, with
no fallback, timeout, retry or omitted observation.

## Audit and next gate

The independent audit recomputes ordering, summaries and paired differences,
matches every dispatch to its recorded request and verifies all 107 frozen
bindings. All 88 recorded command processes have exited, with no reused PIDs
found. The responder stops, its port closes and the private descriptor is
removed. Source, executable and installed package bytes remain unchanged.
No inference or memory-service call occurred.

The verified private archive contains 128 members, 1,499,612 bytes and SHA-256
`99a59355600244b3bc44c8391eb63aea7c4bd5f6be4c9046603a1342c722c79f`.
It includes raw observations, requests, frozen bindings, current and initial
binaries, the numeric regression evidence, checks and audit helpers.

Before the complete comparison, integrate the reviewed private launcher's
environment setup and guarded fallback. The prototype currently invokes the
ordinary adapter directly on fallback; it does not replace the private
launcher's additional configuration checks. Preserve those checks and verify
their failure behavior before using private pilot material.

Then compare complete installed paths on the unchanged 10,000-fact seed with
real embeddings, original scope/membership/settings checks, durable captures,
restart verification and complete returned context. Keep the 100 ms complete
hook and 500 ms recall requirements unchanged. Git resolution and remaining
worker work still contribute to those paths. The earlier failed complete-hook
results remain authoritative until the new comparison runs.

Fresh quality, native-host validation, final packaging and delivery, and actual
private daily-use pilot work remain open. The pilot is inactive with zero days.
No production default, live host configuration or publication state changed.

- [Frozen probe protocol](../../plans/2026-11-07-native-hook-launcher-v1.md)
- [Aggregate receipt](native-hook-launcher-v1-receipt.json)
- [Experimental executable and build instructions](../../../benchmarks/native_hook/README.md)
- [Previous complete Windows transport comparison](native-hook-performance-v1.md)
