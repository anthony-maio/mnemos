# Native hook launcher experiment

This executable tests the process and import cost of the existing resident hook
client. It is not included in the Python package or enabled by generated host
configuration. The worker still resolves project identity, checks current
membership and settings, and performs durable capture and retrieval.

Build from the repository root with Rust 1.95.0, the compiler used for this
checkpoint. Dependencies are pinned in `Cargo.lock`.

```powershell
cargo build --locked --release --manifest-path benchmarks/native_hook/Cargo.toml --target-dir .venv/native-hook-build
$env:MNEMOS_NATIVE_HOOK_EXE = (Resolve-Path .venv/native-hook-build/release/mnemos-native-hook-experiment.exe).Path
python -m pytest tests/test_native_hook_launcher.py tests/test_native_hook_profile.py tests/test_native_hook_environment.py
cargo test --locked --manifest-path benchmarks/native_hook/Cargo.toml --target-dir .venv/native-hook-build
```

The test suite uses disposable descriptors, loopback servers and capture queues.
The optional test module skips when `MNEMOS_NATIVE_HOOK_EXE` is unset. Its checks
do not establish installed performance, native-host acceptance or pilot use.

The explicit development command accepts `HOST EVENT --python ABSOLUTE_PYTHON`.
It reads the original hook JSON from stdin and the existing
`MNEMOS_HOOK_WORKER_DESCRIPTOR`. Missing, mismatched or unavailable workers allow
the ordinary Python adapter to receive the complete original input. A valid
worker refusal permits the same fallback. Any uncertain outcome after sending
ends without another dispatch. Only fixed diagnostic metadata reaches stderr.

Private configuration uses `HOST EVENT --profile ABSOLUTE_JSON
--profile-sha256 SHA256` instead. The reviewed command pins the entire JSON
profile. Schema version 1 contains `host`, absolute `python`,
`fallback_launcher`, `worker_descriptor`, `fallback_launcher_sha256` and an
`environment` object containing the reviewed uppercase `MNEMOS_` settings.
The descriptor has its own field and cannot also appear in `environment`.
Do not put credentials in this file. Generate its settings from the reviewed
launcher's environment setup and verify parity before use.

This mode removes inherited Mnemos selectors, retains unrelated environment
values and overlays the pinned settings without mutating process-global
environment. Its safe fallback runs the pinned launcher as
`PYTHON -I LAUNCHER hook HOST EVENT`, preserving the full original input.
The launcher still parses effective settings and applies its original guard
before invoking an adapter. Changed profiles or launchers produce a sanitized
error without dispatch. The launcher hash is checked again after a worker
decline. There is no ordinary-adapter fallback from a guarded profile.

The protocol remains version 1. Environment hashing matches Python's sorted
ASCII JSON representation, with Windows environment keys normalized as CPython
does. Windows hidden drive-directory entries such as `=C:` are omitted, matching
Python's `os.environ` even when an implicit child environment retains them.
Ordinary environment mismatches still refuse dispatch. Integer payloads retain
arbitrary precision. The transport uses one TCP
connection to literal loopback, with no proxy, redirect, authentication
negotiation or automatic retry. The response must bind the request ID, host,
protocol and environment, and contain the original typed acknowledgement.
Headers and body are bounded; header and body reads share the original
30-second operation deadline. The worker's fixed Content-Length response is
required; chunked or ambiguous framing fails without fallback.

Rust's [TCP timeout API](https://doc.rust-lang.org/std/net/struct.TcpStream.html)
provides per-operation timeouts; the executable recomputes the remaining total
deadline before each operation. Windows descriptor reads retain the standard
[read/write/delete sharing](https://doc.rust-lang.org/std/os/windows/fs/trait.OpenOptionsExt.html#tymethod.share_mode).
No unsafe Rust is permitted in this executable. Other platforms have not been
runtime-tested at this checkpoint.

Run a frozen comparison before claiming a speedup. A loopback transport probe
does not exercise Git scope lookup, durable capture, retrieval, host trust or
complete hook latency. The complete installed and native-host checks remain
required before delivery or activation.


## Optional stage timing build

Build with `--features stage-timing` into a separate target directory to emit one
`native_stage_timing_v1` JSON record on stderr after the hook finishes. Normal
builds have no timing collection or output. The feature changes no environment
selector, descriptor binding, request format or stdout response.

Marks use elapsed milliseconds from entry into main and fixed names only. They
cover launch preparation, stdin parsing, descriptor validation, serialization,
connect, completed request write, first response byte, complete headers, body
read, response validation and completion. Early failures have partial traces;
fallback has its own start mark. Traces contain no request IDs, source text,
paths, tokens or environment values. Failed diagnostic writes are ignored.

Process creation before main, final trace serialization/write and process exit
are outside these marks. Compare them with an outer command timer; do not call
inside-main timing complete command latency. Request-to-first-byte time combines
server work and transport. Worker-side evidence is needed to separate them.
Instrumentation can change scheduling and adds overhead, so this build is for
diagnosis rather than acceptance timing. Keep the original binary and results.

Set `MNEMOS_NATIVE_HOOK_TIMING_EXE` to the diagnostic build to run
`tests/test_native_hook_timing.py`. Keep `MNEMOS_NATIVE_HOOK_EXE` pointed at a
normal build for the existing launcher, profile and environment suites.
