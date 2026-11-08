# IPv6 loopback dispatch

The client accepts `::1` in a runtime descriptor, but dispatch previously formed
`http://::1:43123/v1/dispatch`. HTTPX rejected this as an invalid port before
transport. Readiness already used the correctly bracketed authority. Dispatch
now uses the same bracket form while retaining the existing loopback allowlist.

This candidate is based on canonical commit
`a7c6dcc8f65bb4fb010dd847eb72818b0ccfccd9` in the isolated
`wt/mnemos-ipv6-dispatch` worktree. The service currently binds IPv4 loopback;
this repair does not introduce an IPv6 listener or explain the measured capture
latency failures.

Before the fix, the six new regression cases produced one failure and five
passes. The `::1` case failed with `httpx.InvalidURL: Invalid port: ':1:43123'`.
After the fix, all 22 selected cases passed in 7.19 seconds:

```text
python -m pytest tests/test_client_loopback_dispatch.py tests/test_client_stale_descriptor.py tests/test_client_startup_failure.py -q --tb=short
```

The new cases exercise real HTTPX URL construction with a mock transport. They
verify the target host, port, path, authorization header and method for IPv4,
localhost and IPv6 loopback, and reject non-loopback or already-bracketed
descriptor hosts before transport. They do not establish a live IPv6 socket
connection. The existing cases cover stale descriptors and startup failure.

Black passes for the changed source and new test file. Mypy passes for
`mnemos/client.py` with `--follow-imports=silent`; this is a scoped check, not a
full dependency type check. `git diff --check` passes. The existing pytest-asyncio
unset fixture-loop-scope deprecation warning remains.

Canonical delivery remains clean at its original commit. No installed package,
live configuration, memory database or frozen evaluation artifact was changed.
This is a locally tested candidate, not a published fix, native-host result,
latency pass or pilot day. Optional pilot activation remains unapproved.
