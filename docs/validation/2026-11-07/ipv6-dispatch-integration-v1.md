# Canonical IPv6 dispatch integration

Canonical delivery now includes the IPv6 loopback URL repair at
`4966757755d32c60a287dc4404b195cc6a2bfecd`. The local branch advanced by
fast-forward from `a7c6dcc8f65bb4fb010dd847eb72818b0ccfccd9` with no conflicts.
Its working tree was clean before and after integration, and its product,
tests and package metadata match the tested candidate exactly.

The change touches only `mnemos/client.py`, the new loopback dispatch regression
tests and their validation report. Dispatch now brackets `::1`, matching the
existing readiness path. Before the fix, one of six regression cases failed
with HTTPX's invalid-port exception. The candidate passed all 22 selected
dispatch, stale-descriptor and startup-failure cases, Black on both changed
Python files, and scoped mypy with imports followed silently. Tests were not
repeated after the exact fast-forward. This is not a new full-suite result.

Before integration, a Windows process inventory found no process whose name
matched Python, mnemos, Node or uv and whose command line referenced the
canonical checkout. This limited scan does not prove that every possible
external consumer was absent. Integration did not restart any runtime or
change installed packages, host configuration or databases.

The original validation report in canonical delivery records the pre-integration
state. This report records the subsequent local integration. Neither establishes
a live IPv6 listener: the service still binds IPv4 loopback. The regression
uses real HTTPX URL parsing with a mock transport.

Package version remains `0.7.0b1`; nothing was tagged, pushed, published or
installed. The optional pilot remains bound to its previously validated
`f5a944ea` wheel, not this newer source. Its activation question is still pending.
The improvements worktree's product source and all frozen evaluation evidence
remain unchanged. Quality, complete-command latency and daily-use requirements
remain open.
