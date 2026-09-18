# Capture import boundary

The previous f201653 source failed all three subprocess import guards: importing the package, client, or Codex adapter initialized embedding providers. The compatibility case passed. After deferring public exports and the scoring-only cosine import, 92 focused tests passed with two platform skips. This includes the four import/API cases, engine, retrieval, host rendering, and interrupted capture. Strict mypy passed for all 50 product source files.

The change preserves public names, class identity, star imports, and dir(). Public objects resolve on first access; callers inspecting only the package __dict__ before any access will no longer see every export. Service execution still loads its providers normally. No hook timeout changed.

Earlier installed-package timing probes are preserved, including negative evidence. The first measured adapter imports at about 730-757 ms. Stop v2 completed six captures, but its final reopen assertion failed because SQLiteStore is not a context manager. Its first generated PowerShell command took 9.7 seconds. Stop v3 fixed that assertion and reopened six durable events; generated commands took 4.7-8.6 seconds. A broad read-only file search overlapped v3, so those timings are load-confounded. These are diagnostic fixture observations, not native host acceptance or before/after performance proof. Shell startup and child execution both showed delays. The native three-second deadline remains unresolved.

The XML preserves the pre-repair failure and focused post-repair results. Probe copies are immutable, with .py.txt suffixes to exclude archived programs from static checking. No live host configuration, evaluation snapshot, or memory data changed.
