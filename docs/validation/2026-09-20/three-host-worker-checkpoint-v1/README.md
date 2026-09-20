# Three-host correction worker: offline reproduction and repair

The v4 runner could not satisfy its native correction-completion gate.
Its injected dispatcher factory bypassed the installed service owner's
background-maintenance startup. Correction only queued work, job reads did
not process it, and the phase did not permit consolidation. Controller
maintenance came after the required native completion observation.

This is a benchmark setup defect. The default production service already
starts its scheduler. It is separate from v4's rejected scope argument and
omitted inspection, which remain native model failures.

## Evidence

The new regression failed on the previous runtime: repeated job-status reads
never observed completion without controller maintenance. The repair passed
the same test, including same-ID/version correction, persisted state after
restart, a second correction after restart, terminated owner threads, and
removed runtime descriptors. It uses real service ownership and SQLite,
with a deterministic fixture embedder and no inference.

The repair constructs the isolated dispatcher through a benchmark subclass
of the installed default factory method. That reuses scheduler startup and
shutdown. This experiment polls every 0.25 seconds with no idle delay and
records those settings. It makes no production scheduling or latency claim.
The native completed-job gate, rejected-result checks, and inspection
requirements are unchanged. A status call can still observe pending; it
must not be counted as completion.

Verification: 167 combined offline cases passed, including 11 documentation
checks. Black passed on the three changed files; strict Python 3.10 mypy
passed on the two changed benchmark modules. These counts do not replace
the earlier product source/package totals. No product files or installed
package were changed. The [v5 protocol](../../../plans/2026-09-20-three-host-continuity-protocol-v5.md)
is prospective. Native v5 has not run at this checkpoint.

The archived failed and passing JUnit records are separate observations.
The evidence index binds raw files; this README and the index are outside
that index. The writer packet v21 predates this offline repair and remains
an immutable working snapshot through native v4.
