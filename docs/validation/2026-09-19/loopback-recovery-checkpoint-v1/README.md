# Loopback recovery and cognition setup checkpoint

Verified source `de6cdcf1c05fdad1bb19ed8f52638eff5ed67ae4` includes runtime repair `93a109a` and the requested
Substack research/writing handoff. It passes 1160
source tests with 4 skips, Black, configured strict mypy,
JavaScript syntax, build and Twine. The isolated wheel passes
474 contract cases with 2 matching skips
and seven synthetic replay assertions. Package origins and per-case outcomes
are bound in the archived installed reports.

The previous `d0a3440` full suite failed one owner-recovery test: 1,150 passed,
one failed, four skipped. Five fresh-process repetitions reproduced two
failures with AsyncClient construction consuming 265-268 ms before the
250 ms health probe could contact the healthy owner. A separate 100 ms
construction-delay control also failed. Those raw results remain here.

Local HTTP routes now use a verifying SSLContext without loading certificate
bundles. They still restrict hosts to loopback, ignore environment proxies,
disable redirects, and send authenticated requests. Certificate verification
is not disabled. An exited competing launcher no longer ends readiness polling
for the unchanged owner before the original startup deadline. Missing
descriptors and authorization/schema errors retain their failure behavior.
Three regression cases failed before the repair and passed afterward. The
focused service/host run passed 51 tests with two platform skips. Review found
no blockers.

The next full run at `a6c275b` passed 1,153 tests but failed descriptor
publication when a competing Windows reader held the old file open. The
held-reader regression reproduced the exact WinError 5. Source `de6cdcf`
retries only atomic replacement for Windows errors 5, 32 or 33, for at most
250 ms and no longer than the remaining service startup budget. It never
deletes the old destination or changes permissions. A persistent lock keeps
the original bytes and removes the temporary file; unrelated errors are
immediate. Two new cases failed before the repair. Focused recovery, service
and Windows-permission tests passed all 44 cases afterward. Both failed full
runs are archived alongside the successful successor.

The unchanged recovery test at `a6c275b` passed all five new repetitions, plus the separate
100 ms delay control. Uninjected health probes took 37.630-49.759 ms and full
requests took 48.461-63.012 ms. This is a bounded diagnostic, not a guarantee
that cold imports or arbitrary system contention fit a 250 ms wall-clock bound.
The final checkpoint has byte-identical client code; its additional service
publication repair is covered by the new full and installed checks.

This checkpoint also verifies the earlier cognition-provider change from
`d0a3440`: reuse the default verifying trust context per provider while keeping
request clients bound to their own event loops. Custom SSL_CERT_FILE or
SSL_CERT_DIR settings bypass the default cache. The unchanged-source HTTP/1.0
fixture measured warm median 234.416 ms at `8fc79f0` versus 6.893 ms at
`d0a3440`, with 12 warm calls per arm and separate asyncio.run per call.
Cold calls were 303.906 and 244.903 ms. Both arms opened and finished all 13
connections. The earlier injection-based prototype is separately retained.
These measurements do not include Ollama inference or TLS handshakes and do
not establish full recall latency acceptance.

The failed full source run, diagnostics, raw logs, JUnit reports and helper
bytes are preserved. Build distributions and disposable databases remain at
their recorded external locations. Real-model lifecycle completion, native
host gaps, latency gates, the pilot and final claims remain separate work.
