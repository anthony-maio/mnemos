# Bounded native response reads

Canonical commit `54cb50e7c6c8548b918f62eb49d3f678a5a9e546` replaces one-byte HTTP header reads with bounded 1 KiB reads in the experimental native hook client. The previous path reset the socket timeout and read one byte for each header byte. The new path retains body bytes arriving in the same chunk, scans across chunk boundaries, and decodes only the header as UTF-8. A deterministic complete-response fixture now uses one read.

This reduces avoidable socket calls. It is not a measured complete-path latency improvement and does not establish the 100 ms capture target. No frozen paired comparison was consumed. Authoritative project resolution, live authorization checks, FULL durable commits and capture identity remain unchanged. The client remains an unpublished experiment; no host profile or installed pilot binary changed.

Review identified a deadline edge in the initial chunked reader: a fully prefetched body could return without another clock check. The final implementation checks the deadline after each socket read and before returning success. A controlled reader that returns a complete response after expiry is rejected. Read/framing failure after sending still produces unknown outcome rather than retry/fallback.

Validation on Rust 1.95.0:

- Six Rust tests passed: fragmented delimiter/coalesced binary body, ambiguous/oversized/truncated responses, exact header limit and maximum body, late complete response, slow header, and slow body.
- The final release binary passed all 75 launcher, profile and environment integration tests in 14.62 seconds. These include the existing worker and fallback checks.
- The earlier pre-deadline-fix binary also passed those 75 tests; its separate JUnit record is retained and is not evidence for the final deadline guard.
- Formatting, locked release build and diff checks passed. No dependency changes occurred.

Final binary SHA256: `b154c81345e4a7473086b946599baf2535daafc627133b1131984d7091914683`.
Final JUnit SHA256: `54283c4c11aa2210abd1049d2278a408b160b5bcb98230b3a5d5decf779c10c7`.
Private archive `.venv/native-hook-buffered-evidence-v1.zip`: `2fa68c7cf3f06d377b9a7aec6fd60903ab6b4ecf898377e5e3a6be53209a29bc`, eight members with manifest hashes verified after readback. It contains final source, README, Cargo metadata/lock, binary and both integration records. Rust test results are recorded above from terminal output; no separate raw Rust test log was retained.

The current complete capture result remains p95 376.71 ms on its older bound artifact. Git resolution and durable commit still dominate the measured service interval; this transport change cannot be treated as their solution. The private Python wheel remains bound to `25126632`; this commit changes only the separately built native experiment and its documentation.
