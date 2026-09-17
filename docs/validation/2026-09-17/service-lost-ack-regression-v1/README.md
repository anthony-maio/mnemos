# Service lost acknowledgement regression

This archive records a bounded HTTP and SQLite regression for a capture that is
committed by the service before the client receives its response. The test
drops one or two committed responses at the HTTP reply boundary, restarts the
service, replays the durable outbox entry, and checks that the event and job
IDs remain stable and that no duplicate mutation appears. It covers both a
normal extraction job and a provisional capture-only job.

The finalized v2 run was intentionally red against the client at source
`b346f9318e3a8c602736d9422e9223d7a04d8c8c`: all four cases reached the
uncaught `httpx.RemoteProtocolError` caused by the simulated lost response.
The early v1 test also contained an unconditional provisional-job assertion;
that assertion was corrected before v2, while v1 failed earlier at the same
HTTP exception boundary. The client fix was then applied at
`6cde46561c0c42c0d6124d86a09d1e2832542c9d`, and the same four cases passed in
the preserved green XML.

The focused suite XML contains 35 passing cases. The service, SQLite, and
client are real local components with offline providers and isolated scopes.
No native host, model-quality, or production-network claim follows from this
regression. The test uses an injected reply failure to model an acknowledgement
lost after durable commit; it does not establish why unrelated native host
connections may fail.

All XML and source files in this directory were copied byte-for-byte from the
executed worktree artifacts. `provenance.json` records hashes, source commits,
commands, basetemp, and JUnit counts.
