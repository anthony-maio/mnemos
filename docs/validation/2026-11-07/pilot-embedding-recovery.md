# Installed pilot embedding recovery

The installed `6d0e5ba6` pilot artifact passed one controlled embedding-failure
and recovery check. Two explicit controls were first stored with real BGE-small
vectors. The test then injected MnemosTransientError at the provider's vector
boundary. One new explicit write and one raw capture were durably accepted;
both processing jobs remained pending with released leases, refunded attempts,
recorded provider errors and the ordinary 15-second backoff. No new fact was
created during the fault.

Existing project memory remained searchable with lexical-only degradation and
no semantic score. Queries for the foreign control and the unprocessed marker
returned no facts. After stopping the instrumented service, ordinary installed
client auto-start launched the normal service with real FastEmbed and background
maintenance. No fault wrapper, synthetic vectors, scheduler disablement, job
rewrite or clock advance was used during recovery. Both jobs completed after
the normal backoff: one explicit accepted fact and one unasserted raw chunk.
Completed jobs retain their earlier provider-error history. Their completed
status and committed results establish recovery.

Semantic recall returned the explicit fact and excluded the raw capture. Both
controls retained their identity, content, metadata, vector and version. Replaying
the exact two write payloads and restarting again left all event, job, chunk,
evidence and revision rows unchanged. SQLite integrity and foreign keys passed.
Both ordinary service children exited zero, the descriptor was removed, the
ownership lock was available, and no service remained for this disposable root.
The inactive pilot's source files stayed unchanged; it still has no database
or runtime directory.

The [receipt](pilot-embedding-recovery-v1-receipt.json) binds runner `2548cca6`,
the unchanged wheel and launcher, two real seed embedding calls, five injected
exceptions and the original 19.14-second attempt. Twenty-two existing recovery,
maintenance and degraded-retrieval checks passed before execution; the runner
passed formatting and strict types. The [prospective protocol](../../plans/2026-11-07-pilot-embedding-recovery-v1.md)
was committed before the check. All 82 private archive members verified;
archive SHA-256:
`b56e95182ae12b9d3b22211028ec8a4e43ea996ce5e92738460cf417a524be78`.
An independent audit initially used the wrong evidence column name; its failed
helper and output are preserved with the corrected source-link verification.
The installed attempt was not repeated.

This establishes recovery from the injected typed exception on this artifact.
It does not establish how every native ONNX error is classified, provider
initialization failure, a naturally occurring outage, native hook deadlines,
mid-write crash behavior, extraction quality, repeated reliability or daily use.
No native model was invoked, no live host configuration changed and no pilot
day was counted.
