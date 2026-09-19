# Hermes lifecycle regressions on the preceding installed wheel

Six new regression cases were copied outside the source checkout and run
against the installed `78c63ed36942848c95c1acd93b805fe0e0d13a67` wheel.
All six failed, with no collection errors or skips. The preserved tests, log,
JUnit results, command, and helper establish the earlier behavior:

- A growing transcript recaptured two earlier messages: six queued records
  for four actual transcript entries.
- Identical messages at different positions collapsed: two records for four
  transcript entries.
- Compression created another copy of the completed turn.
- Recreating the provider changed timestamps and conflicted with its pending
  durable outbox payload.
- Native lifecycle tools were absent.
- The uninitialized-provider tool boundary was absent.

These are deterministic adapter/outbox regressions, not model quality or
native conversation evidence. The current repair uses transcript-prefix
identities, persistent first-observation timestamps under Hermes home, and
explicit scoped lifecycle tools. Additional review added provisional handling
when a callback lacks transcript identity and stable request IDs for explicit
writes. The subsequent concurrent-ledger test also exposed a first-use lock;
initialization now reserves the SQLite writer before schema access, with a
bounded wait. Full package and native-host verification must bind their own
source commit rather than inherit a pass from this reproduction.

`evidence-index.json` binds seven raw artifacts. This README is outside the
raw index. The archived tests predate later additions to the regression suite.
