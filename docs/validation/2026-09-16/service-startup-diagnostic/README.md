# Service startup diagnostic

The old and fixed runs use the same comparison runner and fresh SQLite
fixtures. The old run is bound to
frozen source `2360891b23f68606092add4593cd9c29d303ee1a`. The fixed run is bound
to frozen source `9e253390c58302817a18b6504f55899b36a2c27b`. Both imported the
pinned checkout, used simple embeddings and an unused loopback Ollama setting,
and made zero non-loopback calls. Each allowed one loopback socketpair for
Python's Windows asyncio event loop. The service descriptor and owner thread
were gone after cleanup.

The old child failed its five-second startup deadline with a `TimeoutError`.
Child startup duration was `6,967.675 ms`; schema initialization took
`4,774.106 ms` and dispatcher construction took `6,322.894 ms`. The first
`MEMORY_CHUNKS` CREATE ran outside a transaction. The trace shows DDL with
`in_transaction_before=false` and `in_transaction_after=false`, consistent
with standalone DDL autocommits before migration DML opens a transaction.
This identifies the transaction shape
seen in this fixture without claiming that every delay has the same cause.

The fixed child passed startup in `2,453.355 ms`. Schema initialization took
`776.631 ms` and dispatcher construction took `2,442.728 ms`. The first
`MEMORY_CHUNKS` CREATE ran inside the transaction opened by `BEGIN`. The fixed
child read back `journal_mode=wal` and `synchronous=2` after startup. The trace
also records the expected WAL and FULL PRAGMA statements.

These are one fresh startup per source snapshot. They show a change in this
fixture and environment. They do not establish general startup performance or
explain every operating-system delay.

The exact comparison runner is preserved at `comparison-v2/runner.py.txt`.
Its SHA256 is
`1858a8388d6a7af7f34408c8f82a8b11280b89eee16bf1ceeac69e3657843014`, matching
both comparison reports. Complete copied reports and traces are under
`comparison-v2/old/` and `comparison-v2/fixed/`.

Earlier pinned helper attempts are preserved under `harness-history/`:

- `service-startup-diagnostic-pinned-v1` failed before child startup because
  `datetime` was missing.
- `service-startup-diagnostic-pinned-v1b` failed before child startup because
  the child received a relative attempt path after changing its working
  directory.
- `service-startup-diagnostic-pinned-v1c` imported frozen source, then the
  strict socket patch blocked Python's internal loopback socketpair.
- `service-startup-diagnostic-pinned-v1d` imported frozen source and reached
  SQLite startup, then reproduced the timeout. Its pre-Black runner bytes were
  not preserved, so its recorded runner hash is historical rather than proof
  of byte identity with the current helper.

The old storage regression proof is preserved under `regression-old/`. All
three failure-injection tests failed against frozen236: the RuntimeError and
KeyboardInterrupt cases left 11 schema objects, and the migration case changed
the prior update trigger. The saved runner invokes `pytest.main`; the report's
command field describes the equivalent pytest arguments. The first helper
attempt failed to import pydantic before pytest ran. Its output and original
runner bytes were not preserved as separate artifacts.

The original working-checkout diagnostic is preserved under
`working-checkout-v3/`. It used the working checkout and recorded only four
source hashes. Three matched source236 after line-ending normalization while
storage editing had already started before the fourth comparison completed.
That is weaker source binding than the frozen comparison.

The original helper inherited unspecified environment variables, declared its
network and durability settings without independent enforcement, and used
`elapsed_ms` for both absolute event offsets and operation durations. Its
`started_at_utc` value was populated after execution. It did not explicitly
protect the fixture directory. It created no native host credential copy, and
the timed-out service never published a descriptor. Those limitations remain
part of the original evidence and are why its timings are kept separate.
