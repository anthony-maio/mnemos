# Hermes pre-compression source snapshot candidate

The isolated candidate persists and verifies original source rows before
compression, then advances the lease watermark to include its own writes.
That prevents those writes from being copied again as a concurrent tail.
The first watermark interaction failure and all recorded repair attempts remain
in this archive. The carrier repair separates derived and retained components
even when compression produces an assistant carrier.

The v21 focused selection passes 23 tests. The broader preceding v20 selection
passes 114 and fails one: a marker test supplied unrelated messages instead
of its seeded database transcript. V21 changes that fixture to read the actual
stored transcript; its original count and marker assertions remain unchanged.
A separate new test rejects an unrelated transcript without writes or stamps.
This is a documented fixture change, not an unchanged-baseline pass.

The complete SessionDB selection records 1,416 passes, 11 skips, and one failure
in test_apply_wal_concurrent_connects_no_eio. The same named failure was observed
on the earlier untouched baseline; this run does not independently establish
its cause. It is preserved and the broad gate remains open.

Two additional provenance risks await runtime reproduction: unmarked generated
compression text may receive original-source authority, and an edited hybrid
component may retain a stale ancestor reference. Neither is resolved by these
test counts. No live host installation, service delivery, complete capture,
or T17 acceptance is claimed. The tracked patch and current changed sources
are included alongside the new snapshot helper and focused test files.
