# Startup exit diagnostics

This archive records the deterministic client startup regression added for an
autostart child that exits before publishing a runtime descriptor. The test
also covers a live child that never publishes a descriptor, durable capture
queueing on startup failure, an existing owner descriptor, a child that becomes
ready, and a descriptor appearing during the terminal-child race.

Against the unchanged client baseline, the final v3 run had six cases: three
primary failures and three compatibility controls passed. The failures were
the generic descriptor-unavailable result for an exited child, the same
generic diagnostic on a queued capture, and the generic result for a live
child reaching the configured virtual startup deadline. The existing-owner,
ready-child, and descriptor-race controls passed. The v2 XML is preserved as
an intermediate run: its race fixture depended on polling that the unchanged
client did not perform, so it recorded four failures. The v3 control lets the
descriptor appear during the virtual wait while still exercising the race in
the corrected implementation.

The v1 run is preserved separately and has five cases with three failures. The
green v1 XML records the later focused run with 35 passing cases after the
client fix at commit `a607a5a6b0a8c2d4f0c4ed00e3b21167cf96f854`. The baseline
source commit was `df52200783cbfc9ed054a510de1765a7267352e8`.

The six startup cases use a local asyncio shim and virtual clock. They do not
patch the shared event loop, sleep in real time, launch a model, contact a
network, or modify a service. The green focused command also includes 29 real
local service cases using HTTP. Black was checked on the startup test and
strict mypy was checked on the client. This archive does not claim that the
earlier full-suite startup
failure was caused by an exited child; the isolated failure mode is now
reproduced, while that original run had no preserved runtime directory or log.

The four XML files and final test source were copied byte-for-byte from the
executed worktree. `provenance.json` records hashes, commands, basetemps,
JUnit counts, and the source baseline/fix binding.
