# Replay acknowledgement comparison preparation

Status: package preparation and owner functional preflight complete. See
[owner validation](../validation/2026-11-08/replay-ack-owner-preflight-v1.md).
No measured comparison has run. Execution still requires the completed runner/auditor, review and a manifest
binding their exact bytes, settings, schedule, dependencies and native binary.

Compare baseline `7a77e40f4ec8dd5b2fbf47ff0b1e3f0862e0129d` with candidate
`ec4a36b262be61c16d807a2b39006bcad7f8367c`. Wheel contents must differ only in
mnemos/client.py and RECORD. Use the same buffered native client in both arms,
SHA256 `b154c81345e4a7473086b946599baf2535daafc627133b1131984d7091914683`.
The earlier 376.71 ms attribution remains bound to its older artifacts and is
not the comparison baseline.

## Workload and counting

Retain the prior synthetic transport fixture's unavailable cognition provider,
simple embeddings, disabled experimental modules and disposable Git project.
Each arm owns a separate database, SQLite outbox and runtime descriptors. Keep
authoritative Git resolution, live registry checks, privacy enforcement, FULL
synchronous capture commits, scoped event identity and queued status intact.
The memory-service factory must use a fresh registry loader; do not copy the
older helper's caller-owned static registry as a substitute.

Use five warmup rounds followed by fifty measured rounds. Each round sends two
consecutive native Stop commands to an arm: a primer and its immediate successor,
without a replay-drain wait between them. Time both commands from process launch
through output and process exit. Each arm therefore has 100 measured commands,
plus ten warmups, and 110 unique logical events overall. Use identical logical
content/session/turn identities across arms, never duplicate IDs within an arm.
Alternate baseline/candidate and candidate/baseline order by round.

Drain the active arm before running the other arm, and both arms before the next
round. A completed barrier requires an empty outbox, completed jobs for every
acknowledged event and no replay acknowledgement I/O still running. Database
reopen checks and maintenance stay outside command timing. Do not add artificial
commit holds or sleep delays to produce overlap.

Record acknowledgement start/end intervals symmetrically and count actual
overlap with each command. Report the counts alongside all measurements. Little
or no overlap means the intended mechanism was not sufficiently exercised; it
does not authorize another schedule or a rerun.

These intervals cover the entire acknowledgement method, including lock waits;
they are not isolated COMMIT durations. The owner's response trace records
completed server dispatch before HTTP delivery. The auditor must combine it
with native command outcomes to establish a successfully delivered receipt.

## Audit and interpretation

Report primer, successor and all-command distributions separately, with
nearest-rank p95 and paired per-round differences. Preserve every scheduled
command, including failures, timeouts, unknown outcomes and unexecuted commands
after an abort. None may disappear into a success-only latency distribution.
Require every successful queued receipt to match reopened durable content and
its original event ID; verify eventual completed jobs and no duplicate events.

Freeze all inputs before the first scheduled command. Verify installed members,
helpers, native binary and selected dependencies before and after. Retain owned
process identities and require cleanup of both owners and their ports. Do not
retry a timed-out command or restart based on an observation timeout.

This is the remaining bounded implementation comparison, not another stage-only
attribution series. The deterministic held-commit regression justifies testing
the scheduling change, but does not predict that it meets 100 ms. The original
capture target is unchanged. A successful synthetic result still does not prove
real embedding-workload behavior, 10,000-fact recall, semantic quality, actual
host acceptance or pilot completion. Preserve those requirements separately.

Do not activate the identity cache, change live configuration, weaken FULL
durability, discard failed samples or relax a threshold based on this run.
