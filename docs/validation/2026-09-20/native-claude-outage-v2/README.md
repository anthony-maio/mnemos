# Native Claude service-outage recovery v2

The single attempt on September 20, 14:53:29-14:53:38 UTC, passed
`native_service_outage_raw_capture_observed`. It used verified product
`a64afb7`, runner `6918305`, Claude Code 2.1.191, and `claude-sonnet-4-6`.
The [package checkpoint](../autostart-environment-checkpoint-v1/README.md)
records 1,834 source passes, 703 installed passes, four skips in each suite,
six source/package checks, and seven replay assertions.

The service was stopped before the native conversation, with its descriptor
absent and former port closed. Claude produced the exact requested reply
without tool calls. Its two native messages reached the durable outbox while
the service remained stopped. The outbox contained three entries: two normal
message identities and one separate provisional prompt capture.

After the disposable service restarted, an ordinary client request drained
the outbox. Native text reconciliation matched both messages. Independent
SQLite reads verified the raw events, physical scope and source identities,
three pending jobs, zero facts, and integrity. Replaying the same payloads
and restarting again preserved those records. The outbox was empty afterward.
All three recorded health responses identified the expected disposable
database and stopped background scheduler. Cleanup passed.

All five modules were disabled. Nomic embeddings were configured, but no local
provider responses occurred. Pending extraction is not completed memory.
This one observation does not prove provider-outage or in-flight crash
recovery, other hosts' outage behavior, Desktop support, or daily-use reliability.
It does not erase the [v1 failure](../native-claude-outage-failure-v1/README.md)
or its default-database writes. No cleanup deletion of those rows was performed.

Run `python verify.py` from this directory to check the 24 indexed artifacts,
bound source bytes, native transcript and model/session identities, physical
raw-source state, receipt matching, replay, and restart observations. The
verifier makes no network calls. Cleanup and continuous outage assertions
remain recorded execution observations; an offline verifier cannot observe
past process state anew. This README and the verifier are outside the index.
The full default database and its private backup are excluded.
