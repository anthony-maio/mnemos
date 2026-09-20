# Hermes rewind through the capture boundary

The existing [durable identity compatibility patch](../../../integrations/hermes-durable-message-identity.patch)
passes both actual CLI undo and retry methods through Hermes MemoryManager
and the verified Mnemos `4bbab4f` wheel's queued-outbox adapter. The same
fixture fails both cases on the cwd-only host. Both isolated host checkouts
are based on `14a346345486418aa85ed6526549412ee2665ade`.

The [protocol](protocol.md) and runner were committed at `3359155` before
either run. Both reports, source hashes, complete tracked host diffs, and
executed runner bytes are retained. Neither arm attempted a network connection
or changed the inspected host source files. The base command exited 1; the
candidate exited 0. Original JSON reports are the raw result artifacts;
terminal stdout was observed through the execution tool and is not included
as a raw log.

| Phase | Cwd-only host | Identity candidate |
| --- | --- | --- |
| Initial native capture | 4 queued observations | 4 queued observations |
| Duplicate replay | Same 4 | Same 4 |
| ID-less warm rewind, identical regenerated turn | 8, including 4 provisional receipts | 6 native observations |
| Provider restart and durable reload | Not reached after failure | Same 6 |
| Shortened retained view | Not reached after failure | Same 6 |

These results hold separately for `/undo` and `/retry`. The candidate retains
row IDs 1 and 2; identical replacement messages receive IDs 5 and 6. Original
IDs 3 and 4 remain historical observations. All four original queued payloads,
including timestamps, remain unchanged. Both later replays preserve all six.

The unpatched failure here differs from the earlier fully ID-less
[rewind diagnostic](../../2026-09-19/hermes-rewind-diagnostic-v2/README.md).
This fixture starts with native captures, strips IDs only from the warm CLI
view, and supplies native IDs on the regenerated suffix. Missing IDs on the
surviving prefix make that later snapshot provisional. The earlier diagnostic
instead reproduced identical prefix-hash collisions when every message lacked
IDs. Neither result is overwritten or relabeled.

Run `python verify.py` from this directory to verify all 19 indexed artifacts
and inspect both acceptance outcomes independently of their reported labels.

This closes the candidate's host-to-adapter queued-capture boundary check.
The turn content and new-turn persistence are fixture-driven. No native model
conversation, HTTP service, extraction, process crash, or live-host upgrade
was tested. Existing ID-less capture migration and compaction clone provenance
remain open. The patch is still isolated and has not changed live Hermes.
