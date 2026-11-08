# Recovered episode identity integration

Canonical `036357fe4ddd6f13507ca26d37933d0b1b682886` integrates the recovered
local 0.7.0 episode repair and the profile-safe adaptation required by delivery.
The source is pushed to `wt/backup-20261108-delivery`. Original dirty source,
live memory and existing packages remain unchanged.

Interaction now has a serializable ID. New manual interactions with identical
text and timestamps remain distinct; replaying the same serialized interaction
does not create another episode, including after SQLite reopen. SleepDaemon
still applies the write firewall first. It appends to its transient buffer only
when the store returns the newly inserted episode, so duplicate or failed
writes do not create buffered copies.

The canonical adaptation also fixes a reproduced profile collision. New sleep
fingerprints include opaque profile and scope identities. Structured encoding
applies to every profiled event and every effective non-hook interaction ID.
Review found that leaving unprofiled IDs in a delimiter-joined string allowed
id=one|two/content=three to collide with id=one/content=two|three. Both backend
regressions failed before that repair and pass afterward.

The original four recovered cases failed on canonical source. Added profile
and write-failure coverage expanded the initial reproduction to nine failures.
Final validation spans ten files: 156 tests pass in 28.91 seconds, including
type, module, SQLite, memory-safety, related-forget, source-snapshot and support
version checks. Mypy accepts the three runtime modules; Black and diff checks
pass. Review confirmed the fingerprint correction and found no remaining scoped
blocker. The reviewer did not execute tests.

Old rows are not rewritten. Historical callers without a profile or effective
non-hook interaction ID keep the old fingerprint format. An old serialized
interaction lacking an ID acquires a new identity when reconstructed; this is
not transparent retry migration. Preserve Interaction.id across retries. Legacy
claude_hook sleep ingestion retains content-based deduplication within its
profile; it still does not establish native occurrence identity. Functional
capture, replay suppression and authorization contracts are not replaced by
this experimental-path repair.

Private archive `.venv/canonical-episode-tests-v2.zip` contains 12 red,
intermediate and final test logs/JUnit/type-check files plus a verified manifest.
SHA256: `7976ff77f6f756b06e80cf32df0b2efd0d5bc58ca681da2801d9a559ca7a23a0`.
Every listed member was read back and hash-verified. The earlier v1 archive
remains separate. Formatting, final typing and review were also observed at the
terminal; no broader full-suite, installed/native, quality or pilot claim follows.

Together with 7c6dd114 sleep settings, this incorporates the two missing repairs
identified in the original 0.7 inventory. A final delivery artifact still needs
its own build/acceptance evidence. No module default was enabled, no package was
published, and the original semantic and complete-latency requirements remain.
