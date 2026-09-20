# Hermes boundary outbox delivered through the current local service

The exact valid capture from the v28 actual CLI session-boundary probe reaches
the real authenticated loopback service and persists once in private SQLite.
The queued identity, content, and original session attribution remain intact.
All 55 imported Mnemos Python modules match the verified 05c71d2 wheel.

After service shutdown, the same saved entry models an acknowledgement lost
after commit. Replay while offline keeps it pending. Restart reopens the stored
event, acknowledges the queued replay, and leaves the event and job identities
unchanged with no duplicate event. Both service instances stop and remove their
private descriptor. The original boundary outbox remains byte-for-byte intact.

This extends the previous outbox-only evidence to raw delivery, restart, and
duplicate replay. All five cognition modules are disabled, the LLM provider is
unavailable, and the configured embedder is SimpleEmbeddingProvider. There is
no extraction, semantic quality, full interactive CLI, or complete capture
claim. The transport uses the real service owner and dispatcher, with an
explicit private dispatcher factory; background maintenance is not started.

Two failed harness attempts remain preserved. V29 registered a new random
project ID instead of restoring the queued scope identity and retained the
outbox without storing an event. V30 restored the identity and delivered the
event, then its verifier looked up session_id at the event top level instead
of event.metadata. V31 corrects that field path and completes all three phases.
All three SQLite backups pass integrity and foreign-key checks.
