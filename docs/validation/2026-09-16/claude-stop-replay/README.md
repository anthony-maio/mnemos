# Claude Stop replay duplication

A local SQLite reproduction confirmed a capture defect in source `20f271d`.
The first Stop read a user and assistant message from a complete JSONL
transcript and saved the byte offset. After reopening SQLite and constructing
a new adapter with the same checkpoint, replaying the identical Stop payload
added a second assistant event. Its source reference was
`last_assistant_message`; the original used the transcript's native UUID.

The adapter checked fallback text against only the newly read transcript
deltas. On replay there were no new deltas, so the previously captured
assistant message was treated as missing and captured under a different
identity. The event count increased from two to three. A degraded diagnostic
did not prevent that duplicate write.

This used the actual local service and SQLite, constant fixture vectors, and
unavailable cognition. It is an adapter/service correctness result, not a
native-host or extraction-quality result. The native persistent-transcript
attempt failed earlier at a timeout and did not reach this replay check.

The actual checkout was `ac42a68`, whose tracked runtime and tests are
identical to `20f271d`; intervening changes were documentation. The source
binding, exact probe, and original report are preserved. Its private fixture
remains available for diagnosis. A repair must preserve accepted transcript
identity across restart while keeping genuinely new transcript UUIDs eligible,
even when their content repeats. Missing native turn identity must remain an
explicit limitation when a delayed transcript makes two turns ambiguous.
