# Claude host identity boundary

Checked September 17, 2026. The native fixture used Claude Code 2.1.191 and
observed no `prompt_id` in its Stop payload. That is recorded in the
[native source20f report](native-claude-scope-v4/report.json).

Current official documentation introduces common hook `prompt_id` in version
2.1.196. It also describes asynchronous transcript writes, which can leave the
latest reply unavailable when Stop runs. Stop supplies its own final response
text. These are current documented contracts, not fields observed in the
installed 2.1.191 fixture.
[Hook input reference](https://code.claude.com/docs/en/hooks#common-input-fields),
[Stop input](https://code.claude.com/docs/en/hooks#stop-input).

MessageDisplay provides stable display-message and turn identifiers, but the
display-message identifier cannot be joined to transcript identifiers. The
adapter therefore must not equate those namespaces.
[MessageDisplay input](https://code.claude.com/docs/en/hooks#messagedisplay-input).

Transcript JSONL is an internal format that can change across releases. Any
association based on a transcript field needs fixture evidence for the host
version under test.
[Transcript storage](https://code.claude.com/docs/en/sessions#where-transcripts-are-stored).

The repaired adapter accepts an explicit matching transcript prompt reference
when present. Its tests use declared synthetic associations; they do not prove
that 2.1.191 emits those fields. Without association, it preserves native UUID
observations and reports ambiguity instead of discarding repeated text.

A stronger design would durably retain an unassociated fallback as pending
evidence, with separate promotion and expiry behavior. That would require
service/storage changes, replay and forget tests, and an explicit distinction
between captured evidence and accepted facts. It is not implemented by the
current replay repair. A separate newer-version host fixture could establish
which native associations are available without changing live host settings.
