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

The later provisional-capture implementation durably retains a missing-ID
fallback with identity status `provisional_unassociated`, event status
`deferred`, and a `capture_only` job. Maintenance completes that job without
creating facts, revisions, or fact-evidence links from it. It does not promote
an ambiguous fallback into extraction. Completed deferred raw events can
expire under the processed-evidence retention policy; an unfinished job
prevents that pruning.

The [native v8 record](provisional-native-claude-v8/README.md), using installed
source `a38ee70` and Claude Code 2.1.191, observed two normal transcript events
and one deferred fallback. Restart and Stop replay preserved those identities.
Its original exact-two-raw-events gate remains failed; the separate provisional
identity contract passed. This proves bounded capture and identity handling,
not extraction quality or full native correction/forget continuity. The run
supplied explicit disposable settings and does not prove plugin auto-discovery.

The later [loader metadata check](claude-plugin-loader-v1/README.md) observed
the plugin and all five hook names through Claude's `--plugin-dir` metadata
commands. It did not execute hooks or connect MCP, so native plugin hook
execution remains open.

A separate newer-version host fixture could establish which native prompt
associations are available without changing live host settings. No observation
of the installed 2.1.191 fixture establishes that it emits later documented
fields.
