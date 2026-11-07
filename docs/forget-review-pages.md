# Paged related-forget review

The service-backed MCP tools keep large forget previews inline by returning
pages. Each serialized response, including a JSON string wrapper, is at most
12,000 UTF-8 bytes. This is a transport bound, not a measured host token count.
Small previews keep the existing object structure. CLI, UI and direct Hermes
tool responses retain their existing service preview contract.

Start with `mnemos_forget_preview` using the target fact ID and version. If the
response contains `review.next_cursor`, call the same tool with that cursor and
the same fact ID, version and limit. Read every page in order. A changed snapshot
invalidates the cursor; restart the review rather than combining snapshots.

Pages use `review.format=json_fragments_v1`. Each fragment has a `path` into
the original preview and either its complete `value`, or a `text` fragment with
`offset` and `total_characters`. Object keys and zero-based array indexes form
the path. Text offsets count Unicode characters, not encoded bytes. Empty
objects, empty arrays, nulls, original wording and all qualifiers are retained.
Text is neither summarized nor silently clipped.

The page metadata identifies the first fragment, total fragment count, snapshot
digest, next cursor and whether this is the last page. `complete` continues to
describe the storage search limit; it does not mean every page has been read.
Only the final page exposes `apply_preconditions`, including inside fragments.
An incomplete storage preview never provides those preconditions. A cursor is
a read position, not authorization or proof that a person reviewed the text.

Each candidate has a short `selection_ref`, such as `f1`. Each eligible
additional source has one such as `s1`; blocked sources have null refs. A
candidate's `required_source_refs` identifies the source selections it requires.
A null required ref means that source is unavailable for short selection in
this snapshot. Keep the underlying IDs for inspection and provenance.

After reviewing the target, selected candidates and required sources, copy
`apply_preconditions` unchanged into `mnemos_forget_related`. Pass the chosen
refs in `selected_fact_refs` and `selected_source_refs`. The existing ID fields
remain available; do not mix IDs and refs within one selection type. Nothing
selects required sources automatically. The target's linked cleanup is already
covered and has no additional-source ref.

The MCP tool resolves refs against a freshly read preview with the requested
digest, then submits ordinary IDs to the existing service. Storage rechecks
the digest, version, scope and source dependencies inside its write transaction.
A change between reference resolution and mutation rejects the request. Refs
never bypass blocked-source, shared-source or incomplete-preview protections.

Inspect the returned deletion coverage. A committed subset can leave exact
copies, advisory candidates and source events behind. Zero retained counts
still do not establish exhaustive semantic erasure.

The page and alias contract requires new native acceptance evidence. Existing
single-preview traces and historical native passes do not prove that a host
can complete a paged review. Total context size and call count may increase;
this change establishes bounded delivery, not a compression or latency gain.
