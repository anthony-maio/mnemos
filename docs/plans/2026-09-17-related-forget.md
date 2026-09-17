# Preview and remove matching captured copies

The frozen `9e25339` v2 overlap probe captures a prompt containing the exact
remembered assertion, corrects the explicit fact, and forgets it. The original
assertion remains retrievable in the independently captured prompt after a
restart. This occurs with `modules.sleep=false`, which promotes raw captured
content. The product defaults have all five module switches enabled; the
default `baseline` retrieval profile is a separate setting. The earlier v1
probe used unrelated inputs and does not demonstrate the defect.

Add a separate, previewed operation for matching captured copies. Keep existing
single-record forget behavior and response compatibility. The working design
uses exact normalized assertion text and explicit selection of additional
records. It does not infer semantic equivalence, remove paraphrases, rewrite
historical backups, or broaden deletion to another project or profile.

## Behavior

`forget_preview` accepts a validated context, a project-scoped fact ID, and its
expected version. It returns the target plus additional captured facts and
source events containing the full normalized current or historical assertion.
Independent copies must be discoverable without sharing the target's evidence
IDs. Restrict additional fact candidates to service-produced capture facts
(`source=sleep_consolidation`), and source candidates to non-explicit capture.
Preserve unrelated manual facts even when their text overlaps.

The preview includes content for inspection, reasons, IDs, versions, affected
source/job IDs, and a digest of the relevant state. It is read-only and carries
an explicit exact-text matching limitation. Include pending captured events so
later maintenance cannot recreate a selected copy. Bound the returned candidate
and source lists, including linked sources; an incomplete preview cannot be
applied as if complete. This output bound is not a database scan-time guarantee.

`forget_related` accepts that digest and an explicit selection from the
preview. The target is mandatory. Additional deletions must be visible before
apply. If choosing a source would leave a matching derived fact alive, reject
the inconsistent selection or require the matching dependency in the preview.
Do not silently add unreviewed facts to the deletion set.

Apply revalidates scope, versions, matching, candidate membership, and digest
inside one SQLite write transaction. A changed target, changed copy, or newly
matching capture makes the preview stale and causes no deletion. Reuse the
existing forget cleanup for each selected fact, including current/revision
canonical suppressions, source identity suppression, jobs, support excerpts,
feedback, vectors, and graph state. Selected pending sources also receive
content-free replay suppression. Keep suppression rows free of assertion text.

## Implementation sequence

1. Add typed preview/apply requests in `mnemos/contracts.py`, SQLite preview and
   atomic apply in `mnemos/utils/storage.py`, and scoped dispatch/cache cleanup
   in `mnemos/memory_service.py`. Factor the existing forget transaction body
   into a private helper only as needed; preserve `forget_fact` compatibility.
   No live database or schema migration is needed for the initial operation.
2. Expose preview/apply through MCP and the control-plane API. Add a memory UI
   preview that shows additional records, requires explicit selection, and
   reports conflicts without retrying a changed deletion plan automatically.
   Keep the single-record action available. Implement against the reviewed
   backend response contract rather than inventing a second matching policy.
3. Reproduce v2 with preview/apply, maintenance, replay, and restart. Verify the
   selected assertion is absent while the v1 unrelated record and other
   projects/profiles remain intact. Add failure injection proving rollback of
   the whole operation, stale-plan tests, pending-job and inflight extraction
   tests, selection tampering tests, and assertions that suppression records
   contain no plaintext. Run existing forget, correction, backup, graph, and
   scope tests because they share the transaction path.
4. Integrate the UI/MCP path, freeze a new revision, and repeat full package and
   installed-wheel validation. Update the native lifecycle runner to exercise
   the actual preview/apply protocol. Preserve the old failure and keep native
   acceptance open until observed. Do not disable capture or change extraction
   flags merely to hide the overlap.

The initial delivery promises removal of the explicitly selected records and
their replayable sources. Unseen future restatements, paraphrases, and external
transcripts remain outside this operation's guarantee.
