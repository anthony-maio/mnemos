# Hermes reviewed-copy tool checkpoint

Frozen source `db932befcd4d0aac9220f9f1e73f043ccd00093c` passes
1,383 source tests with 4 skips, all six source/package checks,
595 installed cases with 4 matching skips, and seven replay assertions.
The isolated installed run checks each selected case and outcome, imported
module origins, and package bytes against the frozen source and built wheel.

Wheel: 300,664 bytes; SHA-256
`7df060463b35853d97898ca839b500a66e233bef26b8e52a7de1db799681190a`.

## Change and evidence

Hermes now exposes the existing service's `mnemos_forget_preview` and
`mnemos_forget_related` APIs, bringing provider discovery from six to eight
tools. Both receive context from the initialized provider. Unknown arguments
and caller-supplied context, project, profile, workspace, or allowed scopes
are rejected. Digest, version, limit, and selected-ID fields are validated.
Neither operation queues requests for later replay after a service failure.

The read-only preview supplies a digest for reviewed state. Apply requires
that digest and the matching limit. The target is always removed; omitted
or empty selections remove no additional facts or sources. Existing storage
checks still reject stale/incomplete previews, out-of-preview selections,
cross-scope candidates, and unsafe source dependencies atomically.

New provider/service fixtures exercise selected copy removal, target-only
defaults, a preview invalidated by a later capture, foreign project/profile
exclusion, out-of-preview rejection, and retention of unselected advisory
similarity candidates. The development baseline log has 12 passes. The log
named red-v2 has two failures before exposure: missing tool discovery and a
valid preview rejected as unknown. The final focused log has 22 passes;
the broader source and installed reports include the additional scope cases.
Development logs are supplementary; the frozen reports bind the final tests.

## Limits and next native work

This checkpoint is fixture/service and installed-package evidence. There has
been no native model run using the two new tools. The [v5 cohort](../hermes-native-cohort-v5/README.md)
remains zero of three complete passes on its earlier `fd83fb2` runtime.
Two attempts retained older conversational copies; one model described a
delete without making the call. The three-host prerequisite remains unmet.

A complete preview means the returned selection and source dependencies fit
the bound. Similarity search remains advisory and nonexhaustive; complete
does not establish that every semantic copy was found. External transcripts,
arbitrary paraphrases, and Hermes' built-in memory store remain outside this
bounded deletion claim. Guidance and tests preserve this distinction.

A prospective native successor must discover eight tools and correlate the
model's preview and selected deletion with their actual results. Preserve
the frozen v5 inputs and all three failures. Do not lower lifecycle gates or
substitute controller mutations for model calls. Three-host continuity,
complete capture, model-quality acceptance, Desktop, and the pilot remain open.

The evidence index binds 36 files. This README is interpretive and
outside that raw index. Live host configuration and memory were unchanged.
