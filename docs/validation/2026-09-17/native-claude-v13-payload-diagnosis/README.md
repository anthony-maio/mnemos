# v13 native payload diagnosis

This is an offline derivation from the preserved v13 report and its private
native-correct JSONL stream. The raw stream remains outside the repository;
this archive records its path and SHA256 only.

The executed v13 report was `failed`. Its original parser found the three
allowed calls and matching result IDs, but it did not inspect the actual
operation payloads. Claude emitted each `tool_result.content` as a JSON
string containing an outer `{"result": "<JSON>"}` value. The v14 parser
decoded those layers and recovered one target fact at version 2. Its content
hash exactly matches the replacement hash recorded in the v13 seed metadata.
The derived phase1 contract is therefore all true: correction was queued,
consolidation completed, and retrieval returned the corrected project fact.

That does not promote the lifecycle. The same stream contains successful hook
responses whose embedded `stderr` and `output` fields report `status=degraded`
for UserPromptSubmit and Stop. Those are duplicate field observations of the
two event diagnostics, not four independent hook invocations. SessionStart
has no degraded diagnostic. The derived hook contract remains false, and the
overall lifecycle remains `derived_not_promoted`.

The archive makes no model-quality, extraction-quality, or automatic-capture
claim. It does not establish why the hooks degraded. The next native helper
must keep this condition visible while using the corrected nested-result
parser.

The copied runner is the diagnosis helper, not the executed native runner.
The executed v13 runner remains separately preserved in the v13 archive and
is bound here by its recorded SHA256.
