# Native Codex job guidance amendment v6

V5 remains failed. Its native script attempted to read a job ID from
consolidation's aggregate response and failed while retaining an undefined
value. The model then checked the correct store job, but the protocol requires
successful cells. Preserve that failure and its
[receipt](../validation/2026-11-06/native-codex-v5-receipt.json).

## New product artifact

Use product source `929c44a44100592dd46e52b7f77b402f84beb9e3` and the wheel
`mnemos_memory-0.7.0b1-py3-none-any.whl`, SHA-256
`298b20ec0cc7dd1c348cc55db45a4774ef979642f491fca1ade93ee511175fc9`.
It was built from its source distribution with isolated Hatchling 1.33.0.
Both source and wheel artifacts passed Twine. Require a new successful
installed smoke receipt bound to this wheel before inference.

This product change adds tool descriptions for consolidation and correction
and expands the store/job-status guidance. Consolidation returns aggregate
maintenance counts and health. The job ID comes from store or correction.
Verify that original job's completed status and result; inspect a corrected
fact's identity, content and version. The operating references agree.
Processing behavior and acceptance thresholds are unchanged.

The teach and correction prompts now explicitly name the response supplying
the job ID. This documents the real interface; it does not supply fact IDs,
versions, completion results or any evaluator answer beyond the existing
synthetic instruction. The model must still retrieve those values through
native tools. No corrective follow-up prompt is permitted.

## Unchanged acceptance

Keep the pinned native runtime, Code Mode host and model from v2-v3; the
scoped permissions and strict recorded ID correlation from v4-v5; and all six
single-prompt phases, deadlines, lifecycle checks, source replay, positive
controls and cleanup gates. Failed cells remain failures. Preserve the first
failure without retrying inside an attempt. Use a new external output root
and a fresh installed environment, commit all inputs before inference, and
retain private evidence and a public receipt. Native success would establish
this explicit synthetic workflow only, not extraction quality, automatic
capture, supported-answer quality, pilot completion or publication.
