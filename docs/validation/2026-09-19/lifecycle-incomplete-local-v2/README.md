# Local lifecycle incomplete evidence

This archive preserves the failed seven-profile lifecycle matrix from the local
real-model run. The terminal queue status, phase logs, supplemental supervisor
state, and the seven lifecycle report/raw-output pairs are copied from their
original paths. The lifecycle pairs are stored byte-for-byte in
`lifecycle/lifecycle-reports.zip`; `lifecycle/zip-members.json` records each
member's byte count and SHA-256.

The run executed against the pinned source, Qwen, and Nomic bindings recorded
in `summary.json`. Baseline and `ablate-sleep_consolidation` completed 600 of
600 observations. The other five profiles each contain partial observations:
affective routing 10, mutable RAG 12, spreading activation 9, surprisal
gating 8, and full 11. Every partial row has one `missing_accepted_target`
correction error after the preceding automatic capture produced
`no_durable_facts`; there were no provider timeout or mapping-invalid errors.

Completed execution is not a quality or lifecycle acceptance claim. The
reports retain their original incomplete status and all quality gates remain
open. The failed observations are concentrated in a shared correction path;
retrying only those rows would change the profile/session distribution and
would bias comparisons. A future run must use a fresh output/work binding,
preserve the failed evidence, and establish a deliberate retry policy before
any model execution.

The supplemental comparisons did not run: their supervisor stopped after the
main coordinator exited unsuccessfully. No checkpoint databases or private
transcripts are copied here.
