# Actual session boundary repair and broader regressions

The v11 actual /new probe passes uncommitted-message persistence and capture,
plus corrupted-evidence suppression. It exercises actual handler, database,
and memory provider methods with an isolated home and verified installed
Mnemos modules. It observes the capture outbox, not service delivery or a
complete interactive session. Nine targeted boundary tests also pass.

The subsequent 89-case boundary/persistence selection records 84 passes and
five failures. The unchanged Hermes base passes all 31 cases in the two files
containing those failures. These five failures are candidate regressions.

The v13 repair restores durable row-ID stamping for ordinary batch inserts,
while compression inputs preserve their reusable source IDs. The same 31-case
selection then records 30 passes and one failure. Compression rotation in an
unscoped session still fails because hybrid evidence requires a verified
origin. No baseline test was weakened or removed. This candidate is not ready
for installation. Both patch states and all failing results remain preserved.
