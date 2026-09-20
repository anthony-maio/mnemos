# Frozen dataset Windows checkout repair

The first full source run for copyable forget preconditions finished with
1,706 passes, 15 failures, and four skips. All failures were in embedding-prefix
and role-aware preparation/runner tests. Its clean Windows checkout converted the frozen
source-association JSON from LF to CRLF. The strict hash guard correctly
rejected the changed bytes before scoring, even though normalized content
was identical. The full failing run is retained here; package checks did not
continue after pytest failed.

The repair marks only this dataset `-text` in Git attributes. The dataset,
its original SHA-256, and the fail-closed hash check are unchanged. A second
fresh checkout with `core.autocrlf=true` preserves the original bytes; all 21
prefix, role-preparation, and role-runner tests pass there. Full verification
of that successor is recorded separately and is not inferred from 21 tests.

`portability-proof.json` binds both source commits, hashes, line-ending counts,
and failing cases. The index binds the failed source report, its complete
JUnit/log, the focused successor JUnit, and this archive helper. No model calls
or evaluation criteria changed. Previous measured negative results remain.
