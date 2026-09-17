# Scope package verification failure v1

This archive preserves the terminal package verification run bound to source
commit `d58e082f9a2cec1c88d50504603cc0db43ca8a96`.

The run completed 933 pytest cases: 929 passed and 4 were skipped. Black passed.
The Python 3.10 mypy check failed with 7 errors, all from the historical
archived test `docs/validation/2026-09-17/startup-exit-regression-v1/test_client_startup_failure.py`.
Later checks were not run. The verification status is therefore `failed` even
though the pytest and Black checks passed.

The archive contains the verification report, raw pytest/Black/mypy logs, the
JUnit XML, and the executed checkpoint helper as `final_checkpoint_checks_v3.py.txt`.
Databases and source snapshots are excluded. The source checkout, temporary
fixture path, commands, timestamps, and runner hash remain in `verification.json`.

This is a historical terminal verification artifact. It does not establish a
new package or source verification after the archived mypy failure, and it does
not include a later repair or rerun.
