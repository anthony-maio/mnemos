# Hook context authority validation archive

Terminal validation status: failed.
Base commit: 7443582b23466cfabd9253448020350d727f85f2.
The archive preserves the exact runner report, selected modified source and test files, tracked patch, and check outputs that existed at terminal time.

Black: 0.
Mypy: 0.
Pytest: 1.
JUnit totals: {"tests": 986, "failures": 1, "errors": 0, "skipped": 4, "passed": 981}.

Native Codex and Claude acceptance was NOT RUN by this validation. The original native continuity v6 failure remains a separate result.
A terminal failed status is preserved and is not converted into a pass.
The format-attempt files preserve the earlier run that stopped at Black before mypy or pytest. The CLI test was formatted before the v2 run.
