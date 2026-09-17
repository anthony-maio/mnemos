# Lost-ack package failure checkpoint

This archive preserves the terminal source-check attempt for commit `6cde46561c0c42c0d6124d86a09d1e2832542c9d`.

The run covered 922 pytest cases: 917 passed, 4 were skipped, and 1 failed. Black, mypy, JavaScript, build, and Twine checks did not run because the source pytest check stopped the checkpoint runner after the failure. The failing case was `tests/test_service_observability.py::test_autostarted_service_logs_real_retrieval_without_console`; its autostart health response was `degraded` where the test expected `running`.

The raw verification JSON, pytest log, JUnit XML, and the exact checkpoint runner bytes are preserved. The runner hash recorded by verification matches the archived runner. No wheel was built or installed by this failed checkpoint, so installed-package checks and the v9 lost-ack contract run are NOT RUN. This archive does not diagnose the degraded startup response or establish a product fix. The separate package archive helper remains unexecuted.
