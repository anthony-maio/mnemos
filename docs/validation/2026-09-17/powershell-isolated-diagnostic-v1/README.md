# Isolated PowerShell bootstrap diagnostic

This archive records one targeted run against the frozen source checkout at commit `33c6dd90128d8762053e0bf58862e4bc52484bab`.

The diagnostic first ran a no-model PowerShell `-NoProfile -NonInteractive` stdin control, then ran the exact test node `tests/test_hosts.py::test_codex_powershell_bootstrap_executes_allowlisted_env_and_stdin` once. The test kept its source-defined 10 second child-process timeout unchanged and used a fresh external pytest fixture directory.

Both checks passed. The PowerShell control completed in 612.715 ms. The targeted test completed in 1.687 s; its JUnit report records one test passed, with no failures or skips. The fixture hook output was present and contained the expected profile, registry path, event ID, and prompt values.

The preceding full package check at `C:/Users/Ant/AppData/Local/Temp/mnemos-package-checks-33c6dd9-v1` recorded 897 passed, 1 failed, and 4 skipped. Its failure was the same test timing out after 10 seconds while waiting for the generated `pwsh -NoProfile -Command` child. It had no assertion result and no hook fixture output. The fresh pass does not identify why that earlier run timed out. It supports an environment or launch interaction hypothesis only.

The archive excludes the pytest fixture and temporary directories. It contains the executed runner copy, raw test stdout and stderr, JUnit XML, report metadata, and the post-run provenance binding. No model, network, product, or live configuration operation was used.
