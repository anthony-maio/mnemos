# Combined storage draft validation

[Draft PR #8](https://github.com/anthony-maio/mnemos/pull/8) now includes the existing plugin runtime and float32 typing repairs. Its head is `7c6cdccdab333161402334b31b1925b818303325`; the base remains PR #7 at `9c9ebf84`. Updates were fast-forwards. GitHub readback confirmed the head, revised title/body and draft status. No merge, tag or release occurred.

Fresh verification ran on `86acea685cebedc8c6c3e07ced3be38ac07c96a6`, combining the transaction repair with the two previously prepared fixes. Subsequent commits clarify plugin documentation and replace a test's hardcoded python alias with the current interpreter filename. Runtime is unchanged. All 17 plugin tests passed again after that test-only change; the earlier complete-suite result retains its exact source binding.

Independent review found no blocking runtime issue. The test portability correction came from that review; it does not establish a cross-platform test result. The archived PR body below records the intermediate 3b83b4b1 description; GitHub's current body includes the later test change.

| Check | Result |
| --- | --- |
| Full source suite, Windows Python 3.11.13 | 325 passed; zero failures, errors or skips |
| Strict mypy, Python 3.12.11 / mypy 2.4.1 / NumPy 2.2.6 | 31 source files pass |
| Black | 66 Python files pass |
| Imported source origins and all test-file hashes | Verified |
| Existing isolated installation | 35 package files unchanged; dependency inventory unchanged |

The complete run used the existing isolated legacy environment with its interpreter directory first on PATH and a new disposable user home. It did not install or change dependencies, use normal host configuration, or exercise a live memory database. The prior 319/320 result remains failed on the original transaction-only candidate. This new run establishes that the combined source resolves the recorded Windows plugin-launch failure.

The PR now contains eight changed files: graph transaction implementation/tests/changelog, plugin command resolution/tests/documentation, and float32 annotations. Command names resolve against the supplied PATH before launch; explicit paths and virtual-environment symlinks retain their behavior. The selected interpreter must already contain Mnemos and MCP dependencies.

The private archive `.venv/storage-pr8-combined-evidence-v1.zip` has SHA256 `2042d6fa42bb5cd7ce957619ccc530949583f4c8e1ff2a71b560de233c3da585`. Eight members retain the complete-suite log, JUnit, summary, source-origin/dependency record, exact helpers, final PR body and hash manifest. All manifest members passed readback. Mypy and Black results above are recorded from terminal output rather than a separate archived log.

The existing CI workflow only runs for master pushes/PRs and has no workflow_dispatch trigger on this source. No hosted CI, new benchmark, installed-package refresh, interactive host validation or actual pilot day is claimed. After prerequisite #7 is merged, retarget this draft, inspect the resulting diff and complete the required validation before any merge/release decision. All original goal gates remain unchanged.
