# Strict typing for the combined delivery candidate

Candidate `86acea685cebedc8c6c3e07ced3be38ac07c96a6` resolves the three strict typing errors reproduced on
`1fb477fa`. It adds `NDArray[np.float32]` to the simple embedding provider's
projection dictionary and two array-returning methods. The arrays already use
float32. No calculation, configuration, test or benchmark rule changes.

On Windows Python 3.12.11, mypy 2.4.1 passes all 31 source files, Black 26.10.1
passes all 66 Python files, and all six existing embedding tests pass with
NumPy 2.2.6 and pytest 9.2.0. The test log retains an existing pytest-asyncio
fixture-scope deprecation warning. No test failed, errored or skipped.

A syntax-tree comparison removes only annotations and the added numpy.typing
import, then verifies that the old and new modules match. Only embeddings.py
changes from the previous candidate. The new delivery worktree is clean; the
previous candidate and its installed verification artifacts remain intact.

The [receipt](storage-pr7-delivery-typing-v1-receipt.json) identifies the source,
tool versions and private archive. The archive retains the original three
errors, passing checks, six JUnit records, candidate source and annotation audit.
The earlier unsuppressed errors remain recorded against their original source.

This does not transfer the previous 325-test source run or 14 installed checks
to a new artifact. Those results remain bound to `1fb477fa`; this increment
runs the checks appropriate to an annotation-only change. No new wheel, hosted
CI, Python matrix, benchmark or native-host acceptance is claimed. Version
metadata remains 0.6.0, so release version preparation is still required.
Nothing was pushed, tagged, published or activated. The original semantic,
complete-command latency and daily-use pilot requirements remain open.
