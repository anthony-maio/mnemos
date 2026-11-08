# Installed transaction and plugin candidate

The combined candidate `1fb477fa2a55a1b4d0fedfe587d82dcc60548290` builds and
passes 11 installed storage checks plus three automated MCP stdio roundtrips.
These checks run outside the source checkout on Windows Python 3.11.13.

The private wheel and source archive retain the branch's `0.6.0` version.
They are verification artifacts and must not be uploaded as a new release.
The [receipt](storage-pr7-runtime-package-v1-receipt.json) identifies their hashes.

## Artifact and installation checks

Isolated Hatchling 1.33.0 builds the source archive and then builds the wheel
from that archive. Twine accepts both artifacts. All 35 package files match
between the candidate checkout, source archive, wheel and installed target.
Wheel RECORD hashes and sizes pass. The exact external plugin wrapper, sample
configs and MCP test file also match their source archive entries.

The wheel is installed with `uv pip install --no-deps --target` into a new
external directory. Tests use the existing isolated legacy dependency environment,
whose resolved versions are retained. Its 35 pre-existing package files remain
unchanged. This is not a fresh dependency-resolution test or a Python matrix.

Unchanged tests copied outside the checkout verify stale-source deletion by
another process, failed graph replacement rollback, caller-owned transactions,
caller commit and rollback with pending full-text edits, and access updates
without unnecessary FTS reindexing. All 11 pass. Loaded package modules are
verified to originate in the new installed target.

Three further tests exercise the generic stdio configuration, Claude Desktop
sample configuration and Claude plugin manifest using that installed target.
A subprocess import probe verifies the installed origin. The consumer directory
contains no source `mnemos` package. Its exact copied wrapper resolves the explicit
Python command correctly. Mock LLM and simple embeddings keep these integration
checks local; each test uses a fresh SQLite database. A disposable home keeps
sample-config expansion away from normal user settings. All three pass.

The plugin wrapper is outside the wheel; its source archive file participates
in these checks. This verifies the combined distribution surfaces without
claiming that installing the wheel installs a Claude plugin or enables it.

## Evidence and limits

An independent XML audit verifies the 11 storage and three MCP cases, with no
failures, errors or skips. The private archive retains both artifacts, helpers,
build and test logs, JUnit records, installed module origins and dependency
versions. Disposable test databases remain in the external test directory.
The build and test processes exited zero. The candidate worktree is clean.

The prior complete source-suite result is 325 passed on the same Python version.
No production benchmark, hosted CI, new full typing check or interactive native
host acceptance was run in this increment. Earlier typing and benchmark failures
retain their original status. Nothing was tagged, pushed, published or activated.
Canonical delivery and the frozen pilot installation remain unchanged. Semantic
quality, complete-command latency and actual daily-use requirements remain open.
